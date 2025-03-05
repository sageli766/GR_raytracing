from numba import cuda, float64, int32
import math
import numpy as np
from PIL import Image


# ------------------------------
# Device Functions
# ------------------------------

@cuda.jit(device=True)
def init_conditions_device(y0, z0, x_obs, M, state0, u, v):
    """
    Compute initial conditions for the ray in the geodesic plane.
    The observer is located at (x_obs, y0, z0) and the fixed ray direction is (-1, 0, 0).
    This function computes the initial state vector (state0 = [t, r, phi, pr])
    and the basis vectors u and v defining the ray's plane.
    Returns the energy E and angular momentum L.
    """
    # observer 3D position
    r0 = math.sqrt(x_obs * x_obs + y0 * y0 + z0 * z0)

    # initial state
    state0[0] = 0.0
    state0[1] = r0
    state0[2] = 0.0

    # u: radial unit vector = obs_pos / r0.
    u[0] = x_obs / r0
    u[1] = y0 / r0
    u[2] = z0 / r0

    # w = cross(obs_pos, ray_dir) = cross([x_obs, y0, z0], [-1,0,0]) = [0, -z0, y0]
    wx = 0.0
    wy = -z0
    wz = y0
    w_norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if w_norm > 0.0:
        w0 = wx / w_norm
        w1 = wy / w_norm
        w2 = wz / w_norm
    else:
        w0 = 0.0;
        w1 = 0.0;
        w2 = 0.0

    # v = cross(w, u) gives the second basis vector in the plane.
    v[0] = w1 * u[2] - w2 * u[1]
    v[1] = w2 * u[0] - w0 * u[2]
    v[2] = w0 * u[1] - w1 * u[0]

    # project the fixed ray direction (-1,0,0) onto the plane orthogonal to w
    dot_val = -w0
    ray_dir_plane0 = -1.0 - dot_val * w0
    ray_dir_plane1 = 0.0 - dot_val * w1
    ray_dir_plane2 = 0.0 - dot_val * w2
    norm_rdp = math.sqrt(ray_dir_plane0 * ray_dir_plane0 +
                         ray_dir_plane1 * ray_dir_plane1 +
                         ray_dir_plane2 * ray_dir_plane2)
    if norm_rdp > 0.0:
        ray_dir_plane0 /= norm_rdp
        ray_dir_plane1 /= norm_rdp
        ray_dir_plane2 /= norm_rdp

    # express the projected ray direction in the (u, v) basis
    d1 = ray_dir_plane0 * u[0] + ray_dir_plane1 * u[1] + ray_dir_plane2 * u[2]
    d2 = ray_dir_plane0 * v[0] + ray_dir_plane1 * v[1] + ray_dir_plane2 * v[2]

    # polar derivatives
    dr_dl_initial = d1
    dphi_dl_initial = d2 / r0

    # conserved constants
    E_val = 1.0
    L_val = r0 * r0 * dphi_dl_initial

    # Use the null condition: pr^2 = E^2 - (L^2/r^2)*(1 - 2M/r)
    tmp = E_val * E_val - (L_val * L_val / (r0 * r0)) * (1.0 - 2.0 * M / r0)
    if tmp > 0.0:
        pr0 = -math.sqrt(tmp)  # negative so that r decreases initially
    else:
        pr0 = 0.0
    state0[3] = pr0

    return E_val, L_val


@cuda.jit(device=True)
def geodesic_derivs_device(state, M, E, L, deriv):
    """
    Compute the derivatives of the state vector.
    state = [t, r, phi, pr]
    deriv is an output array of length 4.
    """
    t = state[0]
    r = state[1]
    phi = state[2]
    pr = state[3]
    inner = 1.0 - 2.0 * M / r
    deriv[0] = E / inner
    deriv[1] = pr
    deriv[2] = L / (r * r)
    deriv[3] = (-M * E * E / (r * r * inner * inner)
                - M * pr * pr / (r * r * inner)
                + (L * L * inner) / (r * r * r))


@cuda.jit(device=True)
def rk4_step_device(state, h, M, E, L, new_state):
    """
    rk4 step
    """
    k1 = cuda.local.array(4, dtype=float64)
    k2 = cuda.local.array(4, dtype=float64)
    k3 = cuda.local.array(4, dtype=float64)
    k4 = cuda.local.array(4, dtype=float64)
    temp = cuda.local.array(4, dtype=float64)

    geodesic_derivs_device(state, M, E, L, k1)
    for i in range(4):
        temp[i] = state[i] + 0.5 * h * k1[i]
    geodesic_derivs_device(temp, M, E, L, k2)
    for i in range(4):
        temp[i] = state[i] + 0.5 * h * k2[i]
    geodesic_derivs_device(temp, M, E, L, k3)
    for i in range(4):
        temp[i] = state[i] + h * k3[i]
    geodesic_derivs_device(temp, M, E, L, k4)
    for i in range(4):
        new_state[i] = state[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@cuda.jit(device=True)
def integrate_ray_device(state0, h, M, E, L, x_img, u, v, max_steps, final_pos):
    """
    Integrate the ray's geodesic.
    state0: initial state [t, r, phi, pr]
    u, v: basis vectors for the geodesic plane.
    x_img: x coordinate of the image plane.
    final_pos: output 3-element array for final 3D position.
    Returns 1 if the ray reaches the image plane, 0 if it falls into the black hole.
    """
    state = cuda.local.array(4, dtype=float64)
    for i in range(4):
        state[i] = state0[i]
    r_horizon = 2.0 * M
    for step in range(max_steps):
        new_state = cuda.local.array(4, dtype=float64)
        rk4_step_device(state, h, M, E, L, new_state)
        for i in range(4):
            state[i] = new_state[i]
        r_current = state[1]
        phi_current = state[2]
        # map 2D state (r, phi) back to 3D:
        pos0 = r_current * math.cos(phi_current) * u[0] + r_current * math.sin(phi_current) * v[0]
        pos1 = r_current * math.cos(phi_current) * u[1] + r_current * math.sin(phi_current) * v[1]
        pos2 = r_current * math.cos(phi_current) * u[2] + r_current * math.sin(phi_current) * v[2]
        # if the ray reaches the image plane
        if pos0 <= x_img:
            final_pos[0] = pos0
            final_pos[1] = pos1
            final_pos[2] = pos2
            return 1
        # ray falls into black hole + small margin for numerical errors
        if r_current <= r_horizon * 1.005:
            final_pos[0] = 0.0
            final_pos[1] = 0.0
            final_pos[2] = 0.0
            return 0
    # max_steps exceeded
    final_pos[0] = 0.0
    final_pos[1] = 0.0
    final_pos[2] = 0.0
    return 0


# ------------------------------
# Global Kernel
# ------------------------------

@cuda.jit
def ray_trace_kernel(output, background, n, m, h, M, x_obs, x_img, max_steps):
    """
    Each thread processes one pixel (i,j) from the output image.
    """
    i, j = cuda.grid(2)
    if i >= n or j >= m:
        return

    # convert to observer coordinates
    y0 = float(i - n // 2)
    z0 = float(j - m // 2)

    # if the ray is already in the path of the event horizon, map to black
    if math.sqrt(y0 * y0 + z0 * z0) <= 2.0 * M:
        output[i, j, 0] = 0
        output[i, j, 1] = 0
        output[i, j, 2] = 0
        return

    # allocate arrays to thread memory (no dynamic memory allocation with CUDA)
    state0 = cuda.local.array(4, dtype=float64)
    u = cuda.local.array(3, dtype=float64)
    v = cuda.local.array(3, dtype=float64)

    # compute initial conditions
    E_val, L_val = init_conditions_device(y0, z0, x_obs, M, state0, u, v)

    # integrate ray
    final_pos = cuda.local.array(3, dtype=float64)
    res = integrate_ray_device(state0, h, M, E_val, L_val, x_img, u, v, max_steps, final_pos)

    # map the final position (if valid) to background indices.
    if res == 1:
        y_index = int(final_pos[1] + n // 2 - 1)
        z_index = int(final_pos[2] + m // 2 - 1)
        if 0 <= y_index < background.shape[0] and 0 <= z_index < background.shape[1]:
            output[i, j, 0] = background[y_index, z_index, 0]
            output[i, j, 1] = background[y_index, z_index, 1]
            output[i, j, 2] = background[y_index, z_index, 2]
        else:
            output[i, j, 0] = 0
            output[i, j, 1] = 0
            output[i, j, 2] = 0
    else:
        output[i, j, 0] = 0
        output[i, j, 1] = 0
        output[i, j, 2] = 0


# ------------------------------
# Host Code to Launch Kernel
# ------------------------------

if __name__ == '__main__':
    # load background
    image_path = r'cropped_milkyway.png'
    background = np.array(Image.open(image_path))
    n = background.shape[0]
    m = background.shape[1]

    # simluation parameters
    h = 0.01
    M = 10
    x_obs = 100
    x_img = -100
    max_steps = 50000

    # allocate output array
    output = np.zeros((n, m, 3), dtype=np.uint8)

    # Copy background to device.
    d_background = cuda.to_device(background)
    d_output = cuda.to_device(output)

    # Define grid dimensions.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (m + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # launch kernel
    ray_trace_kernel[blocks_per_grid, threads_per_block](d_output, d_background, n, m, h, M, x_obs, x_img, max_steps)

    # copy back to host
    d_output.copy_to_host(output)

    img = Image.fromarray(output, 'RGB')
    img.show()
    img.save('cuda_bars.png')
