import numpy as np
import numba
from numba import jit, njit

def init_conditions(y0, z0, x_obs, M):
    '''
    Determine initial conditions for numerical integration
    :param y0: starting y position on the plane of the observer
    :param z0: starting z position on the plane of the observer
    :param x_obs: x position of observing plane
    :param x_img: x position of image plane
    :param M: mass of schwarzschild black hole
    :return: tuple of (state0, u, v, E, L) where state0 is the initial state of the vector in the state space,
             u and v are basis vectors of the state space, E and L are energy and angular momentum, respectively.
    '''

    obs_pos = np.array([x_obs, y0, z0])
    r0 = (np.linalg.norm(obs_pos))

    # initial direction of each ray
    ray_dir = np.array([-1.0, 0.0, 0.0])
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # The geodesic lies in the plane spanned by obs_pos and ray_dir, so we build an orthonormal basis for the plane
    u = obs_pos / r0  # radial unit vector
    w = np.cross(obs_pos, ray_dir)  # normal vector
    w = w / np.linalg.norm(w)
    v = np.cross(w, u) # orthogonal basis vector


    # project the ray direction onto the plane
    ray_dir_plane = ray_dir - np.dot(ray_dir, w) * w  # remove any component normal to the plane
    ray_dir_plane /= np.linalg.norm(ray_dir_plane)
    # Express the projected ray direction in the (u, v) basis:
    d1 = np.dot(ray_dir_plane, u)  # component along u (radial)
    d2 = np.dot(ray_dir_plane, v)  # component along v (tangential)

    #   dx'/dl = dr/dl = d1
    #   dy'/dl = r0 * dphi/dl = d2
    dr_dl_initial = d1
    dphi_dl_initial = d2 / r0

    # Conserved quantities
    E = 1.0
    L = r0 ** 2 * dphi_dl_initial

    # The null condition implies: (choosing sign depending on starting position)
    pr0 = -np.sqrt(E ** 2 - (L ** 2 / (r0 ** 2)) * (1 - 2 * M / r0))
    if x_obs < 0:
        pr0 = -pr0

    # initial state for the 2D integration
    state0 = np.array([0.0, r0, 0.0, pr0])  # [t, r, φ, pr]

    return state0, u, v, E, L

@njit
def geodesic_derivs(state, M, E, L):
    """
    Derivatives for a Schwarzschild null geodesic in the 2D (geodesic) plane.
    state = [t, r, φ, pr]
    """
    t, r, phi, pr = state
    inner_param = 1 - 2 * M / r
    dt_dl = E / inner_param
    dr_dl = pr
    dphi_dl = L / r ** 2
    dpr_dl = (-M * E ** 2 / (r ** 2 * inner_param ** 2)
              - M * pr ** 2 / (r ** 2 * inner_param)
              + (L ** 2 * inner_param) / (r ** 3))
    return np.array([dt_dl, dr_dl, dphi_dl, dpr_dl])

@njit
def rk4_step(state, h, M, E, L):
    k1 = geodesic_derivs(state, M, E, L)
    k2 = geodesic_derivs(state + 0.5 * h * k1, M, E, L)
    k3 = geodesic_derivs(state + 0.5 * h * k2, M, E, L)
    k4 = geodesic_derivs(state + h * k3, M, E, L)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def integrate_ray(state0, M, E, L, h, x_img, u, v, max_steps=10000, keep_path=True):
    """
    Integrate the 2D geodesic in the ray's plane.
    Each 2D point (r, phi) is mapped back into 3D
    We stop when the x-coordinate of pos3d crosses the image plane (x <= x_img)
    or when r <= r_horizon.
    :returns trajectory in state space and 3-position space if keep_path is True, else returns final 3-position location.
    """
    if keep_path:
        traj_2d = [state0.copy()]
        traj_3d = []

    state = state0.copy()

    # Map initial 2D point to 3D
    r_current, phi_current = state[1], state[2]
    pos3d = r_current * np.cos(phi_current) * u + r_current * np.sin(phi_current) * v

    for _ in range(max_steps):
        state = rk4_step(state, h, M, E, L)
        r_current, phi_current = state[1], state[2]
        pos3d = r_current * np.cos(phi_current) * u + r_current * np.sin(phi_current) * v
        if keep_path:
            traj_2d.append(state.copy())
            traj_3d.append(pos3d)

        # Ray hits image
        if pos3d[0] <= x_img:
            break
        r_horizon = 2 * M

        # Ray falls into black hole
        if r_current <= r_horizon * 1.05:
            pos3d = None
            if keep_path:
                return None
            else:
                break

    if pos3d is not None:
        if pos3d[0] >= x_img + 1:
            pos3d = None

    if keep_path:
        return np.array(traj_2d), np.array(traj_3d)
    else:
        return pos3d