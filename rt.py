import matplotlib.pyplot as plt
from PIL import Image
from schwarzschild_metric import *

image_path = r'milkyway_ref.png'
background = np.array(Image.open(image_path))
print("Background image shape:", background.shape)

# Ray position in Cartesian (x_obs, y0, z0):
final_image = np.zeros_like(background)

h = 0.01  # step size
mass = 1
r_horizon = 2 * mass
x_obs = 50
x_img = -50

state_0, u, v, E, L = init_conditions(10, 10, x_obs, mass)
traj2d, traj3d = integrate_ray(state_0, mass, E, L, h, x_img, u, v)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(traj3d[:, 0], traj3d[:, 1], traj3d[:, 2], label='Photon Trajectory')
ax.scatter(0, 0, 0, color='black', s=100, label='Black Hole')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-50, 50)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)
ax.set_title('3D Photon Trajectory')
ax.legend()

# Projection onto the x-y plane
ax2 = fig.add_subplot(122)
ax2.plot(traj3d[:, 0], traj3d[:, 1], label='Trajectory (x-y)')
ax2.axvline(x=x_obs, color='blue', linestyle=':', label='Observer Plane')
ax2.axvline(x=x_img, color='green', linestyle=':', label='Image Plane')
ax2.scatter(0, 0, c = 'black', s=100, label='Black Hole')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(-55, 55)
ax2.set_ylim(-20, 20)
ax2.set_title('Projection onto x-y Plane')
ax2.legend()
plt.savefig('trajectory.png')
plt.show()

final_pos = traj3d[-1]
if traj2d[-1, 1] <= r_horizon:
    print("The ray fell into the black hole")
else:
    print("The ray hit the image plane at:")
    print("x = {:.2f}, y = {:.2f}, z = {:.2f}".format(final_pos[0], final_pos[1], final_pos[2]))
