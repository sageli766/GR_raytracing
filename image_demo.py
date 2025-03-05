import matplotlib.pyplot as plt
from PIL import Image
from schwarzschild_metric import *
import concurrent.futures
from tqdm import tqdm

image_path = r'horizontal_bars_50px.png'
background = np.array(Image.open(image_path))
# background = background[270:370, 540:740]
# img = Image.fromarray(background, 'RGB')
# img.show()
# print("Background image shape:", background.shape)

n = background.shape[0]
m = background.shape[1]

# step size
h = 0.01

M = 1
r_horizon = 2 * M
x_obs = 50
x_img = -50

def process_row(i):
    """Compute one row of the final image."""
    row_result = np.zeros((m, 3), dtype=background.dtype)
    for j in range(m):
        x0 = i - n // 2
        y0 = j - m // 2

        # save on some computation time- light rays originating within the event horizon will eventually fall into it
        if np.linalg.norm([x0, y0]) <= r_horizon:
            row_result[j] = [0, 0, 0]
        else:
            state_0, u, v, E, L = init_conditions(x0, y0, x_obs, M)
            final_pos = integrate_ray(state_0, M, E, L, h, x_img, u, v, keep_path=False)
            if type(final_pos) != np.ndarray:
                row_result[j] = [0, 0, 0]
            else:
                rounded_y = int(np.round(final_pos[1], 0))
                rounded_z = int(np.round(final_pos[2], 0))
                y_index = rounded_y + n // 2 - 1
                z_index = rounded_z + m // 2 - 1

                if 0 <= y_index < n and 0 <= z_index < m:
                    row_result[j] = background[y_index, z_index]
                else:
                    row_result[j] = [0, 0, 0]
    return row_result


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Wrap the iterator with tqdm; 'total=n' gives the number of rows.
        results = list(tqdm(executor.map(process_row, range(n)), total=n, desc="Rendering"))

    image = np.stack(results, axis=0)

    img = Image.fromarray(image, 'RGB')
    img.show()
    img.save(f'figures/{image_path[:-3]}_blackhole.png')
