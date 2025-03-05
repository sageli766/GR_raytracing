from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

n = 100
m = 100

image = np.zeros((n, m, 3), dtype=np.uint8)

for i in range(n):
    for j in range(m):
        if i % 10 in [0, 1, 2, 3, 4]:
            image[i, j] = np.array([255, 255, 255])
        else:
            image[i, j] = np.array([100, 100, 100])

img = Image.fromarray(image)
img.show()
img.save('horizontal_bars_5px_by_100px.png')
