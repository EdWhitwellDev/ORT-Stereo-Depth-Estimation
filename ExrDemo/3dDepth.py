import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


max_depth = 20
min_depth = 0.1
depth_map = np.load('normalized_img.npy')

size = depth_map.shape
print(size)
X = np.arange(0, size[1], 1)
Y = np.arange(0, size[0], 1)
X, Y = np.meshgrid(X, Y)

Z = depth_map
# set the maximum depth value to 30
Z[Z > max_depth] = max_depth
# where the depth value is 0, set it to the maximum value for that x coordinate
for i in range(size[0]):
    xline = Z[i, :]
    highest = np.max(xline)
    xline[xline <= min_depth] = highest
    Z[i, :] = xline

# Plot the depth map as a 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='cool', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Labels and titles
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Depth')
ax.set_title('3D Depth Map')

plt.show()