import OpenEXR
import Imath
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

numDisparities = 5
baseLine = 270
width = 512
fov = 47
focalLength = width / (2 * np.tan(np.radians(fov / 2)))

def read_exr_to_numpy(iteration, channels=['R']):
    # Open the EXR file
    exr_file = OpenEXR.InputFile('depth_exr/' + str(iteration).zfill(5) + 'Left.exr')

    # Get the header and the dimensions of the image
    header = exr_file.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Extract RGB channels and reshape them into 2D arrays
    channel_data = [exr_file.channel(c, Imath.PixelType(Imath.PixelType.HALF)) for c in channels]
    rgb_array = [np.frombuffer(c, dtype=np.float16).reshape(height, width) for c in channel_data]

    # Stack channels along the third dimension to get a 3D array (height x width x 3)
    img_array = np.stack(rgb_array, axis=-1)
    img_array[img_array == np.inf] = 0
    return img_array.astype(np.float16)[:, :, 0]

def depth_to_disparity(depth):
    disparity = (focalLength * baseLine) / (depth + 0.0001)
    return disparity

depths = np.empty((6000, 512, 512), np.float16)
for i in range(6000):
    depths[i] = read_exr_to_numpy(i+1)

disparities = depth_to_disparity(depths)
print("Disparities shape: ", disparities.shape)
input("Press Enter to save the disparities")
np.save('disparities.npy', disparities)