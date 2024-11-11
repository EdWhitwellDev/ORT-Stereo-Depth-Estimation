import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def read_exr_to_numpy(file_path, channels=['R']):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get the header and the dimensions of the image
    header = exr_file.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Extract RGB channels and reshape them into 2D arrays
    # print the channel names
    print(header['channels'].keys())
    channel_data = [exr_file.channel(c, Imath.PixelType(Imath.PixelType.HALF)) for c in channels]
    rgb_array = [np.frombuffer(c, dtype=np.float16).reshape(height, width) for c in channel_data]

    # Stack channels along the third dimension to get a 3D array (height x width x 3)
    img_array = np.stack(rgb_array, axis=-1)
    # get the maximum value at each x coordinate
    # print the maximum value at each x coordinate
    remove_inf = img_array.copy()
    remove_inf[remove_inf == np.inf] = 0

    furthest = np.max(remove_inf, axis=0)
    for i in range(width):
        xline = img_array[:, i, :] 
        xline[xline == np.inf] = furthest[i]
        img_array[:, i, :] = xline
    return img_array.astype(np.float32)

def log_normalize_exr_data(img_array):
    normalized_img = np.log1p(img_array)
    normalized_img = normalized_img / np.max(normalized_img)
    return normalized_img

def normalize_exr_data(img_array):
    # Normalize data to the 0-1 range for displaying
    img_min = img_array.min()
    img_max = img_array.max()
    normalized_img = (img_array - img_min) / (img_max - img_min)
    return normalized_img
def reinhard_tonemap(hdr_image):
    # Reinhard tone mapping operator
    tonemap = cv.createTonemapReinhard(gamma=2.2, intensity=1, light_adapt=0.5, color_adapt=0.5)
    ldr_image = tonemap.process(hdr_image)
    ldr_image = np.clip(ldr_image, 0, 1)
    return ldr_image



print("Reading EXR file...")
file_path = '00001Left.exr'
img_array = read_exr_to_numpy(file_path)
print("Image shape:", img_array.shape, ", Type:", img_array.dtype)
#normalized_img = log_normalize_exr_data(img_array.astype(np.float32))
normalized_img = img_array
print(normalized_img.shape)
# Display the image
#print the dtype of the normalized image
print(normalized_img.dtype)
np.save('normalized_img.npy', normalized_img[:,:,0])
plt.imshow(normalized_img, cmap='cool')
plt.axis('off')  # Hide the axes
plt.show()