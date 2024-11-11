import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

numDisparities = 5
baseLine = 270
width = 512
fov = 47
focalLength = width / (2 * np.tan(np.radians(fov / 2)))

imgL = cv.imread('rgb/00001Left.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('rgb/00001Right.png', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities=numDisparities*16, blockSize=15)
stereo.setTextureThreshold(20)
disparity = stereo.compute(imgL,imgR)

fig, ax = plt.subplots(2,2)

ax[0,0].imshow(imgL,'gray')
ax[0,1].imshow(imgR,'gray')

ax[1,0].imshow(disparity,'cool')

depth = (focalLength * baseLine) / (disparity + 0.0001)
ax[1,1].imshow(depth,'cool')




plt.show()