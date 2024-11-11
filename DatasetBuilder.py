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

def read_imgs_to_numpy(iteration):
    imgL = cv.imread('rgb/' + str(iteration).zfill(5) + 'Left.png', cv.IMREAD_GRAYSCALE)
    imgR = cv.imread('rgb/' + str(iteration).zfill(5) + 'Right.png', cv.IMREAD_GRAYSCALE)
    return imgL, imgR

def compute_disparity(imgL, imgR, numDisparities=5):
    stereo = cv.StereoBM_create(numDisparities=numDisparities*16, blockSize=15)
    stereo.setTextureThreshold(20)
    disparity = stereo.compute(imgL,imgR)
    return disparity.astype(np.float32)


def getSample(iteration):
    imgL, imgR = read_imgs_to_numpy(iteration)
    pixelMatched = compute_disparity(imgL, imgR)
    return imgL, imgR, pixelMatched

def testSample():
    imgL, imgR, pixelMatched, depth =  getSample(1)

    # print the shape of the images
    print("imgL shape: ", imgL.shape)
    print("imgR shape: ", imgR.shape)
    print("pixelMatched shape: ", pixelMatched.shape)
    print("depth shape: ", depth.shape)

    # Display the images
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(imgL,'gray')
    ax[0,1].imshow(imgR,'gray')
    ax[1,0].imshow(pixelMatched,'cool')
    ax[1,1].imshow(depth,'cool')
    plt.show()

#testSample()

noSamples = 100

leftImages = np.empty((noSamples, 512, 512), dtype=np.float16)
rightImages = np.empty((noSamples, 512, 512), dtype=np.float16)
matchesDisp = np.empty((noSamples, 512, 512), dtype=np.float16)

for i in range(0, noSamples):
    imgL, imgR, pixelMatched=  getSample(i+1)
    #print("imgL shape: ", imgL.shape, " dtype: ", imgL.dtype)
    #print("imgR shape: ", imgR.shape, " dtype: ", imgR.dtype)
    #print("pixelMatched shape: ", pixelMatched.shape, " dtype: ", pixelMatched.dtype)
    leftImages[i] = imgL.astype(np.float16)
    rightImages[i] = imgR.astype(np.float16)
    matchesDisp[i] = pixelMatched.astype(np.float16)

print("leftImages shape: ", leftImages.shape)
print("rightImages shape: ", rightImages.shape)
print("matchesDisp shape: ", matchesDisp.shape)

x_train = np.stack((leftImages, rightImages, matchesDisp), axis=-1)

print("x_train shape: ", x_train.shape)
# save to numpy_data
np.save('numpy_data/x_train.npy', x_train)







    