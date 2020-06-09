# import required libraries
import numpy as np
import cv2
import imutils

WIDTH = 600
HEIGHT = 400

# parameter for image to scan/process
args_image = "C:/Users/PC/Desktop/key.jpg"
# read the image
image = cv2.imread(args_image)
image = cv2.resize(image,(WIDTH,HEIGHT))

#make new image
roImage = np.zeros((WIDTH,HEIGHT,3), np.uint8)

matrix = cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), 90, 1)

roImage = cv2.warpAffine(image, matrix, (WIDTH,HEIGHT))

cv2.imshow("t",image)

