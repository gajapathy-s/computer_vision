import cv2
import numpy as np

# Read the image in grayscale and resize
img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/world.png', 0)
img = cv2.resize(img, (450, 450))

# Define a kernel for erosion
kernel_a = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel_a, iterations=1)

# Define a kernel for dilation
kernel_b = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(img, kernel_b, iterations=1)

# Define a kernel for opening
kernel_c = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_c)

# Define a kernel for closing
kernel_d = np.ones((7, 7), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_d)

# Display the original image and processed images
cv2.imshow('Original', img)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
