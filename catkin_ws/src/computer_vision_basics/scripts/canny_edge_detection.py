#!/usr/bin/env python

import cv2
import rospy
import numpy as np 

# Load and resize the image
img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img.png')
img = cv2.resize(img, (450, 350))

# Parameters for Canny edge detection
minV = 30  # Minimum intensity gradient
maxV = 100  # Maximum intensity gradient

# Apply Canny edge detection
edges = cv2.Canny(img, minV, maxV)

# Display the original and edges images
cv2.imshow('Original Image', img)
cv2.imshow('Edges Detected', edges)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
