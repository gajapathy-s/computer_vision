#!/usr/bin/env python

import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_4/Course_images/test_e.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Initialize the FAST detector
fast = cv2.FastFeatureDetector_create()

# Initialize the BRIEF descriptor extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detect keypoints using FAST
keypoints = fast.detect(gray, None)

# Compute BRIEF descriptors for the detected keypoints
brief_keypoints, descriptor = brief.compute(gray, keypoints)

# Make copies of the original image to draw keypoints on
brief_image = np.copy(image)
non_brief_image = np.copy(image)

# Draw keypoints on top of the input image
cv2.drawKeypoints(image, brief_keypoints, brief_image, color=(0,250,250))
cv2.drawKeypoints(image, keypoints, non_brief_image, color=(0,35,250))

# Display the images with keypoints
cv2.imshow('Fast corner detection', non_brief_image)
cv2.imshow('BRIEF descriptors', brief_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
