#!/usr/bin/env python

import cv2
import numpy as np

# Load two images from specified paths
image_1_path = '/home/user/catkin_ws/src/feature_matching/scripts/h.jpg'
image_2_path = '/home/user/catkin_ws/src/feature_matching/scripts/t.jpg'

image_1 = cv2.imread(image_1_path, 1)
image_2 = cv2.imread(image_2_path, 1)

# Check if images are loaded correctly
if image_1 is None:
    print(f"Error: Could not load image from {image_1_path}")
    exit(1)

if image_2 is None:
    print(f"Error: Could not load image from {image_2_path}")
    exit(1)

# Convert the images to grayscale
gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

# Initialize the ORB (Oriented FAST and Rotated BRIEF) feature detector with a maximum of 1000 features
orb = cv2.ORB_create(nfeatures=1000)

# Make copies of the original images to display the keypoints found by ORB
preview_1 = np.copy(image_1)
preview_2 = np.copy(image_2)

# Create another copy of image_1 to display keypoints as dots only
dots = np.copy(image_1)

# Extract the keypoints and descriptors from both grayscale images using ORB
train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

# Draw the keypoints on the first image
cv2.drawKeypoints(image_1, train_keypoints, preview_1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image_1, train_keypoints, dots, flags=2)

#############################################
################## MATCHER ##################
#############################################

# Initialize the BruteForce Matcher with Hamming distance as measurement and enable cross-checking
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the feature points from both images
matches = bf.match(train_descriptor, test_descriptor)

# Sort the matches based on their distances. The matches with shorter distances are better.
matches = sorted(matches, key=lambda x: x.distance)

# Select the top 100 matches
good_matches = matches[:100]

# Extract the coordinates of the matched keypoints
train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the homography matrix using RANSAC (Random Sample Consensus)
M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC, 5.0)

# Get the width and height of the first grayscale image
h, w = gray_1.shape[:2]

# Create a floating-point matrix of the image cornersA
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# Transform the corners using the homography matrix
dst = cv2.perspectiveTransform(pts, M)

# Draw the matching lines between the two images
dots = cv2.drawMatches(dots, train_keypoints, image_2, test_keypoints, good_matches, None, flags=2)

# Draw the perspective-transformed bounding box on the second image
result = cv2.polylines(image_2, [np.int32(dst)], True, (50, 0, 255), 3, cv2.LINE_AA)

# Display the images with keypoints, matches, and the bounding box

desired_width, desired_height = 800, 600  # Adjust these values as needed
image_1 = cv2.resize(preview_1, (desired_width, desired_height))
image_2 = cv2.resize(result, (desired_width, desired_height))
image_3 = cv2.resize( dots, (desired_width, desired_height))


cv2.imshow('Points',image_1 )
cv2.imshow('Matches', image_3)
cv2.imshow('Detection',image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
