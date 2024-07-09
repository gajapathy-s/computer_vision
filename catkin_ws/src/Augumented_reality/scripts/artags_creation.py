#!/usr/bin/env python

# Import the necessary libraries
import rospy
import cv2
from cv2 import aruco
import numpy as np

# Create a dictionary object from the predefined dictionary of 6x6 ArUco markers
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Loop through marker IDs 1 to 4
for i in range(1, 5):
    # Define the size of the marker image
    size = 700

    # Generate an ArUco marker image with the specified ID and size
    img = aruco.drawMarker(aruco_dict, i, size)

    # Save the generated marker image to a specified path
    cv2.imwrite('/home/user/catkin_ws/src/Augumented_reality/Tags/image_' + str(i) + ".jpg", img)

    # Display the generated marker image in a window
    cv2.imshow('artag', img)
    
    # Wait for a key press indefinitely
    cv2.waitKey(0)
    
    # Destroy the window created by imshow
    cv2.destroyAllWindows()
