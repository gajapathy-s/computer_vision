import cv2 
import numpy as np 
from cv2 import aruco

# Function to order the coordinates of the detected marker corners
def order_coordinates(pts):
    # Initialize an empty array to save the coordinates
    coordinates = np.zeros((4, 2), dtype="int")

    # Calculate the sum of the x and y coordinates
    s = pts.sum(axis=1)
    # The top-left point will have the smallest sum
    coordinates[0] = pts[np.argmin(s)]
    # The bottom-right point will have the largest sum
    coordinates[2] = pts[np.argmax(s)]

    # Calculate the difference between the x and y coordinates
    diff = np.diff(pts, axis=1)
    # The top-right point will have the smallest difference
    coordinates[1] = pts[np.argmin(diff)]
    # The bottom-left point will have the largest difference
    coordinates[3] = pts[np.argmax(diff)]

    return coordinates

# Read the image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
# Get the dimensions of the image
h, w = image.shape[:2]

# Resize the image to 70% of its original size
image = cv2.resize(image, (int(w * 0.7), int(h * 0.7)))
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the aruco dictionary and its parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Detect the corners and ids of the markers in the image
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw the detected markers on the image
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Display the image with the detected markers
cv2.imshow('markers', frame_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()
