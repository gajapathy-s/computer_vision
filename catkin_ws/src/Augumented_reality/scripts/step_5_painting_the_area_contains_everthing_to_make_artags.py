import os
import cv2 
import numpy as np 
from cv2 import aruco

# Print the location of the cv2 module
print(cv2.__file__)

def order_coordinates(pts, var):
    # Initialize an empty array to hold the coordinates of the four points
    coordinates = np.zeros((4,2),dtype="int")

    if(var):
        # Parameters sort model 1 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
        coordinates[3] = pts[np.argmax(s)]  # Bottom-right point has the largest sum

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]  # Top-right point has the smallest difference
        coordinates[2] = pts[np.argmax(diff)]  # Bottom-left point has the largest difference
    
    else:
        # Parameters sort model 2 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
        coordinates[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]  # Top-right point has the smallest difference
        coordinates[3] = pts[np.argmax(diff)]  # Bottom-left point has the largest difference
    
    return coordinates

# Load the main image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a1.jpg')
h, w = image.shape[:2]

# Resize the image
image = cv2.resize(image, (int(w*0.7), int(h*0.7)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the aruco dictionary and its parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Detect the corners and ids in the images
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Initialize an empty list for the coordinates
params = []

# Iterate through each detected marker
for i in range(len(ids)):
    # Catch the corners of each tag
    c = corners[i][0]

    # Draw a circle in the center of each detection
    cv2.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
    
    # Save the coordinates of the center of each tag
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

# Transform the coordinates list to an array
params = np.array(params)
if(len(params) >= 4):
    # Sort model 1 
    params = order_coordinates(params, False)
    
    # Sort Model 2
    params_2 = order_coordinates(params, True)

# Read the image we want to overlap
paint = cv2.imread('/home/user/catkin_ws/src/Augumented_reality/scripts/me.jpeg')
height, width = paint.shape[:2]

# Extract the coordinates of this new image which are basically the full-sized image
coordinates = np.array([[0,0], [width,0], [0,height], [width,height]])

# Find a perspective transformation between the planes
hom, status = cv2.findHomography(coordinates, params_2)
  
# Save the warped image in a dark space with the same size as the main image
warped_image = cv2.warpPerspective(paint, hom, (int(w*0.7), int(h*0.7)))

# Create a black mask to do the image operations
mask = np.zeros([int(h*0.7), int(w*0.7), 3], dtype=np.uint8)

# Replace the area described by the ar tags with white on the black mask
cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)
cv2.imshow('black mask', mask)

# Subtract the mask from the main image
substraction = cv2.subtract(image, mask)
cv2.imshow('substraction', substraction)

# Add the warped image to the subtracted image
addition = cv2.add(warped_image, substraction)
cv2.imshow('detection', addition)

cv2.waitKey(0)
cv2.destroyAllWindows()
