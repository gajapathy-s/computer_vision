import cv2
import numpy as np
from cv2 import aruco

# Function to order coordinates based on the given method
def order_coordinates(pts, var):
    coordinates = np.zeros((4,2), dtype="int")
    if var:
        # Sort model 1: Order points by summing and diffing coordinates
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]  # Top-left
        coordinates[3] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]  # Top-right
        coordinates[2] = pts[np.argmax(diff)]  # Bottom-left
    else:
        # Sort model 2: Different ordering approach
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]  # Top-left
        coordinates[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]  # Top-right
        coordinates[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return coordinates

# Load the main image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a1.jpg')
h, w = image.shape[:2]

# Resize the image for processing
image = cv2.resize(image, (int(w*0.7), int(h*0.7)))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the ArUco dictionary and parameters for detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Detect ArUco markers in the grayscale image
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Initialize an empty list to store the coordinates of the marker centers
params = []

# Iterate over each detected marker
for i in range(len(ids)):
    # Get the corner points of the current marker
    c = corners[i][0]

    # Draw a circle at the center of the marker
    cv2.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
    
    # Save the center coordinates of the marker
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

# Convert the list of coordinates to a numpy array
params = np.array(params)

# Ensure there are at least 4 markers detected for homography calculation
if len(params) >= 4:
    # Order the coordinates using the first sorting method
    params = order_coordinates(params, False)
    
    # Order the coordinates using the second sorting method
    params_2 = order_coordinates(params, True)

# Load the image to be overlapped on the main image
paint = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/earth.jpg')
height, width = paint.shape[:2]

# Define the corner coordinates of the new image
coordinates = np.array([[0, 0], [width, 0], [0, height], [width, height]])

# Find the homography matrix to warp the new image to the marker positions
hom, status = cv2.findHomography(coordinates, params_2)
  
# Warp the new image using the homography matrix
warped_image = cv2.warpPerspective(paint, hom, (int(w*0.7), int(h*0.7)))

# Create a black mask with the same size as the main image
mask = np.zeros([int(h*0.7), int(w*0.7), 3], dtype=np.uint8)

# Fill the mask with white where the markers are located
cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)
cv2.imshow('black mask', mask)

# Subtract the mask from the main image to create a darkened area
substraction = cv2.subtract(image, mask)
cv2.imshow('substraction', substraction)

cv2.waitKey(0)
cv2.destroyAllWindows()
