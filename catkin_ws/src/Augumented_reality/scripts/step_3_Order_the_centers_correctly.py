import cv2 
import numpy as np 
from cv2 import aruco

def order_coordinates(pts):
    # Initialize an empty array to save the next values
    coordinates = np.zeros((4, 2), dtype="int")

    # Sum the points along axis 1 and find the min and max sums
    s = pts.sum(axis=1)
    coordinates[0] = pts[np.argmin(s)]  # Top-left point
    coordinates[2] = pts[np.argmax(s)]  # Bottom-right point

    # Calculate the difference and find the min and max differences
    diff = np.diff(pts, axis=1)
    coordinates[1] = pts[np.argmin(diff)]  # Top-right point
    coordinates[3] = pts[np.argmax(diff)]  # Bottom-left point

    return coordinates

# Read the input image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
h, w = image.shape[:2]

# Resize the image
image = cv2.resize(image, (int(w * 0.7), int(h * 0.7)))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the ArUco dictionary and its parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Detect the corners and IDs in the image
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw the detected markers on the image
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Show the image with detected markers
cv2.imshow('markers', frame_markers)

# Initialize an empty list for the coordinates
params = []

# Loop over each detected marker
for i in range(len(ids)):
    # Get the corners of each marker
    c = corners[i][0]

    # Draw a circle at the center of each marker
    cv2.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
    
    # Save the center coordinates of each marker
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

# Convert the coordinates list to an array
params = np.array(params)

# Draw a polygon connecting the centers of the markers
cv2.drawContours(image, [params], -1, (255, 0, 150), -1)

# If there are at least 4 markers detected
if(len(params) >= 4):
    # Sort the coordinates
    params = order_coordinates(params)

    # Draw the polygon with the sorted coordinates
    cv2.drawContours(image, [params], -1, (255, 0, 150), -1)

# Show the final image with detection
cv2.imshow('detection', image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
