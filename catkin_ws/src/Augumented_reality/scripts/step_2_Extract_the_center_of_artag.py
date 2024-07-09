import cv2 
import numpy as np 
from cv2 import aruco

# Function to order the coordinates of the ArUco marker corners
def order_coordinates(pts):
    # Initialize an empty array to save the ordered values
    coordinates = np.zeros((4, 2), dtype="int")

    # Calculate the sum and difference of the coordinates
    s = pts.sum(axis=1)
    coordinates[0] = pts[np.argmin(s)]  # Top-left point (smallest sum)
    coordinates[2] = pts[np.argmax(s)]  # Bottom-right point (largest sum)

    diff = np.diff(pts, axis=1)
    coordinates[1] = pts[np.argmin(diff)]  # Top-right point (smallest difference)
    coordinates[3] = pts[np.argmax(diff)]  # Bottom-left point (largest difference)

    return coordinates

# Load the image
image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/a3.jpg')
h, w = image.shape[:2]

# Resize the image to 70% of its original size
image = cv2.resize(image, (int(w*0.7), int(h*0.7)))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the ArUco dictionary and its parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Detect the corners and IDs of the ArUco markers in the image
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw the detected markers on the image
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Show the image with the detected markers
cv2.imshow('markers', frame_markers)

# Initialize an empty list to store the center coordinates of each marker
params = []

# Iterate over each detected marker
for i in range(len(ids)):
    # Get the corners of each marker
    c = corners[i][0]

    # Calculate the center of the marker
    center_x = int(c[:, 0].mean())
    center_y = int(c[:, 1].mean())

    # Draw a circle at the center of each marker
    cv2.circle(image, (center_x, center_y), 3, (255, 255, 0), -1)
    
    # Save the center coordinates
    params.append((center_x, center_y))

# Convert the coordinates list to a NumPy array
params = np.array(params)

# Draw a polygon connecting the centers of the markers
cv2.drawContours(image, [params], -1, (255, 0, 150), -1)

# Show the final image with the polygon
cv2.imshow('no_conversion', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
