#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from skimage import exposure 
from skimage import feature

class LoadPeople(object):

    def __init__(self):
        # Initialize the subscriber to the camera topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        # Initialize the CvBridge object to convert ROS images to OpenCV format
        self.bridge_object = CvBridge()

    def camera_callback(self, data):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            # Print any conversion errors
            print(e)
        
        # Load a static image from disk for HOG feature extraction
        img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/test_e.jpg')

        # Resize the image for faster processing
        imX = 720
        imY = 1080
        img = cv2.resize(img, (imX, imY))
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute HOG features and the HOG image for visualization
        (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True)

        # Rescale the intensity of the HOG image to 8-bit (0 to 255) range
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        # Display the HOG features image
        cv2.imshow('features', hogImage)

        # Wait for a short period to allow the image to be displayed
        cv2.waitKey(1)

def main():
    # Initialize the ROS node
    rospy.init_node('load_people_node', anonymous=True)
    # Create an instance of the LoadPeople class
    load_people_object = LoadPeople()
    try:
        # Keep the node running
        rospy.spin()
    except KeyboardInterrupt:
        # Handle keyboard interrupt to shutdown cleanly
        print("Shutting down")
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
