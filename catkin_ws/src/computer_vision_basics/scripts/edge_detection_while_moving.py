#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Sobel:
    def __init__(self):
        # Initialize the ROS node and create a subscriber to the camera's image topic
        rospy.init_node('sobel_node', anonymous=True)
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge = CvBridge()

    def camera_callback(self, data):
        try:
            # Convert the ROS image message to an OpenCV image
            drone_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image for display (optional)
        img_gray_resized = cv2.resize(img_gray, (450, 350))

        # Apply Sobel filters to detect edges in x and y directions
        sobel_x = cv2.Sobel(img_gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray_resized, cv2.CV_64F, 0, 1, ksize=3)

        # Display the original grayscale image and processed images
        cv2.imshow('Original Grayscale', img_gray_resized)
        cv2.imshow('Sobel X', sobel_x)
        cv2.imshow('Sobel Y', sobel_y)

        # Wait for a key press (necessary for OpenCV window to update)
        cv2.waitKey(1)

def main():
    sobel = Sobel()  # Create an instance of the Sobel class
    try:
        rospy.spin()  # Keep the node running
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    main()
