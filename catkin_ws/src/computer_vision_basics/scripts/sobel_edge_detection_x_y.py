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

        # Load and process the image
        image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img_b.jpg')
        img = cv2.resize(image, (450, 350))

        # Convert the image to HSV and grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gryscl = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gry = cv2.resize(gryscl, (450, 350))

        # Define color ranges for masking
        min_green = np.array([40, 50, 50])
        max_green = np.array([60, 255, 255])
        min_red = np.array([0, 45, 142])
        max_red = np.array([10, 255, 255])
        min_blue = np.array([100, 50, 50])
        max_blue = np.array([120, 255, 255])

        # Create masks for green, blue, and red colors
        mask_g = cv2.inRange(hsv, min_green, max_green)
        mask_b = cv2.inRange(hsv, min_blue, max_blue)
        mask_r = cv2.inRange(hsv, min_red, max_red)

        # Apply the masks to the image
        res_g = cv2.bitwise_and(img, img, mask=mask_g)
        res_b = cv2.bitwise_and(img, img, mask=mask_b)
        res_r = cv2.bitwise_and(img, img, mask=mask_r)

        # Apply Sobel filters to detect edges
        sobel_x = cv2.Sobel(img_gry, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gry, cv2.CV_64F, 0, 1, ksize=3)

        # Display the original and processed images
        cv2.imshow('Original', img)
        cv2.imshow('Sobel X', sobel_x)
        cv2.imshow('Sobel Y', sobel_y)
        cv2.imshow('Green', res_g)
        cv2.imshow('Red', res_r)
        cv2.imshow('Blue', res_b)

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
