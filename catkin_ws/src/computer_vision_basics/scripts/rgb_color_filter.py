#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ColorFilter(object):
    def __init__(self):
        # Initialize the ROS subscriber to the topic "/camera/rgb/image_raw"
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        # Initialize the CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()
        # Initialize the image variable
        self.img = None

    def camera_callback(self, data):
        # Callback function to handle image data received from the camera topic
        try:
            # Convert the ROS Image message to an OpenCV image
            self.img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Load and save a test image (this can be used for debugging purposes)
        test_img_path = '/home/user/computer_vision/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/Filtering.png'
        img = cv2.imread(test_img_path)

        # Check if the image is loaded properly
        if img is None:
            print(f"Failed to load image from {test_img_path}")
            return

        # Resize the image to 300x300 pixels
        resized_img = cv2.resize(img, (300, 300))

        # Convert the resized image to HSV color space
        hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

        # Define the color ranges for green, red, and blue
        min_green = np.array([50, 220, 220])
        max_green = np.array([60, 255, 255])

        min_red = np.array([170, 220, 220])
        max_red = np.array([180, 255, 255])

        min_blue = np.array([110, 220, 220])
        max_blue = np.array([120, 255, 255])

        # Create masks for green, red, and blue colors
        mask_g = cv2.inRange(hsv, min_green, max_green)
        mask_r = cv2.inRange(hsv, min_red, max_red)
        mask_b = cv2.inRange(hsv, min_blue, max_blue)

        # Apply the masks to get the color filtered images
        res_g = cv2.bitwise_and(resized_img, resized_img, mask=mask_g)
        res_r = cv2.bitwise_and(resized_img, resized_img, mask=mask_r)
        res_b = cv2.bitwise_and(resized_img, resized_img, mask=mask_b)

        # Display the original and filtered images
        cv2.imshow('Original', resized_img)
        cv2.imshow('Green', res_g)
        cv2.imshow('Red', res_r)
        cv2.imshow('Blue', res_b)
        cv2.waitKey(1)

def main():
    # Initialize the ROS node
    rospy.init_node("color_filtering", anonymous=True)
    # Create an instance of the ColorFilter class
    ColorFilter()

    try:
        # Keep the program alive until interrupted
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
