#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

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

        # Initialize the HOG descriptor
        hog = cv2.HOGDescriptor()

        # Set the HOG descriptor as a people detector
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Load a static image from disk for HOG-based people detection
        img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/test_e.jpg')

        # Resize the image for faster processing
        imX = 720
        imY = 1080
        img = cv2.resize(img, (imX, imY))

        # Perform people detection with a defined window stride of 8x8 pixels
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        # Draw detection boxes and centers
        for (xA, yA, xB, yB) in boxes:
            # Calculate the center of the detection box
            xC = int(xA + (xB - xA) / 2)
            yC = int(yA + (yB - yA) / 2)

            # Draw a circle at the center of the detection box
            cv2.circle(img, (xC, yC), 1, (0, 255, 255), -1)

            # Draw the detection box
            cv2.rectangle(img, (xA, yA), (xB, yB), (255, 255, 0), 2)

        # Display the processed image with detection boxes
        cv2.imshow('frame_2', img)

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
