#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class LoadVideo(object):

    def __init__(self):
        # Initialize the class attributes
        self.ctrl_c = False 
        self.bridge_object = CvBridge()

        # Setup the shutdown hook to properly handle shutdown signals
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        # Sets the control flag to true, ending the loop in video_detection method
        self.ctrl_c = True

    def video_detection(self):
        # Open the video file
        cap = cv2.VideoCapture("/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/chris5-2.mp4")

        # Initialize the HOG descriptor and set it as a people detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        while not self.ctrl_c:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frames for faster processing
            img_original = cv2.resize(frame, (300, 650))
            img = cv2.resize(frame, (300, 650))

            # Detect people in the frame
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

            # Display the frames
            cv2.imshow('people', img_original)
            cv2.imshow('people_original', img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('load_video_node', anonymous=True)
    load_video_object = LoadVideo()
    try:
        load_video_object.video_detection()
        rospy.spin()  # Correct function to keep the script running
    except rospy.ROSInterruptException:
        pass
    
    cv2.destroyAllWindows()
