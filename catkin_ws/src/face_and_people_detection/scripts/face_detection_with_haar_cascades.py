#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

class HaarCascadeDetector(object):

    def __init__(self):
        # Initialize the face cascade with the provided xml file
        self.face_cascade = cv2.CascadeClassifier('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/haar_cascades/frontalface.xml')
        
        # Read the images
        self.image_1 = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/face.jpg')
        self.image_2 = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/many.jpg')

        # Resize the first image
        self.image_1 = cv2.resize(self.image_1, (400, 700))

        # Detect faces in the images
        self.face_detect()

    def face_detect(self):
        # Convert images to grayscale
        gray_scale_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        gray_scale_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)

        # Set parameters for face detection
        scaleFactor = 1.2
        minNeighbors = 3

        # Detect faces in the first image
        faces_1 = self.face_cascade.detectMultiScale(gray_scale_1, scaleFactor, minNeighbors)

        # Detect faces in the second image
        faces_2 = self.face_cascade.detectMultiScale(gray_scale_2, scaleFactor, minNeighbors)

        # Draw rectangles around detected faces in the first image
        for (x, y, w, h) in faces_1:
            cv2.rectangle(self.image_1, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Draw rectangles around detected faces in the second image
        for (x, y, w, h) in faces_2:
            cv2.rectangle(self.image_2, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Display the images with detected faces
        cv2.imshow('Face', self.image_1)
        cv2.imshow('Faces', self.image_2)

        # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Initialize the ROS node
    rospy.init_node("haar_cascades")
    
    # Create an instance of HaarCascadeDetector
    HaarCascadeDetector()
    
    try:
        # Keep the node running
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
