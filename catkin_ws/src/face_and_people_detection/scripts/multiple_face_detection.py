#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MultiFaceDetector(object):
    def __init__(self):
        # Initialize the image using OpenCV from a specific path
        self.image = cv2.imread("/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_3/Course_images/many.jpg")
        
        # Initialize the face cascade classifier using a Haar cascade file
        self.face_cascade = cv2.CascadeClassifier("/home/user/catkin_ws/src/face_and_people_detection/haar_cascades/frontalface.xml")

    def detect_and_publish(self):
        # Loop until ROS node is shutdown
        while not rospy.is_shutdown():
            # Resize the original image for display purposes
            img_original = cv2.resize(self.image, (700, 600))
            
            # Create a copy of the resized image for processing
            img = cv2.resize(img_original, (700, 600))

            # Convert the image to grayscale for face detection
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Parameters for face detection
            ScaleFactor = 1.2  # Scale factor for image pyramid
            minNeighbors = 3    # Minimum number of neighbors required for a detected face
            
            # Detect faces in the grayscale image
            faces = self.face_cascade.detectMultiScale(gray_scale, ScaleFactor, minNeighbors)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                # Optional: Extract and process the region of interest (ROI)
                roi = img[y:y+h, x:x+w]

            # Display the original image with rectangles around faces
            cv2.imshow('Faces_original', img_original)
            
            # Display the processed image with rectangles around faces
            cv2.imshow('Faces', img)
            
            # Wait for a key press and check if the user has pressed any key
            cv2.waitKey(1)

def main():
    # Initialize the ROS node
    rospy.init_node('multi_face_detect')
    
    # Create an instance of the MultiFaceDetector class
    mfd = MultiFaceDetector()
    
    # Start detecting faces
    mfd.detect_and_publish()
    
    # Keep the node running until shutdown
    rospy.spin()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
