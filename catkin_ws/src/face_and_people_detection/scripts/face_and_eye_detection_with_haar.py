#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class LoadFace(object):
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()

        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier('/home/user/catkin_ws/src/unit3_exercises/haar_cascades/frontalface.xml')
        self.eyes_cascade = cv2.CascadeClassifier('/home/user/catkin_ws/src/unit3_exercises/haar_cascades/eye.xml')

        # Check if the cascades are loaded correctly
        if self.face_cascade.empty():
            rospy.logerr("Failed to load face cascade classifier.")
        if self.eyes_cascade.empty():
            rospy.logerr("Failed to load eyes cascade classifier.")

    def camera_callback(self, data):
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        ScaleFactor = 1.2
        minNeighbors = 3
        faces = self.face_cascade.detectMultiScale(gray, ScaleFactor, minNeighbors)

        # Draw rectangles around faces and eyes
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = cv_image[y:y + h, x:x + w]

            eyes = self.eyes_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Display the image with detected faces and eyes
        cv2.imshow('Face Detection', cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('load_face_node', anonymous=True)
    load_face_object = LoadFace()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
