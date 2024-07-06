#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LoadImg(object):
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

  
       # cv2.imshow('image', self.img)
    
        

        # Load and save a test image (this can be used for debugging purposes)
        img = cv2.imread('/home/user/computer_vision/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_image_1.jpg')
        cv2.imshow('image', img)
        cv2.imwrite('drone_image.jpg', self.img)
        cv2.waitKey(0)
      
def main():
    # Initialize the ROS node
    rospy.init_node("img_load", anonymous=True)
    # Create an instance of the LoadImg class
    load_img_obj = LoadImg()

    try:
        # Keep the program alive until interrupted
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
