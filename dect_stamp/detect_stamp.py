import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pyrealsense2 as rs2
import tf2_ros
from time import time 
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(msg_Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription # prevent unused variable warning
        self.getballposition
        self.bridge = CvBridge()
        self.intrinsics = None
        self.pix = [0,0]
        self.pix_grade = None
        self.center = [0,0]
        self.publisher_ = self.create_publisher(PoseArray, '/pose', 10)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

    def getballposition(self, filename):
        img = filename
        
        # define kernel size
        kernel = np.ones((7, 7), np.uint8)

        # convert to hsv colorspace
        hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # lower bound and upper bound for Yellow color
        lower_bound = np.array([35, 140, 60])
        upper_bound = np.array([255, 255, 180])
        # find the colors within the boundaries
        mask = cv2.inRange(hsv2, lower_bound, upper_bound)
        # define kernel size
        kernel = np.ones((7, 7), np.uint8)
        # Remove unnecessary noise from mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        gray = cv2.medianBlur(mask, 9)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 3,
                               param1=100, param2=10,
                               minRadius=4, maxRadius=20)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles2=sorted(circles[0],key=lambda x:x[2],reverse=True)
            i = circles2[0]
            if (i[0] >= gray.shape[1]) or (i[1] >= gray.shape[0]) :
                return mask
            self.center = (i[0], i[1])
            r = (i[2])
            cv2.circle(hsv1, self.center, 1, (255,0,0), 3) #if you want to see the circle.
            cv2.imshow(hsv1,'detect')
        return mask
    
    def listener_callback(self, data):
        """
        Callback function.
        """
        current_frame = self.bridge.imgmsg_to_cv2(data)
        
        if self.center is not None or not [0,0]:
            self.getballposition(current_frame)
        else:
            return
            
        cv2.waitKey(1)

    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return


def main(args=None):
    
    rclpy.init(args=args)
    
    listener = ImageListener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()    
    
if __name__ == '__main__':
    main()