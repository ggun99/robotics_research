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
import os

class ImageListener(Node):
    def __init__(self, depth_image_topic, depth_info_topic):
        super().__init__('listener')
        self.subscription = self.create_subscription(msg_Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription # prevent unused variable warning
        self.getballposition
        self.bridge = CvBridge()
        self.sub = self.create_subscription(msg_Image, depth_image_topic, self.imageDepthCallback, 10)
        self.sub_info = self.create_subscription(CameraInfo, depth_info_topic, self.imageDepthInfoCallback, 1)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = self.create_subscription(msg_Image, confidence_topic, self.confidenceCallback, 1)
        self.intrinsics = None
        self.pix = [0,0]
        self.pix_grade = None
        self.center = [0,0]
        self.publisher_ = self.create_publisher(PoseArray, '/pose', 10)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.o = 0
        self.r = 0

    def getballposition(self, filename):
        img = filename
        
        # define kernel size
        kernel = np.ones((7, 7), np.uint8)

        # convert to hsv colorspace
        hsv2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # lower bound and upper bound for Yellow color
        lower_bound = np.array([20, 80, 80])
        upper_bound = np.array([30, 255, 255])
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
                               minRadius=8, maxRadius=70)
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles2=sorted(circles[0],key=lambda x:x[2],reverse=True)
            i = circles2[0]
            self.center = (i[0], i[1])
            self.r = (i[2])
            if (i[0] >= gray.shape[1]) or (i[1] >= gray.shape[0]) :
                
                return mask
            
        return mask, self.r
    
    def append_matrix_to_file(self, matrix, filename):
    # 행렬을 문자열로 변환
        matrix_str = "[" + "\n".join([",".join(map(str, row)) for row in matrix]) + "],"

        # 파일에 행렬 추가
        with open(filename, "a") as file:
            file.write(matrix_str + "\n")
    
    def imageDepthCallback(self, msg):
        if self.center is not [0,0] or not None:
            
            pixel_x = self.center[0]  # X coordinate of the pixel
            pixel_y = self.center[1]  # Y coordinate of the pixel
            image_index = (pixel_y * msg.width) + pixel_x
            depth = float(msg.data[image_index * 2] + (msg.data[image_index*2 + 1] << 8)) / 1000
            if depth > 3000:
                return pixel_x,pixel_y
            pix = (pixel_x, pixel_y)
            self.pix = pix
            result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
            if result[2]+result[0]+result[1] is 0:
                return
           
            print(f"[{result[2]},{-result[0]},{-result[1]}]")
            
            
            incoming_vector = [result[2],-result[0],-result[1]]
            self.append_matrix_to_file([incoming_vector], "matrix_data.txt")

            poseArrayMsg = PoseArray()
            poseArrayMsg.header = Header()
            poseArrayMsg.header.stamp = self.get_clock().now().to_msg()
            poseArrayMsg.header.frame_id = 'camera_link' 
            
            x = result[2]*1000
            y = -result[0]*1000
            z = -result[1]*1000
            
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            poseArrayMsg.poses.append(pose)
            self.publisher_.publish(poseArrayMsg)      
        else:
            return
        


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

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError as e:
            print(e)
            return


def main(args=None):
    
    rclpy.init(args=args)
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'
    listener = ImageListener(depth_image_topic, depth_info_topic)
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()    
    
if __name__ == '__main__':
    main()