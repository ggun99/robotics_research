import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as msg_Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pyrealsense2 as rs2
import tf2_ros
from time import time, sleep
from rainbow import cobot
cobot.ToCB('192.168.0.201')
e = None
while e is None:
    if cobot.GetCurrentCobotStatus() is cobot.COBOT_STATUS.IDLE:
        e = True
        break
    else:
        e = None
        continue
cobot.SetBaseSpeed(1.0)
# Initialize (activate) the cobot.
cobot.CobotInit()
cobot.SetProgramMode(cobot.PG_MODE.REAL)
sleep(2)
# from std_msgs.msg import Header
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(msg_Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription
        self.find_closest_circle
        self.detect_circles_and_stamp
        self.bridge = CvBridge()
        self.intrinsics = None
        self.pix = [0,0]
        self.pix_grade = None
        self.center = [0,0]
        self.closest_circle = None
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
    
    def find_closest_circle(self, image, circles, target_hsv):
        # closest_circle = None
        min_color_difference = float('inf')
        for (cx, cy, radius) in circles:
            # 동전 영역 부분 크롭 영상 만들기
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(image.shape[1], cx + radius)
            y2 = min(image.shape[0], cy + radius)

            crop = image[y1:y2, x1:x2]  # 크롭 영상 생성

            avg_color = np.mean(crop)

            # 색상 차이 계산
            color_difference = np.abs(avg_color - target_hsv)

            # 색상 차이가 현재까지의 최소 차이보다 작으면 업데이트
            if color_difference < min_color_difference:
                min_color_difference = color_difference
                self.closest_circle = (cx, cy, radius)

        return self.closest_circle

    def detect_circles_and_stamp(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

        # (노이즈를 줄이기 위해) 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 허프 변환(원 검출)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=50, param2=30, minRadius=20, maxRadius=80)

        dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv1 = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        # 원의 중점에 점, Stamped, Empty 확인
        if circles is not None:
            number_of_circles = circles.shape[1]
            print(f"Detected circles: {number_of_circles}")
            circles = np.round(circles[0, :]).astype("int")
            self.closest_circle = self.find_closest_circle(hsv1, circles, 80.0)
            if self.closest_circle is not None:
                    
                won = "Empty"
                won = won + " (" + str(self.closest_circle[0]) + ", " + str(self.closest_circle[1]) + ") "
                # 중심에 원을 그리기
                cv2.circle(dst, (self.closest_circle[0], self.closest_circle[1]), 1, (0, 255, 0), -1)

                # 원 경계를 그리기
                cv2.circle(dst, (self.closest_circle[0], self.closest_circle[1]), self.closest_circle[2], (0, 255, 0), 2)
                # 결과를 화면에 출력
                cv2.putText(dst, won, (self.closest_circle[0] - 10, self.closest_circle[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # 결과
                # cv2.imshow("hsv", hsv1)
                cv2.imshow('Result Image', dst)

            return self.closest_circle
        else:
            print("No circles detected.")
        return dst
    
    def move_RB730(self, pose):
        mat_e2c = np.array([[0.0,  0.0, 1.0,    0.0],
                            [0.0, -1.0, 0.0, -0.027],
                            [1.0,  0.0, 0.0,  0.062],
                            [0.0,  0.0, 0.0,    1.0]])
        
        current_tcp = cobot.GetCurrentTCP()
        R = np.array([[1.0,0.0,0.0],[0.0,np.cos(current_tcp[3]),-np.sin(current_tcp[3])],[0.0,np.sin(current_tcp[3]),np.cos(current_tcp[3])]])*np.array([[np.cos(current_tcp[4]),0.0,np.sin(current_tcp[4])],[0.0,1.0,0.0],[-np.sin(current_tcp[4]),0.0,np.cos(current_tcp[4])]])*np.array([[np.cos(current_tcp[5]),-np.sin(current_tcp[5]),0.0],[np.sin(current_tcp[5]),np.cos(current_tcp[5]),0.0],[0.0,0.0,1.0]])
        mat_b2e = np.array([[R[0,0],R[0,1],R[0,2],current_tcp[0]],[R[1,0],R[1,1],R[1,2],current_tcp[1]],[R[2,0],R[2,1],R[2,2],current_tcp[2]],[0.0,0.0,0.0,1.0]])
        mat_c2s = np.array([self.closest_circle[0],self.closest_circle[1],0.2,1.0])
        mat_c2s_T = np.transpose(mat_c2s)
        mat_b2s = mat_b2e*mat_e2c*mat_c2s_T
        cobot.MoveL(mat_b2s[0,0],mat_b2s[1,0],mat_b2s[2,0],0.0,0.0,0.0,-1.0,-1.0)

    def listener_callback(self, data):
        """
        Callback function.
        """
        current_frame = self.bridge.imgmsg_to_cv2(data)
        
        if self.center is not None or not [0,0]:
            self.detect_circles_and_stamp(current_frame)
        
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