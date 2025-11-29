import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from multicam_aruco import ARUCOBoardPose
        
class SteroArucoROS(Node):

    def __init__(self):
        super().__init__('stereo_aruco_ros')
        self.bridge = CvBridge()
        self.aruco_board_detector = ARUCOBoardPose()

        self.camera_left_sub = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.image_left_callback,
            10)
        self.camera_right_sub = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.image_right_callback,
            10)
        self.camera_info_left = self.create_subscription(
            CameraInfo,
            '/camera1/camera1/color/camera_info',
            self.camera_info_callback_left,
            1)
        self.camera_info_right = self.create_subscription(
            CameraInfo,
            '/camera2/camera2/color/camera_info',
            self.camera_info_callback_right,
            1)
        
        self.timer_ = self.create_timer(0.1, self.loop)
        
        # containers for camera info and images
        self.camera_info_left_param = None
        self.camera_info_right_param = None
        self.camera_image_left = None
        self.camera_image_right = None

        self.get_logger().info("Stereo ArUco ROS Node Initialized")

    def camera_info_callback_left(self, msg:CameraInfo):
        if self.camera_info_left_param is None:
            self.camera_info_left_param = msg
            
    def camera_info_callback_right(self, msg:CameraInfo):
        if self.camera_info_right_param is None:
            self.camera_info_right_param = msg
    
    def image_left_callback(self, msg:Image):
        self.camera_image_left = msg
        # self.latest_img1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
 
    def image_right_callback(self, msg:Image):
        self.camera_image_right = msg

    def loop(self):
        if self.camera_image_left is None or self.camera_image_right is None:
            return

        if self.camera_info_left_param is None or self.camera_info_right_param is None:
            return
        
        img_left = self.bridge.imgmsg_to_cv2(self.camera_image_left, "bgr8")
        img_right = self.bridge.imgmsg_to_cv2(self.camera_image_right, "bgr8")

        camera_k_left = np.array(self.camera_info_left_param.k).reshape(3, 3)
        camera_d_left = np.array(self.camera_info_left_param.d)

        camera_k_right = np.array(self.camera_info_right_param.k).reshape(3, 3)
        camera_d_right = np.array(self.camera_info_right_param.d)

        left_result = self.aruco_board_detector.run(camera_k_left, camera_d_left, img_left)
        right_result = self.aruco_board_detector.run(camera_k_right, camera_d_right, img_right)

        if left_result is None or right_result is None:
            pass
        else:
            t_left, R_left = left_result
            t_right, R_right = right_result
            # self.get_logger().info(f"Left Marker Position: {t_left.ravel()}")
            # self.get_logger().info(f"Right Marker Position: {t_right.ravel()}")
            # self.get_logger().info("----")
            # self.get_logger().info(f"Left Marker Rotation:\n{R_left}")
            # self.get_logger().info(f"Right Marker Rotation:\n{R_right}")
            H_camleft2aruco = self._conver_tR_to_H(t_left, R_left) #  H_camleft_aruco
            H_camright2aruco = self._conver_tR_to_H(t_right, R_right)

            H_aruco2camleft = self.inverse_homogeneous_matrix(H_camleft2aruco) #  H_aruco_camleft
            H_aruco2camright = self.inverse_homogeneous_matrix(H_camright2aruco)

            # self.get_logger().info(f"camleft to aruco:\n{H_camleft2aruco}")
            # self.get_logger().info(f"camleft to world aruco:\n{H_aruco2camleft}")
            # self.get_logger().info(f"camright to world aruco:\n{H_aruco2camright}")
            self.get_logger().info("====")

            H_left2right = H_camleft2aruco @ H_aruco2camright
            self.get_logger().info(f"camleft to camright:\n{H_left2right}") # H_camleft_camright
            # Hrobot_camleft = np.eye(4)
            # Hrobot_camleft[0:3, 3] = np.array([0.1, 0.0, 0.0])
            
            # Hrobot_world = Hrobot_camleft @ H_aruco2camleft
            # self.get_logger().info(f"robot to world:\n{Hrobot_world}")
            # self.get_logger().info("====")

        cv2.imshow('Camera1 - ArUco Stereo', img_left)
        cv2.imshow('Camera2 - ArUco Stereo', img_right)
        cv2.waitKey(1)

    def _conver_tR_to_H(self, t, R):
        H = np.eye(4)
        H[0:3, 0:3] = R
        H[0:3, 3] = t.ravel()
        return H
    
    def inverse_homogeneous_matrix(self, H):
        """
        동차 변환 행렬의 효율적이고 안정적인 역행렬 계산
        H = [R t]  →  H^-1 = [R^T  -R^T*t]
            [0 1]              [0      1   ]
        """
        R = H[:3, :3]  # 회전 행렬 (3x3)
        t = H[:3, 3]   # 평행이동 벡터 (3x1)
        
        # 회전 행렬의 역행렬 = 전치행렬 (직교행렬이므로)
        R_inv = R.T
        
        # 평행이동의 역변환
        t_inv = -R_inv @ t
        
        # 역 동차 변환 행렬 구성
        H_inv = np.eye(4)
        H_inv[:3, :3] = R_inv
        H_inv[:3, 3] = t_inv
        
        return H_inv
        
def main(args=None):
    rclpy.init(args=args)
    stereo_aruco_ros = SteroArucoROS()
    rclpy.spin(stereo_aruco_ros)
    stereo_aruco_ros.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()