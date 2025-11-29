import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from multicam_aruco import ARUCOBoardPose
import yaml
import json
import os
from datetime import datetime
        
class SteroArucoROS(Node):

    def __init__(self):
        super().__init__('stereo_aruco_ros')
        self.bridge = CvBridge()
        self.aruco_board_detector = ARUCOBoardPose()

        # ✅ 저장 관련 변수
        self.H_left2right_samples = []  # 여러 측정값 저장
        self.current_H_left2right = None
        self.save_count = 0
        self.min_samples = 10  # 평균을 위한 최소 샘플 수

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

        # ✅ 저장 디렉토리 생성
        self.save_dir = "stereo_calibration_results"
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info("Stereo ArUco ROS Node Initialized")
        self.get_logger().info("=== Control Keys ===")
        self.get_logger().info("  's' - Save current H_left2right")
        self.get_logger().info("  'q' - Quit")

    def camera_info_callback_left(self, msg:CameraInfo):
        if self.camera_info_left_param is None:
            self.camera_info_left_param = msg
            
    def camera_info_callback_right(self, msg:CameraInfo):
        if self.camera_info_right_param is None:
            self.camera_info_right_param = msg
    
    def image_left_callback(self, msg:Image):
        self.camera_image_left = msg
 
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
            # ✅ 검출 실패 시 안내
            if left_result is None:
                cv2.putText(img_left, "ArUco Board Not Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if right_result is None:
                cv2.putText(img_right, "ArUco Board Not Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            t_left, R_left = left_result
            t_right, R_right = right_result
            
            H_camleft2aruco = self._conver_tR_to_H(t_left, R_left)
            H_camright2aruco = self._conver_tR_to_H(t_right, R_right)
            H_aruco2camleft = self.inverse_homogeneous_matrix(H_camleft2aruco)
            H_aruco2camright = self.inverse_homogeneous_matrix(H_camright2aruco)

            H_left2right = H_camleft2aruco @ H_aruco2camright
            self.current_H_left2right = H_left2right.copy()
            
            # ✅ 기본 정보 출력
            baseline_distance = np.linalg.norm(H_left2right[:3, 3])
            self.get_logger().info(f"Current baseline distance: {baseline_distance:.4f}m")
            
            # ✅ 이미지에 정보 표시
            cv2.putText(img_left, f"Baseline: {baseline_distance:.3f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_left, f"Samples: {len(self.H_left2right_samples)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img_left, "Press 's':save, 'c':collect, 'a':average", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(img_right, f"Baseline: {baseline_distance:.3f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_right, f"Samples: {len(self.H_left2right_samples)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ✅ 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and self.current_H_left2right is not None:
            self.save_current_transformation()

        elif key == ord('q'):
            self.get_logger().info("Shutting down...")
            rclpy.shutdown()

        cv2.imshow('camera1 - ArUco Stereo', img_left)
        cv2.imshow('camera2 - ArUco Stereo', img_right)


    def save_current_transformation(self):
        """현재 변환 행렬을 즉시 저장"""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ✅ 여러 형식으로 저장
        self._save_as_yaml(self.current_H_left2right, f"H_camera1_camera2_current")

        baseline = np.linalg.norm(self.current_H_left2right[:3, 3])
        self.get_logger().info(f"💾 Current transformation saved! Baseline: {baseline:.4f}m")


    def _save_as_yaml(self, H_matrix, filename, extra_info=None):
        """YAML 형식으로 저장"""
        data = {
            'stereo_transformation': {
                'from_frame': 'camera_left',
                'to_frame': 'camera_right',
                'transformation_matrix': H_matrix.tolist(),
                'translation': {
                    'x': float(H_matrix[0, 3]),
                    'y': float(H_matrix[1, 3]), 
                    'z': float(H_matrix[2, 3])
                },
                'rotation_matrix': H_matrix[:3, :3].tolist(),
                'baseline_distance_m': float(np.linalg.norm(H_matrix[:3, 3])),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if extra_info:
            data['stereo_transformation'].update(extra_info)
        
        filepath = os.path.join(self.save_dir, f"{filename}.yaml")
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        
        self.get_logger().info(f"📄 YAML saved: {filepath}")


    def _save_statistics(self, H_samples, filename):
        """샘플 통계 저장"""
        baselines = [np.linalg.norm(H[:3, 3]) for H in H_samples]
        translations = np.array([H[:3, 3] for H in H_samples])
        
        stats = {
            'statistics': {
                'num_samples': len(H_samples),
                'baseline_distance': {
                    'mean_m': float(np.mean(baselines)),
                    'std_m': float(np.std(baselines)),
                    'min_m': float(np.min(baselines)),
                    'max_m': float(np.max(baselines))
                },
                'translation_std': {
                    'x_m': float(np.std(translations[:, 0])),
                    'y_m': float(np.std(translations[:, 1])),
                    'z_m': float(np.std(translations[:, 2]))
                },
                'all_baselines': baselines,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        filepath = os.path.join(self.save_dir, f"{filename}.yaml")
        with open(filepath, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, indent=2)
        
        self.get_logger().info(f"📈 Statistics saved: {filepath}")
        self.get_logger().info(f"   Mean baseline: {stats['statistics']['baseline_distance']['mean_m']:.4f} ± {stats['statistics']['baseline_distance']['std_m']:.4f}m")

    def _conver_tR_to_H(self, t, R):
        H = np.eye(4)
        H[0:3, 0:3] = R
        H[0:3, 3] = t.ravel()
        return H
    
    def inverse_homogeneous_matrix(self, H):
        """동차 변환 행렬의 효율적이고 안정적인 역행렬 계산"""
        R = H[:3, :3]
        t = H[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        H_inv = np.eye(4)
        H_inv[:3, :3] = R_inv
        H_inv[:3, 3] = t_inv
        return H_inv

        
def main(args=None):
    rclpy.init(args=args)
    stereo_aruco_ros = SteroArucoROS()
    
    try:
        rclpy.spin(stereo_aruco_ros)
    except KeyboardInterrupt:
        stereo_aruco_ros.get_logger().info("🛑 Keyboard interrupt received")
    finally:
        cv2.destroyAllWindows()
        stereo_aruco_ros.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()