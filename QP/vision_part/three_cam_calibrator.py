import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
# Step 1에서 작성한 모듈을 임포트
from charuco_stereo_calib_module import detect_and_collect_charuco_data, perform_stereo_calibration, DICTIONARY

class ThreeCamCalibrator(Node):
    def __init__(self):
        super().__init__('three_cam_calibrator')
        self.bridge = CvBridge()
        self.min_samples = 20
        self.calib_pairs = [('camera1/camera1', 'camera2/camera2'), ('camera1/camera1', 'camera3/camera3'), ('camera2/camera2', 'camera3/camera3')]
        self.data_buffer = {pair: [] for pair in self.calib_pairs}
        self.K_matrices = {}
        self.latest_images = {}
        self.latest_timestamps = {}
        self.extrinsic_results = {} # 결과를 저장할 딕셔너리
        
        # 3 카메라 토픽 구독 설정
        for cam_id in ['camera1/camera1', 'camera2/camera2', 'camera3/camera3']:
            self.create_subscription(CameraInfo, f'/{cam_id}/color/camera_info', 
                                     lambda msg, c=cam_id: self.info_callback(msg, c), 1)
            self.create_subscription(Image, f'/{cam_id}/color/image_raw', 
                                     lambda msg, c=cam_id: self.image_callback(msg, c), 10)
        
        self.timer = self.create_timer(0.1, self.main_loop) # 10Hz 처리
        self.get_logger().info('Three-Camera Calibrator Node Started.')

    def info_callback(self, msg, cam_id):
        if cam_id not in self.K_matrices:
            self.K_matrices[cam_id] = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f'{cam_id} K matrix loaded.')
            
    def image_callback(self, msg, cam_id):
        self.latest_images[cam_id] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_timestamps[cam_id] = msg.header.stamp.nanosec

    def main_loop(self):
        K_ready = all(k in self.K_matrices for k in ['camera1/camera1', 'camera2/camera2', 'camera3/camera3'])
        Img_ready = all(img in self.latest_images for img in ['camera1/camera1', 'camera2/camera2', 'camera3/camera3'])
        
        if not (K_ready and Img_ready):
            self.get_logger().warn('Waiting for all CameraInfo and Image topics...')
            cv2.waitKey(1)
            return

        image_size = self.latest_images['camera1/camera1'].shape[:2][::-1]

        # 1. 데이터 수집 단계 (모든 쌍에 대해)
        for camA, camB in self.calib_pairs:
            if len(self.data_buffer[(camA, camB)]) < self.min_samples:
                
                # 타임스탬프 기반 동기화 확인 (10ms 허용 오차)
                # tsA = self.latest_timestamps[camA]
                # tsB = self.latest_timestamps[camB]
                # if abs(tsA - tsB) > 10000000:
                #     continue

                # 데이터 감지 및 수집
                ptsA, ptsB, obj_pts, cornersA, cornersB = detect_and_collect_charuco_data(
                    self.latest_images[camA], self.latest_images[camB], 
                    self.K_matrices[camA], self.K_matrices[camB]
                )

                # 시각화용 이미지 준비
                img_dispA = self.latest_images[camA].copy()
                img_dispB = self.latest_images[camB].copy()
                
                if ptsA is not None:
                    self.data_buffer[(camA, camB)].append((ptsA, ptsB, obj_pts))
                    self.get_logger().info(f'[{camA}-{camB}] Sample Collected: {len(self.data_buffer[(camA, camB)])}/{self.min_samples}')
                    cv2.drawChessboardCorners(img_dispA, (4, 3), ptsA, True)
                    cv2.drawChessboardCorners(img_dispB, (4, 3), ptsB, True)
                    status_text = "SAMPLE SAVED!"
                    color = (0, 255, 0) # Green
                else:
                    status_text = "Searching Board..."
                    color = (0, 0, 255) # Red

                # 디스플레이 업데이트
                cv2.putText(img_dispA, f"[{camA}-{camB}] Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img_dispA, f"Samples: {len(self.data_buffer[(camA, camB)])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f'View {camA}-{camB}', np.hstack((img_dispA, img_dispB)))
                cv2.waitKey(1)
        
        # 2. 캘리브레이션 및 검증 단계
        all_calibrated = all(len(self.data_buffer[pair]) >= self.min_samples for pair in self.calib_pairs)
        
        if all_calibrated:
            self.perform_all_calibrations_and_verify(image_size)
            rclpy.shutdown()

    def perform_all_calibrations_and_verify(self, image_size):
        # 1. 모든 쌍에 대한 캘리브레이션 실행 및 결과 저장
        for camA, camB in self.calib_pairs:
            self.get_logger().info(f"--- Calibrating {camA} <-> {camB} ---")
            
            data = self.data_buffer[(camA, camB)]
            obj_points = [d[2] for d in data]
            img_points_A = [d[0] for d in data]
            img_points_B = [d[1] for d in data]
            
            R, T, error = perform_stereo_calibration(
                obj_points, img_points_A, img_points_B, 
                self.K_matrices[camA], self.K_matrices[camB], image_size
            )
            
            if R is not None:
                self.extrinsic_results[(camA, camB)] = {'R': R, 'T': T, 'Error': error}
                self.get_logger().info(f"[{camA}-{camB}] Reprojection Error: {error:.4f}")
            else:
                self.get_logger().error(f"[{camA}-{camB}] Calibration Failed.")

        # 2. 교차 검증 수행
        self.verify_triangle_closure()
        
    def verify_triangle_closure(self):
        self.get_logger().info("\n--- Starting Triangle Closure Verification (C1-C2-C3) ---")
        
        # E_A_to_B 행렬 생성 함수 (4x4)
        def to_homogeneous(R, T):
            E = np.identity(4)
            E[:3, :3] = R
            E[:3, 3] = T.flatten()
            return E

        # E_C3_to_C2 측정값 (Measured)
        R_C3_C2_m = self.extrinsic_results[('camera2/camera2', 'camera3/camera3')]['R']
        T_C3_C2_m = self.extrinsic_results[('camera2/camera2', 'camera3/camera3')]['T']
        E_C3_C2_m = to_homogeneous(R_C3_C2_m, T_C3_C2_m)
        
        # E_C3_to_C2 계산값 (Calculated via C1)
        # E_C3_to_C2_c = E_C1_to_C2 * E_C3_to_C1
        R_C2_C1 = self.extrinsic_results[('camera1/camera1', 'camera2/camera2')]['R']
        T_C2_C1 = self.extrinsic_results[('camera1/camera1', 'camera2/camera2')]['T']
        E_C2_C1 = to_homogeneous(R_C2_C1, T_C2_C1)
        E_C1_C2 = np.linalg.inv(E_C2_C1) # 역행렬 (C2 -> C1 역)
        
        R_C3_C1 = self.extrinsic_results[('camera1/camera1', 'camera3/camera3')]['R']
        T_C3_C1 = self.extrinsic_results[('camera1/camera1', 'camera3/camera3')]['T']
        E_C3_C1 = to_homogeneous(R_C3_C1, T_C3_C1)

        E_C3_C2_c = E_C1_C2 @ E_C3_C1
        
        # 3. 오차 분석
        R_C3_C2_c = E_C3_C2_c[:3, :3]
        T_C3_C2_c = E_C3_C2_c[:3, 3].reshape(3, 1)

        # 회전 오차 계산 (라디안)
        R_diff = R_C3_C2_c @ R_C3_C2_m.T
        trace_R = np.trace(R_diff)
        # trace가 3보다 크면 작은 실수 오차가 발생한 것이므로 클램핑
        trace_R = min(3.0, max(-1.0, trace_R)) 
        rot_error_rad = math.acos((trace_R - 1) / 2)
        rot_error_deg = math.degrees(rot_error_rad)
        
        # 변이 오차 계산 (미터)
        trans_error_vec = T_C3_C2_c - T_C3_C2_m
        trans_error_m = np.linalg.norm(trans_error_vec)

        self.get_logger().info(f"--- Verification Results ---")
        self.get_logger().info(f"Calculated T (C3->C2): {T_C3_C2_c.flatten()}")
        self.get_logger().info(f"Measured T (C3->C2): {T_C3_C2_m.flatten()}")
        self.get_logger().info(f"Calculated R (C3->C2):\n{R_C3_C2_c}")
        self.get_logger().info(f"Measured R (C3->C2):\n{R_C3_C2_m}")
        
        self.get_logger().info(f"✅ Rotation Error: {rot_error_deg:.4f} degrees")
        self.get_logger().info(f"✅ Translation Error: {trans_error_m*1000:.4f} mm")
        
        # 최종 결과 출력 (C1 기준)
        self.get_logger().info("\n--- FINAL CAMERA POSITIONS (Relative to Cam1) ---")
        self.get_logger().info(f"C2 R:\n{R_C2_C1}")
        self.get_logger().info(f"C2 T (m):\n{T_C2_C1.flatten()}")
        self.get_logger().info(f"C3 R:\n{R_C3_C1}")
        self.get_logger().info(f"C3 T (m):\n{T_C3_C1.flatten()}")

# --- ROS 2 실행 ---
def main(args=None):
    rclpy.init(args=args)
    calibrator = ThreeCamCalibrator()
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows() 
        calibrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()