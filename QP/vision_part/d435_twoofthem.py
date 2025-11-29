#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import yaml

class StereoArucoCalibrator(Node):
    def __init__(self):
        super().__init__('stereo_aruco_calibrator')
        self.bridge = CvBridge()
        
        # ✅ ArUco 마커 설정 (75mm, ID 10-13)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        self.marker_length = 0.075  # 75mm
        
        # ✅ 4개 마커의 실제 위치 (월드 좌표계)
        self.reference_markers = {
            10: np.array([0.0, 0.0, 0.0]),      # 원점
            11: np.array([0.10, 0.0, 0.0]),     # X축 10cm
            12: np.array([0.0, -0.10, 0.0]),    # Y축 -10cm  
            13: np.array([0.10, -0.10, 0.0])    # 대각선
        }
        
        self.valid_marker_ids = [10, 11, 12, 13]
        
        self.min_data_samples = 15  # 최소 수집 데이터 쌍
        self.data_samples = []      # 수집된 데이터 저장 리스트
        
        # 카메라 정보
        self.cam1_info = None
        self.cam2_info = None
        
        # 이미지 버퍼
        self.latest_img1 = None
        self.latest_img2 = None
        self.latest_ts1 = 0
        self.latest_ts2 = 0
        
        # ROS 2 토픽 구독 설정
        self.image1_sub = self.create_subscription(
            Image, '/camera1/camera1/color/image_raw', self.image1_callback, 10)
        self.image2_sub = self.create_subscription(
            Image, '/camera2/camera2/color/image_raw', self.image2_callback, 10)
        self.info1_sub = self.create_subscription(
            CameraInfo, '/camera1/camera1/color/camera_info', self.info1_callback, 1)
        self.info2_sub = self.create_subscription(
            CameraInfo, '/camera2/camera2/color/camera_info', self.info2_callback, 1)
        
        print("\n" + "="*60)
        print("=== Stereo ArUco Calibrator (75mm) ===")
        print("📋 Required:")
        print("   - ArUco markers: ID 10, 11, 12, 13")
        print("   - Size: 75mm x 75mm each")
        print("   - Family: DICT_4X4_50")
        print("📍 Marker Layout:")
        print("   ID 10: (0,0)      ID 11: (10cm,0)")
        print("   ID 12: (0,-10cm)  ID 13: (10cm,-10cm)")
        print(f"🎯 Target: {self.min_data_samples} stereo pairs")
        print("="*60)
        
        self.get_logger().info('ArUco Stereo Calibrator Started!')

    # --- 카메라 정보 콜백 ---
    def info1_callback(self, msg):
        if self.cam1_info is None:
            self.cam1_info = np.array(msg.k).reshape((3, 3))
            self.get_logger().info('Camera1 K matrix loaded.')
            
    def info2_callback(self, msg):
        if self.cam2_info is None:
            self.cam2_info = np.array(msg.k).reshape((3, 3))
            self.get_logger().info('Camera2 K matrix loaded.')

    # --- 이미지 콜백 ---
    def image1_callback(self, msg):
        self.latest_img1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_ts1 = msg.header.stamp.nanosec

    def image2_callback(self, msg):
        self.latest_img2 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_ts2 = msg.header.stamp.nanosec

    def detect_aruco_markers(self, gray_image):
        """ArUco 마커 검출 - ID 10-13만 필터링"""
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_image)
        except:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_image, self.aruco_dict, parameters=self.aruco_params)
        
        # ID 10-13만 필터링
        if ids is not None:
            filtered_corners = []
            filtered_ids = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.valid_marker_ids:
                    filtered_corners.append(corners[i])
                    filtered_ids.append(marker_id)
            
            if len(filtered_ids) > 0:
                return filtered_corners, np.array(filtered_ids).reshape(-1, 1)
        
        return [], None

    def process_frames(self):
        """메인 처리 루프"""
        # 준비 상태 확인
        if self.cam1_info is None or self.cam2_info is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        if self.latest_img1 is None or self.latest_img2 is None:
            cv2.waitKey(1)
            return

        # 동기화 확인
        sync_threshold_ns = 10000000  # 5ms
        if abs(self.latest_ts1 - self.latest_ts2) > sync_threshold_ns:
            self.display_status_images("Sync Error")
            return

        # 이미지 처리
        img1 = self.latest_img1.copy()
        img2 = self.latest_img2.copy()
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # ArUco 마커 검출
        corners1, ids1 = self.detect_aruco_markers(gray1)
        corners2, ids2 = self.detect_aruco_markers(gray2)
        
        data_collected = False
        common_markers = []
        
        # 양쪽 카메라에서 마커가 검출되었는지 확인
        if (ids1 is not None and ids2 is not None and 
            len(ids1) >= 3 and len(ids2) >= 3):  # 최소 3개 마커 필요
            
            # 공통 마커 찾기
            common_markers = self.find_common_markers(ids1, ids2)
            
            if len(common_markers) >= 3:  # 최소 3개 공통 마커
                # 스테레오 쌍 데이터 수집
                success = self.collect_stereo_data(
                    corners1, ids1, corners2, ids2, common_markers)
                
                if success:
                    self.get_logger().info(
                        f'✅ Sample {len(self.data_samples)}/{self.min_data_samples} collected '
                        f'with markers: {sorted(common_markers)}')
                    data_collected = True
        
        # 시각화
        self.display_status_images(common_markers, data_collected)
        
        # 캘리브레이션 시작 조건
        if len(self.data_samples) >= self.min_data_samples:
            self.perform_stereo_calibration()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def find_common_markers(self, ids1, ids2):
        """두 카메라에서 공통으로 검출된 마커 찾기"""
        common = []
        for id1 in ids1.flatten():
            if id1 in ids2.flatten() and id1 in self.valid_marker_ids:
                common.append(id1)
        return common

    def collect_stereo_data(self, corners1, ids1, corners2, ids2, common_markers):
        """스테레오 캘리브레이션 데이터 수집"""
        try:
            # 3D 객체점과 2D 이미지점 수집
            object_points = []
            image_points1 = []
            image_points2 = []
            
            # ✅ 각 마커는 4개의 코너를 가지므로 모든 코너를 사용
            for marker_id in common_markers:
                if marker_id in self.reference_markers:
                    # 마커 중심의 3D 위치
                    marker_center_3d = self.reference_markers[marker_id]
                    
                    # 마커의 4개 코너 3D 좌표 (마커 중심 기준)
                    half_size = self.marker_length / 2
                    marker_corners_3d = np.array([
                        marker_center_3d + [-half_size, -half_size, 0],  # 좌하
                        marker_center_3d + [ half_size, -half_size, 0],  # 우하
                        marker_center_3d + [ half_size,  half_size, 0],  # 우상
                        marker_center_3d + [-half_size,  half_size, 0]   # 좌상
                    ])
                    
                    # 카메라1에서 해당 마커의 이미지 코너 찾기
                    idx1 = np.where(ids1.flatten() == marker_id)[0][0]
                    corner1 = corners1[idx1][0]  # 4x2 array
                    
                    # 카메라2에서 해당 마커의 이미지 코너 찾기
                    idx2 = np.where(ids2.flatten() == marker_id)[0][0]
                    corner2 = corners2[idx2][0]  # 4x2 array
                    
                    # 각 마커의 4개 코너점 추가
                    object_points.extend(marker_corners_3d)
                    image_points1.extend(corner1)
                    image_points2.extend(corner2)
            
            if len(object_points) >= 12:  # 최소 3개 마커 = 12개 점
                self.data_samples.append({
                    'object_points': np.array(object_points, dtype=np.float32),
                    'image_points1': np.array(image_points1, dtype=np.float32),
                    'image_points2': np.array(image_points2, dtype=np.float32),
                    'markers': common_markers.copy()
                })
                return True
        
        except Exception as e:
            self.get_logger().error(f'Data collection error: {e}')
        
        return False

    def display_status_images(self, common_markers=None, collected=False):
        """상태 시각화"""
        if self.latest_img1 is None or self.latest_img2 is None:
            return
            
        img1 = self.latest_img1.copy()
        img2 = self.latest_img2.copy()
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 각 카메라에서 ArUco 검출 및 그리기
        self.draw_aruco_detection(img1, gray1, "Camera1")
        self.draw_aruco_detection(img2, gray2, "Camera2")
        
        # 상태 텍스트
        if isinstance(common_markers, str):  # 에러 메시지
            status_text = common_markers
            color = (0, 0, 255)  # 빨간색
        elif common_markers and len(common_markers) >= 3:
            if collected:
                status_text = f"✅ COLLECTED! Markers: {sorted(common_markers)}"
                color = (0, 255, 0)  # 초록색
            else:
                status_text = f"🎯 Ready to collect: {sorted(common_markers)}"
                color = (0, 255, 255)  # 노란색
        else:
            status_text = "❌ Need 3+ common markers (10,11,12,13)"
            color = (0, 0, 255)  # 빨간색
        
        samples_text = f"Samples: {len(self.data_samples)} / {self.min_data_samples}"
        
        # 텍스트 표시
        cv2.putText(img1, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img1, samples_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img2, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img2, samples_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 진행률 표시
        if len(self.data_samples) >= self.min_data_samples:
            cv2.putText(img1, "🚀 Starting Calibration...", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img2, "🚀 Starting Calibration...", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Camera1 - ArUco Stereo', img1)
        cv2.imshow('Camera2 - ArUco Stereo', img2)
        cv2.waitKey(1)

    def draw_aruco_detection(self, color_img, gray_img, camera_name):
        """ArUco 마커 검출 결과 그리기"""
        corners, ids = self.detect_aruco_markers(gray_img)
        
        if ids is not None and len(ids) > 0:
            # 마커 그리기
            colors = {10: (255, 0, 0), 11: (0, 255, 0), 12: (0, 0, 255), 13: (255, 255, 0)}
            
            for i, marker_id in enumerate(ids.flatten()):
                color = colors.get(marker_id, (128, 128, 128))
                
                # 마커 테두리
                corner = corners[i][0].astype(int)
                cv2.polylines(color_img, [corner], True, color, 3)
                
                # 마커 중심에 ID 표시
                center = np.mean(corner, axis=0).astype(int)
                cv2.circle(color_img, tuple(center), 8, color, -1)
                cv2.putText(color_img, f'ID{marker_id}', 
                           (center[0]-15, center[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 실제 위치 정보
                if marker_id in self.reference_markers:
                    world_pos = self.reference_markers[marker_id]
                    pos_text = f'({world_pos[0]*100:.0f},{world_pos[1]*100:.0f})cm'
                    cv2.putText(color_img, pos_text, 
                               (center[0]-25, center[1]+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 검출 정보
            detected_ids = sorted(ids.flatten().tolist())
            cv2.putText(color_img, f"Found: {detected_ids}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(color_img, "No ArUco markers detected", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 카메라 이름 표시
        h, w = color_img.shape[:2]
        cv2.putText(color_img, camera_name, (w//2 - 60, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(color_img, camera_name, (w//2 - 60, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    def perform_stereo_calibration(self):
        """스테레오 캘리브레이션 수행"""
        self.get_logger().info('🔄 Starting Stereo Calibration...')
        
        # 데이터 준비
        all_object_points = []
        all_image_points1 = []
        all_image_points2 = []
        
        for sample in self.data_samples:
            all_object_points.append(sample['object_points'])
            all_image_points1.append(sample['image_points1'])
            all_image_points2.append(sample['image_points2'])
        
        # 카메라 매개변수
        K1 = self.cam1_info
        K2 = self.cam2_info
        D1 = np.zeros((5, 1))  # 왜곡 보정 가정
        D2 = np.zeros((5, 1))
        
        image_size = self.latest_img1.shape[:2][::-1]  # (W, H)
        
        try:
            # 스테레오 캘리브레이션 수행
            ret, K1_new, D1_new, K2_new, D2_new, R, T, E, F = cv2.stereoCalibrate(
                objectPoints=all_object_points,
                imagePoints1=all_image_points1,
                imagePoints2=all_image_points2,
                cameraMatrix1=K1,
                distCoeffs1=D1,
                cameraMatrix2=K2,
                distCoeffs2=D2,
                imageSize=image_size,
                flags=cv2.CALIB_FIX_INTRINSIC,  # 내부 매개변수 고정
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-6)
            )
            
            self.get_logger().info('✅ Stereo Calibration Successful!')
            self.get_logger().info(f'📊 Reprojection Error: {ret:.3f} pixels')
            
            if ret < 2.0:
                quality = "Excellent"
                status = "✅"
            elif ret < 5.0:
                quality = "Good"
                status = "✅"
            elif ret < 10.0:
                quality = "Acceptable"
                status = "⚠️"
            else:
                quality = "Poor"
                status = "❌"
            
            self.get_logger().info(f'{status} Quality: {quality}')
            
            # 결과 출력
            distance = np.linalg.norm(T)
            self.get_logger().info(f'📏 Camera distance: {distance:.3f}m')
            self.get_logger().info(f'📍 Translation (Camera2 relative to Camera1):')
            self.get_logger().info(f'   X: {T[0,0]:.3f}m, Y: {T[1,0]:.3f}m, Z: {T[2,0]:.3f}m')
            
            # 회전각 계산 (디버깅용)
            from scipy.spatial.transform import Rotation as Rot
            rotation = Rot.from_matrix(R)
            angles = rotation.as_euler('xyz', degrees=True)
            self.get_logger().info(f'🔄 Rotation angles (deg): X:{angles[0]:.1f}, Y:{angles[1]:.1f}, Z:{angles[2]:.1f}')
            
            # 결과 저장
            self.save_calibration_result(R, T, ret)
            
        except Exception as e:
            self.get_logger().error(f'❌ Stereo Calibration Failed: {e}')

    def save_calibration_result(self, R, T, error):
        """캘리브레이션 결과 저장"""
        result = {
            'stereo_calibration': {
                'method': 'ArUco_75mm_stereo',
                'marker_ids': [10, 11, 12, 13],
                'marker_size_mm': 75,
                'num_samples': len(self.data_samples),
                'reprojection_error_pixels': float(error),
                'camera_pair': 'camera1_to_camera2'
            },
            'extrinsics': {
                'rotation_matrix': R.tolist(),
                'translation_vector': T.flatten().tolist(),
                'camera_distance_m': float(np.linalg.norm(T))
            },
            'reference_markers': {
                str(k): v.tolist() for k, v in self.reference_markers.items()
            }
        }
        
        # 파일 저장
        with open('stereo_aruco_calibration.yaml', 'w') as f:
            yaml.dump(result, f, default_flow_style=False, indent=2)
        
        self.get_logger().info('📁 Results saved to stereo_aruco_calibration.yaml')

def main(args=None):
    rclpy.init(args=args)
    calibrator = StereoArucoCalibrator()
    
    # 타이머로 주기적 처리
    calibrator.create_timer(0.1, calibrator.process_frames)
    
    try:
        print("🚀 ArUco Stereo Calibrator running... Press Ctrl+C to stop")
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("\n🛑 Calibration interrupted")
    finally:
        cv2.destroyAllWindows()
        calibrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()