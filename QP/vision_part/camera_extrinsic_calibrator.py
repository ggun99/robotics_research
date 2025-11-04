#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class ExtrinsicCalibrator(Node):
    def __init__(self):
        super().__init__('extrinsic_calibrator')
        
        self.bridge = CvBridge()
        self.camera_info = {}
        self.calibrated_cameras = {}  # 캘리브레이션 결과 저장
        
        # 알려진 마커들 (월드 좌표계에서의 위치)
        self.reference_markers = {
            10: np.array([0.0, 0.0, 0.0]),    # 원점
            11: np.array([-0.105, 0.0, 0.0]),  # X축 12.8cm
            12: np.array([0.0, 0.128, 0.0]),  # Y축 10.5cm
            13: np.array([-0.105, 0.128, 0.0]) # 대각선
        }
        
        # ArUco 설정 - 버전 호환성 개선
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # OpenCV 버전에 따른 파라미터 생성
        try:
            # OpenCV 4.7 이후
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                # OpenCV 4.5-4.6
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                # 더 오래된 버전
                self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.marker_length = 0.05
        
        # OpenCV 버전 확인 및 출력
        cv_version = cv2.__version__
        self.get_logger().info(f"OpenCV version: {cv_version}")
        
        # 카메라 구독
        self.setup_subscribers()
        
        self.get_logger().info("Extrinsic Calibrator ready")
        self.get_logger().info("Place reference markers (ID 10-13) on the table and press 'c' to calibrate")
    
    def setup_subscribers(self):
        """카메라 구독 설정"""
        camera_names = ['camera1', 'camera2', 'camera3']
        
        for camera_name in camera_names:
            # 이미지 구독
            self.create_subscription(
                Image,
                f'/{camera_name}/{camera_name}/color/image_raw',
                lambda msg, name=camera_name: self.calibrate_camera(msg, name),
                10
            )
            
            # 카메라 정보 구독
            self.create_subscription(
                CameraInfo,
                f'/{camera_name}/{camera_name}/color/camera_info',
                lambda msg, name=camera_name: self.camera_info_callback(msg, name),
                10
            )
    
    def camera_info_callback(self, msg, camera_name):
        """카메라 정보 저장"""
        self.camera_info[camera_name] = {
            'camera_matrix': np.array(msg.k).reshape(3, 3),
            'dist_coeffs': np.array(msg.d)
        }
        # self.get_logger().info(f"Camera info received for {camera_name}")
    
    def calibrate_camera(self, msg, camera_name):
        """각 카메라의 외부 매개변수 캘리브레이션"""
        if camera_name not in self.camera_info:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # ArUco 마커 검출 - 버전 호환성 개선
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # OpenCV 버전에 따른 검출 방법
            try:
                # 최신 버전
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                # 이전 버전
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                # 시각화
                vis_image = cv_image.copy()
                cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
                
                # 검출된 마커 정보 표시
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in self.reference_markers:
                        corner = corners[i][0]
                        center = np.mean(corner, axis=0).astype(int)
                        cv2.putText(vis_image, f'ID:{marker_id}', 
                                  (center[0]-20, center[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 캘리브레이션 가능한지 확인
                detected_refs = [mid for mid in ids.flatten() if mid in self.reference_markers]
                status_text = f"Detected: {detected_refs} ({'c' if len(detected_refs) >= 4 else 'Need 4+'} to calibrate)"
                cv2.putText(vis_image, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if len(detected_refs) >= 4 else (0, 0, 255), 2)
                
                cv2.imshow(f'{camera_name}_calibration', vis_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c') and len(detected_refs) >= 4:  # 'c' 키를 누르면 캘리브레이션 수행
                    self.perform_calibration(corners, ids, camera_name)
                elif key == ord('q'):  # 'q' 키로 종료
                    rclpy.shutdown()
            else:
                # 마커가 검출되지 않은 경우
                vis_image = cv_image.copy()
                cv2.putText(vis_image, "No markers detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(f'{camera_name}_calibration', vis_image)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Calibration error for {camera_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_calibration(self, corners, ids, camera_name):
        """실제 캘리브레이션 수행"""
        cam_info = self.camera_info[camera_name]
        
        # 관측된 기준 마커들과 월드 좌표 매칭
        object_points = []
        image_points = []
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.reference_markers:
                # 월드 좌표
                world_pos = self.reference_markers[marker_id]
                object_points.append(world_pos)
                
                # 이미지 좌표 (마커 중심)
                corner = corners[i][0]
                center = np.mean(corner, axis=0)
                image_points.append(center)
        
        if len(object_points) >= 4:  # 최소 4개 점 필요
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            self.get_logger().info(f"Calibrating {camera_name} with {len(object_points)} markers")
            
            # solvePnP로 카메라 포즈 추정
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                cam_info['camera_matrix'],
                cam_info['dist_coeffs']
            )
            
            if success:
                # 회전 벡터를 회전 행렬로 변환
                R_cw, _ = cv2.Rodrigues(rvec)  # 월드 -> 카메라 회전 행렬
                t_cw = tvec.flatten()          # 월드 -> 카메라 평행이동
                
                # **올바른 변환**: 카메라 -> 월드 
                R_wc = R_cw.T                  # 카메라 -> 월드 회전 행렬
                t_wc = -R_cw.T @ t_cw         # 카메라의 월드 좌표 위치
                
                self.get_logger().info(f"\n=== {camera_name} Calibration Result ===")
                self.get_logger().info(f"Position (World): [{t_wc[0]:.4f}, {t_wc[1]:.4f}, {t_wc[2]:.4f}]")
                self.get_logger().info(f"Used markers: {[mid for mid in ids.flatten() if mid in self.reference_markers]}")
                
                # 검증: 카메라가 테이블 위에 있는지 확인
                if t_wc[2] < 0:
                    self.get_logger().warn(f"Warning: {camera_name} Z position is negative! Check marker placement.")
                
                # 추가 검증: 변환 테스트
                self.verify_transformation(object_points[0], image_points[0], 
                                        R_cw, t_cw, cam_info, camera_name)
                
                # 결과를 파일로 저장
                self.save_calibration_result(camera_name, t_wc, R_wc)
            else:
                self.get_logger().error(f"solvePnP failed for {camera_name}")
        else:
            self.get_logger().warn(f"Not enough reference markers for {camera_name} ({len(object_points)}/4)")

    def verify_transformation(self, world_point, image_point, R_cw, t_cw, cam_info, camera_name):
        """변환 검증"""
        # 월드 점을 카메라 좌표로 변환
        cam_point = R_cw @ world_point + t_cw
        
        # 카메라 좌표를 이미지 좌표로 투영
        if cam_point[2] > 0:  # 카메라 앞쪽에 있어야 함
            projected_point = cv2.projectPoints(
                world_point.reshape(1, 1, 3),
                np.zeros(3),  # 이미 카메라 좌표계
                np.zeros(3),
                cam_info['camera_matrix'],
                cam_info['dist_coeffs']
            )[0].flatten()
            
            error = np.linalg.norm(projected_point - image_point)
            self.get_logger().info(f"  Reprojection error: {error:.2f} pixels")
            
            if error > 5.0:
                self.get_logger().warn(f"  High reprojection error for {camera_name}!")
        else:
            self.get_logger().error(f"  Point is behind camera for {camera_name}!")
            
    def save_calibration_result(self, camera_name, position, rotation):
        """캘리브레이션 결과 저장 및 상대 위치 계산"""
        import yaml
        
        # 개별 카메라 결과 저장
        result = {
            'camera_name': camera_name,
            'position': position.tolist(),
            'rotation_matrix': rotation.tolist()
        }
        
        filename = f'{camera_name}_extrinsic.yaml'
        with open(filename, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        
        # 메모리에 저장
        self.calibrated_cameras[camera_name] = {
            'position': position,
            'rotation': rotation
        }
        
        self.get_logger().info(f"Calibration result saved to {filename}")
        
        # 모든 카메라가 캘리브레이션되면 상대 위치 계산
        if len(self.calibrated_cameras) >= 2:
            self.calculate_relative_poses()
    
    def calculate_relative_poses(self):
        """카메라들 간의 상대 위치 계산"""
        self.get_logger().info("\n=== Camera Relative Positions ===")
        
        camera_names = list(self.calibrated_cameras.keys())
        
        # 모든 카메라 쌍에 대해 상대 위치 계산
        for i in range(len(camera_names)):
            for j in range(i+1, len(camera_names)):
                cam1 = camera_names[i]
                cam2 = camera_names[j]
                
                pos1 = self.calibrated_cameras[cam1]['position']
                pos2 = self.calibrated_cameras[cam2]['position']
                
                # 거리 계산
                distance = np.linalg.norm(pos2 - pos1)
                
                self.get_logger().info(f"{cam1} ↔ {cam2}: Distance = {distance:.3f}m")
        
        # 전체 결과를 하나의 파일로 저장
        self.save_multi_camera_calibration()
    
    def save_multi_camera_calibration(self):
        """모든 카메라 캘리브레이션 결과를 통합 저장"""
        import yaml
        
        # 절대 위치 저장
        multi_camera_result = {
            'reference_frame': 'world',
            'reference_markers': {str(k): v.tolist() for k, v in self.reference_markers.items()},
            'cameras': {}
        }
        
        for camera_name, data in self.calibrated_cameras.items():
            multi_camera_result['cameras'][camera_name] = {
                'position': data['position'].tolist(),
                'rotation_matrix': data['rotation'].tolist()
            }
        
        # 파일 저장
        with open('multi_camera_calibration.yaml', 'w') as f:
            yaml.dump(multi_camera_result, f, default_flow_style=False)
        
        self.get_logger().info("Multi-camera calibration saved to multi_camera_calibration.yaml")

def main(args=None):
    rclpy.init(args=args)
    
    calibrator = ExtrinsicCalibrator()
    
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