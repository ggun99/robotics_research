#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os

class SimpleCalibrator(Node):
    def __init__(self):
        super().__init__('simple_calibrator')
        
        self.bridge = CvBridge()
        self.camera_info = {}
        self.calibrated_cameras = {}
        
        # ✅ 75mm ArUco 마커들의 실제 위치 (측정해서 정확히 입력하세요!)
        self.reference_markers = {
            10: np.array([0.0, 0.0, 0.0]),      # 원점 (기준점)
            11: np.array([0.10, 0.0, 0.0]),     # X축 10cm 오른쪽
            12: np.array([0.0, -0.10, 0.0]),    # Y축 10cm 뒤쪽  
            13: np.array([0.10, -0.10, 0.0])    # 대각선 위치
        }
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # ✅ 75mm 마커 크기
        self.marker_length = 0.075  # 75mm = 7.5cm
        
        # 사용할 마커 ID 리스트
        self.valid_marker_ids = [10, 11, 12, 13]
        
        # 카메라 구독
        self.setup_subscribers()
        
        print("\n" + "="*60)
        print("=== ArUco 4x4 75mm Calibrator ===")
        print("📋 Required ArUco Markers:")
        print("   - Family: DICT_4X4_50")
        print("   - Size: 75mm x 75mm") 
        print("   - IDs: 10, 11, 12, 13")
        print("\n📍 Marker Placement:")
        print("   ID 10: (0,0)     - Origin point")
        print("   ID 11: (10cm,0)  - X-axis reference") 
        print("   ID 12: (0,-10cm) - Y-axis reference")
        print("   ID 13: (10cm,-10cm) - Diagonal corner")
        print("\n🎯 Usage:")
        print("1. Place all 4 markers on flat surface")
        print("2. Press 'c' in camera window to calibrate")
        print("3. Press 'v' to verify marker distances")
        print("4. Press 'q' to quit")
        print("="*60)

    def setup_subscribers(self):
        """카메라 구독자 설정"""
        cameras = ['camera1', 'camera2', 'camera3']
        
        for cam_name in cameras:
            # 이미지 토픽
            image_topic = f'/{cam_name}/{cam_name}/color/image_raw'
            self.create_subscription(
                Image, image_topic,
                lambda msg, name=cam_name: self.image_callback(msg, name), 10
            )
            
            # 카메라 정보 토픽
            info_topic = f'/{cam_name}/{cam_name}/color/camera_info'  
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, name=cam_name: self.camera_info_callback(msg, name), 10
            )
            
            print(f"📷 Subscribed to: {image_topic}")

    def camera_info_callback(self, msg, camera_name):
        """카메라 정보 저장"""
        self.camera_info[camera_name] = {
            'camera_matrix': np.array(msg.k).reshape(3, 3),
            'dist_coeffs': np.array(msg.d)
        }

    def detect_aruco_markers(self, gray_image):
        """ArUco 마커 검출 - ID 10-13만 필터링"""
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_image)
        except:
            corners, ids, _ = cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.aruco_params)
        
        # ✅ ID 10-13만 필터링
        if ids is not None:
            filtered_corners = []
            filtered_ids = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.valid_marker_ids:
                    filtered_corners.append(corners[i])
                    filtered_ids.append(marker_id)
            
            if len(filtered_ids) > 0:
                return filtered_corners, np.array(filtered_ids).reshape(-1, 1)
            else:
                return [], None
        
        return [], None

    def image_callback(self, msg, camera_name):
        """이미지 콜백 - 마커 검출 및 캘리브레이션"""
        if camera_name not in self.camera_info:
            return
        
        try:
            # 이미지 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # ArUco 마커 검출 (ID 10-13만)
            corners, ids = self.detect_aruco_markers(gray)
            
            # 시각화 이미지 준비
            display_image = cv_image.copy()
            
            if ids is not None and len(ids) > 0:
                # ✅ 마커 그리기 - 각 ID별로 다른 색상
                colors = {10: (255, 0, 0), 11: (0, 255, 0), 12: (0, 0, 255), 13: (255, 255, 0)}
                
                found_markers = []
                for i, marker_id in enumerate(ids.flatten()):
                    found_markers.append(marker_id)
                    
                    # 마커 테두리 그리기
                    corner = corners[i][0]
                    corner_int = corner.astype(int)
                    color = colors.get(marker_id, (128, 128, 128))
                    cv2.polylines(display_image, [corner_int], True, color, 3)
                    
                    # 마커 중심에 정보 표시
                    center = np.mean(corner, axis=0).astype(int)
                    cv2.circle(display_image, tuple(center), 8, color, -1)
                    cv2.putText(display_image, f'ID{marker_id}', 
                               (center[0]-20, center[1]-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 실제 위치 정보 표시
                    world_pos = self.reference_markers.get(marker_id)
                    if world_pos is not None:
                        pos_text = f'({world_pos[0]*100:.0f},{world_pos[1]*100:.0f})cm'
                        cv2.putText(display_image, pos_text,
                                   (center[0]-30, center[1]+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # ✅ 상태 표시
                found_markers.sort()
                missing_markers = [mid for mid in self.valid_marker_ids if mid not in found_markers]
                
                status_color = (0, 255, 0) if len(found_markers) == 4 else (0, 165, 255)
                cv2.putText(display_image, f"Found: {found_markers}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                if len(found_markers) == 4:
                    cv2.putText(display_image, "✓ All markers detected! Press 'c' to calibrate", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif len(missing_markers) > 0:
                    cv2.putText(display_image, f"Missing: {missing_markers}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # ✅ 마커 품질 체크
                quality_ok = True
                for i, corner in enumerate(corners):
                    area = cv2.contourArea(corner[0])
                    if area < 500:  # 너무 작은 마커
                        quality_ok = False
                        cv2.putText(display_image, f"ID{ids[i][0]} too small!", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if not quality_ok:
                    cv2.putText(display_image, "⚠️ Poor marker quality - move closer", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(display_image, "No ArUco markers (10-13) detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_image, "Make sure markers are 75mm, DICT_4X4_50", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # ✅ 캘리브레이션 상태 표시
            if camera_name in self.calibrated_cameras:
                error = self.calibrated_cameras[camera_name]['error']
                status = "✓ GOOD" if error < 5.0 else "⚠️ HIGH ERROR"
                cv2.putText(display_image, f"{camera_name}: {status} ({error:.1f}px)", 
                           (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 이미지 표시
            cv2.imshow(f'{camera_name} - ArUco 75mm Calibration', display_image)
            
            # ✅ 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ids is not None:
                if len(ids) == 4:  # 모든 마커가 보일 때만
                    self.calibrate_camera(corners, ids, camera_name)
                else:
                    print(f"❌ Need all 4 markers for {camera_name} (got {len(ids)})")
            elif key == ord('v'):  # 마커 거리 검증
                self.verify_marker_distances()
            elif key == ord('r'):  # 기준 위치 재설정
                self.reset_reference_markers()
            elif key == ord('q'):
                rclpy.shutdown()
                
        except Exception as e:
            print(f"Error in {camera_name}: {e}")

    def calibrate_camera(self, corners, ids, camera_name):
        """카메라 캘리브레이션 수행 - 75mm 마커 사용"""
        print(f"\n=== Calibrating {camera_name} with 75mm ArUco markers ===")
        
        cam_info = self.camera_info[camera_name]
        
        # ✅ 마커별 3D-2D 대응점 생성
        object_points = []
        image_points = []
        used_markers = []
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.reference_markers:
                # 3D 월드 좌표
                world_pos = self.reference_markers[marker_id]
                object_points.append(world_pos)
                
                # 2D 이미지 좌표 (마커 중심)
                corner = corners[i][0]
                center = np.mean(corner, axis=0)
                image_points.append(center)
                used_markers.append(marker_id)
                
                print(f"  Marker {marker_id}: World{world_pos} -> Image{center}")
        
        if len(object_points) == 4:
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            print(f"Using all markers: {sorted(used_markers)}")
            
            # ✅ solvePnP로 카메라 포즈 추정
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                cam_info['camera_matrix'], cam_info['dist_coeffs']
            )
            
            if success:
                # 회전 행렬 변환
                R_cw, _ = cv2.Rodrigues(rvec)
                t_cw = tvec.flatten()
                
                # 카메라의 월드 위치 계산 (카메라 좌표계 -> 월드 좌표계)
                R_wc = R_cw.T
                t_wc = -R_cw.T @ t_cw
                
                print(f"📍 Position: [{t_wc[0]:.3f}, {t_wc[1]:.3f}, {t_wc[2]:.3f}]m")
                print(f"📏 Height: {t_wc[2]:.3f}m")
                print(f"📐 Distance from origin: {np.linalg.norm(t_wc):.3f}m")
                
                # ✅ 재투영 오차 계산
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec,
                    cam_info['camera_matrix'], cam_info['dist_coeffs']
                )
                projected = projected.reshape(-1, 2)
                
                errors = np.linalg.norm(projected - image_points, axis=1)
                avg_error = np.mean(errors)
                max_error = np.max(errors)
                
                print(f"📊 Reprojection errors:")
                for i, (mid, err) in enumerate(zip(used_markers, errors)):
                    print(f"   Marker {mid}: {err:.2f}px")
                print(f"📈 Average: {avg_error:.2f}px, Max: {max_error:.2f}px")
                
                # ✅ 품질 평가
                if avg_error <= 2.0:
                    quality = "Excellent"
                    color = "✅"
                elif avg_error <= 5.0:
                    quality = "Good"
                    color = "✅"
                elif avg_error <= 10.0:
                    quality = "Acceptable"  
                    color = "⚠️"
                else:
                    quality = "Poor"
                    color = "❌"
                
                print(f"{color} Quality: {quality}")
                
                if avg_error <= 10.0:  # 10픽셀 이하면 저장
                    self.save_result(camera_name, t_wc, R_wc, avg_error)
                    print(f"✅ {camera_name} calibrated and saved!")
                else:
                    print(f"❌ Error too high for {camera_name}! Check marker placement.")
                    print("   - Ensure markers are flat and well-lit")
                    print("   - Verify marker size is exactly 75mm")
                    print("   - Check camera focus and stability")
            else:
                print(f"❌ solvePnP failed for {camera_name}")
        else:
            print(f"❌ Wrong number of markers ({len(object_points)}/4)")

    def verify_marker_distances(self):
        """기준 마커들 간의 거리 검증"""
        print(f"\n=== Verifying 75mm ArUco Marker Layout ===")
        
        markers = list(self.reference_markers.keys())
        
        print("📏 Expected distances between markers:")
        for i in range(len(markers)):
            for j in range(i+1, len(markers)):
                id1, id2 = markers[i], markers[j]
                pos1 = self.reference_markers[id1]
                pos2 = self.reference_markers[id2]
                
                distance = np.linalg.norm(pos2 - pos1)
                print(f"   Marker {id1} ↔ {id2}: {distance*1000:.1f}mm")
        
        print("\n🔍 Please measure these distances with a ruler!")
        print("   If distances don't match, press 'r' to reset positions")

    def reset_reference_markers(self):
        """기준 마커 위치 재설정"""
        print(f"\n=== Resetting Reference Marker Positions ===")
        print("Enter new positions for 75mm markers (in cm):")
        
        try:
            # 간단한 예시 - 실제로는 측정값 입력
            new_positions = {
                10: [0.0, 0.0, 0.0],      # 원점
                11: [10.0, 0.0, 0.0],     # 10cm 오른쪽
                12: [0.0, -10.0, 0.0],    # 10cm 뒤쪽
                13: [10.0, -10.0, 0.0]    # 대각선
            }
            
            for marker_id, pos_cm in new_positions.items():
                pos_m = np.array([p/100.0 for p in pos_cm])  # cm -> m 변환
                self.reference_markers[marker_id] = pos_m
                print(f"   Marker {marker_id}: {pos_cm} cm -> {pos_m} m")
            
            print("✅ Reference positions updated!")
            
        except Exception as e:
            print(f"❌ Failed to reset positions: {e}")

    def save_result(self, camera_name, position, rotation, error):
        """결과 저장"""
        # 메모리에 저장
        self.calibrated_cameras[camera_name] = {
            'position': position,
            'rotation': rotation,
            'error': error
        }
        
        # 개별 파일 저장
        result = {
            'camera': camera_name,
            'marker_type': 'ArUco_4x4_75mm',
            'marker_ids_used': [10, 11, 12, 13],
            'position': position.tolist(),
            'rotation_matrix': rotation.tolist(),
            'reprojection_error_pixels': float(error)
        }
        
        with open(f'{camera_name}_75mm_aruco.yaml', 'w') as f:
            yaml.dump(result, f, default_flow_style=False, indent=2)
        
        # 통합 파일 저장
        self.save_multi_camera_file()

    def save_multi_camera_file(self):
        """통합 캘리브레이션 파일 저장"""
        multi_data = {
            'calibration_info': {
                'marker_type': 'ArUco 4x4',
                'marker_size_mm': 75,
                'marker_family': 'DICT_4X4_50',
                'marker_ids': [10, 11, 12, 13]
            },
            'reference_markers': {str(k): v.tolist() for k, v in self.reference_markers.items()},
            'cameras': {}
        }
        
        for name, data in self.calibrated_cameras.items():
            multi_data['cameras'][name] = {
                'position': data['position'].tolist(),
                'rotation_matrix': data['rotation'].tolist(),
                'reprojection_error': data['error']
            }
        
        with open('multi_camera_calibration.yaml', 'w') as f:
            yaml.dump(multi_data, f, default_flow_style=False, indent=2)
        
        print(f"📁 Saved: multi_camera_calibration.yaml")
        
        # ✅ 카메라 간 거리 출력
        if len(self.calibrated_cameras) >= 2:
            print("\n=== Inter-Camera Distances ===")
            names = list(self.calibrated_cameras.keys())
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    pos1 = self.calibrated_cameras[names[i]]['position'] 
                    pos2 = self.calibrated_cameras[names[j]]['position']
                    dist = np.linalg.norm(pos2 - pos1)
                    print(f"📏 {names[i]} ↔ {names[j]}: {dist:.3f}m")

def main(args=None):
    rclpy.init(args=args)
    
    calibrator = SimpleCalibrator()
    
    try:
        print("🚀 ArUco 75mm Calibrator started... Press Ctrl+C to stop")
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        cv2.destroyAllWindows()
        calibrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()