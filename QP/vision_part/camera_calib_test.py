#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml

class ProjectionTest(Node):
    def __init__(self):
        super().__init__('projection_test')
        
        self.bridge = CvBridge()
        self.camera_info = {}
        self.camera_extrinsics = {}
        
        # 캘리브레이션 로드
        self.load_calibration()
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        self.marker_length = 0.075
        self.target_marker = 1  # 추적할 마커
        self.marker_world_pos = None
        
        # 구독자 설정
        self.setup_subscribers()
        
        print("Projection Test Ready - Place marker ID 1")

    def load_calibration(self):
        """캘리브레이션 파일 로드"""
        try:
            with open('multi_camera_calibration.yaml', 'r') as f:
                data = yaml.safe_load(f)
            
            for name, cam_data in data['cameras'].items():
                self.camera_extrinsics[name] = {
                    'position': np.array(cam_data['position']),
                    'rotation': np.array(cam_data['rotation_matrix'])
                }
            
            print(f"✅ Loaded calibration for: {list(self.camera_extrinsics.keys())}")
        except:
            print("❌ No calibration file found!")

    def setup_subscribers(self):
        """구독자 설정"""
        cameras = ['camera1', 'camera2', 'camera3']
        
        for cam_name in cameras:
            # 이미지
            self.create_subscription(
                Image, f'/{cam_name}/{cam_name}/color/image_raw',
                lambda msg, name=cam_name: self.image_callback(msg, name), 10
            )
            # 카메라 정보
            self.create_subscription(
                CameraInfo, f'/{cam_name}/{cam_name}/color/camera_info',
                lambda msg, name=cam_name: self.camera_info_callback(msg, name), 10
            )

    def camera_info_callback(self, msg, camera_name):
        """카메라 정보 저장"""
        self.camera_info[camera_name] = {
            'camera_matrix': np.array(msg.k).reshape(3, 3),
            'dist_coeffs': np.array(msg.d)
        }

    def image_callback(self, msg, camera_name):
        """이미지 처리"""
        if (camera_name not in self.camera_info or 
            camera_name not in self.camera_extrinsics):
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # ArUco 검출
            try:
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)
            except:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            display_image = cv_image.copy()
            
            # 타겟 마커 처리
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_marker:
                        # 실제 마커 표시
                        cv2.aruco.drawDetectedMarkers(display_image, [corners[i]], np.array([marker_id]))
                        
                        # 마커 중심
                        center = np.mean(corners[i][0], axis=0).astype(int)
                        cv2.circle(display_image, tuple(center), 8, (0, 255, 0), 3)
                        
                        # 월드 좌표 계산 (첫 번째 카메라에서만)
                        if camera_name == 'camera1':
                            self.calculate_world_position(corners[i], camera_name)
            
            # 월드 좌표가 있으면 투영
            if self.marker_world_pos is not None:
                projected_point = self.project_to_image(self.marker_world_pos, camera_name)
                if projected_point is not None:
                    x, y = projected_point
                    # 투영된 위치 표시 (빨간 십자가)
                    cv2.circle(display_image, (x, y), 10, (0, 0, 255), 3)
                    cv2.line(display_image, (x-20, y), (x+20, y), (0, 0, 255), 3)
                    cv2.line(display_image, (x, y-20), (x, y+20), (0, 0, 255), 3)
                    
                    cv2.putText(display_image, "Projected", (x+15, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 정보 표시
            cv2.putText(display_image, f"{camera_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'{camera_name} - Test', display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error in {camera_name}: {e}")

    def calculate_world_position(self, corner, camera_name):
        """월드 좌표 계산"""
        try:
            cam_info = self.camera_info[camera_name]
            
            # ArUco 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corner], self.marker_length,
                cam_info['camera_matrix'], cam_info['dist_coeffs']
            )
            
            # 카메라 좌표
            cam_pos = tvecs[0].flatten()
            
            # 월드 좌표로 변환
            extrinsic = self.camera_extrinsics[camera_name]
            R_wc = extrinsic['rotation']
            t_wc = extrinsic['position']
            
            world_pos = (R_wc @ cam_pos) + t_wc
            self.marker_world_pos = world_pos
            
        except Exception as e:
            print(f"World position calculation error: {e}")

    def project_to_image(self, world_point, camera_name):
        """월드 좌표를 이미지로 투영"""
        try:
            extrinsic = self.camera_extrinsics[camera_name]
            cam_info = self.camera_info[camera_name]
            
            R_wc = extrinsic['rotation'] 
            t_wc = extrinsic['position']
            
            # 월드 -> 카메라 변환
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            
            world_pt = np.array(world_point).reshape(3, 1)
            cam_pt = R_cw @ world_pt + t_cw.reshape(3, 1)
            cam_pt = cam_pt.flatten()
            
            # 카메라 앞쪽 확인
            if cam_pt[2] <= 0:
                return None
            
            # 이미지로 투영
            img_pts, _ = cv2.projectPoints(
                cam_pt.reshape(1, 1, 3),
                np.zeros(3), np.zeros(3),
                cam_info['camera_matrix'], cam_info['dist_coeffs']
            )
            
            x, y = img_pts[0][0]
            return (int(x), int(y))
            
        except Exception as e:
            print(f"Projection error: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    
    tester = ProjectionTest()
    
    try:
        print("Testing projection... Press Ctrl+C to stop")
        rclpy.spin(tester)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()