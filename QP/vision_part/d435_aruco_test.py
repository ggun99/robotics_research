#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import yaml
import os
from scipy.spatial.transform import Rotation as R

class SingleMarkerTracker(Node):
    def __init__(self):
        super().__init__('single_marker_tracker')

        # ROS2
        self.bridge = CvBridge()
        self.marker_pose_pub = self.create_publisher(PoseStamped, '/marker_1_world_pose', 10)

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.075  # 마커 크기 [m]
        self.target_marker_id = 1  # 추적할 마커 ID

        # 카메라 설정
        self.detection_camera = 'camera1'  # 1번 마커를 감지할 카메라
        self.verification_cameras = ['camera2', 'camera3']  # 검증용 카메라들
        
        # 카메라 정보 저장
        self.camera_info = {}  # camera_name -> {camera_matrix, dist_coeffs, width, height}
        self.camera_extrinsics = {}  # camera_name -> {position, rotation}
        
        # 마커 위치 저장
        self.marker_world_pose = None
        self.marker_timestamp = 0.0
        
        # 캘리브레이션 로드
        self.load_camera_calibrations()
        
        # 구독자 설정
        self.setup_subscribers()
        
        self.get_logger().info(f"Single Marker Tracker initialized - tracking marker {self.target_marker_id}")
        self.get_logger().info(f"Detection camera: {self.detection_camera}")
        self.get_logger().info(f"Verification cameras: {self.verification_cameras}")

    def load_camera_calibrations(self):
        """카메라 캘리브레이션 로드"""
        # ✅ 먼저 기본값 설정 (이 부분이 빠져있었습니다!)
        self.camera_extrinsics = {
            'camera1': {
                'position': np.array([0.0, -1.5, 1.2]), 
                'rotation': np.eye(3)
            },
            'camera2': {
                'position': np.array([1.3, -0.75, 1.2]), 
                'rotation': R.from_euler('xyz', [0, 0, -np.pi/6]).as_matrix()
            },
            'camera3': {
                'position': np.array([-1.3, -0.75, 1.2]), 
                'rotation': R.from_euler('xyz', [0, 0, np.pi/6]).as_matrix()
            }
        }
        # 파일 경로 확인
        yaml_path = '/home/airlab/robotics_research/QP/multi_camera_calibration.yaml'
        
        self.get_logger().info(f"Looking for calibration file: {yaml_path}")
        self.get_logger().info(f"File exists: {os.path.exists(yaml_path)}")
        
        # 현재 작업 디렉토리도 확인
        current_dir = os.getcwd()
        local_yaml = os.path.join(current_dir, 'multi_camera_calibration.yaml')
        self.get_logger().info(f"Current directory: {current_dir}")
        self.get_logger().info(f"Local yaml exists: {os.path.exists(local_yaml)}")
        
        # 캘리브레이션 파일 로드 시도
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                self.get_logger().info(f"YAML data loaded: {list(data.keys())}")
                
                for name, cam in data.get('cameras', {}).items():
                    if name in self.camera_extrinsics:
                        self.camera_extrinsics[name]['position'] = np.array(cam['position'])
                        self.camera_extrinsics[name]['rotation'] = np.array(cam['rotation_matrix'])
                        self.get_logger().info(f"Loaded calibration for {name}")
                
                self.get_logger().info("✅ Camera calibrations loaded from file")
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to load calibration file: {e}")
                self.get_logger().info("Using default calibration values")
        
        elif os.path.exists(local_yaml):
            # 현재 디렉토리에서 파일 찾기
            try:
                with open(local_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                
                for name, cam in data.get('cameras', {}).items():
                    if name in self.camera_extrinsics:
                        self.camera_extrinsics[name]['position'] = np.array(cam['position'])
                        self.camera_extrinsics[name]['rotation'] = np.array(cam['rotation_matrix'])
                
                self.get_logger().info("✅ Camera calibrations loaded from local file")
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to load local calibration file: {e}")
        
        else:
            self.get_logger().warn("⚠️  No calibration file found, using default values")
            
        # 최종 확인
        self.get_logger().info("=== Final Camera Extrinsics ===")
        for name, extrinsic in self.camera_extrinsics.items():
            pos = extrinsic['position']
            self.get_logger().info(f"{name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    def setup_subscribers(self):
        """구독자 설정"""
        cameras_to_use = [self.detection_camera] + self.verification_cameras
        
        for camera_name in cameras_to_use:
            # 이미지 토픽
            image_topic = f'/{camera_name}/{camera_name}/color/image_raw'
            self.create_subscription(
                Image, image_topic, 
                lambda msg, name=camera_name: self.image_callback(msg, name), 10
            )
            
            # 카메라 정보 토픽
            info_topic = f'/{camera_name}/{camera_name}/color/camera_info'
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, name=camera_name: self.camera_info_callback(msg, name), 10
            )
            
            self.get_logger().info(f"Subscribed to {image_topic}")

    def camera_info_callback(self, msg: CameraInfo, camera_name):
        """카메라 정보 콜백"""
        try:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            d = np.array(msg.d, dtype=float)
            w = int(msg.width) if hasattr(msg, 'width') else 640
            h = int(msg.height) if hasattr(msg, 'height') else 480
            
            self.camera_info[camera_name] = {
                'camera_matrix': K,
                'dist_coeffs': d,
                'width': w,
                'height': h
            }
        except Exception as e:
            self.get_logger().error(f"Camera info callback error for {camera_name}: {e}")

    def image_callback(self, msg: Image, camera_name):
        """이미지 콜백"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if camera_name == self.detection_camera:
                # 감지 카메라: 마커를 찾고 월드 좌표 계산
                self.detect_and_track_marker(cv_image, camera_name)
            else:
                # 검증 카메라: 계산된 월드 좌표를 이미지에 투영하여 검증
                self.verify_marker_projection(cv_image, camera_name)
                
        except Exception as e:
            self.get_logger().error(f"Image callback error for {camera_name}: {e}")

    def detect_and_track_marker(self, image, camera_name):
        """감지 카메라에서 마커 찾기 및 월드 좌표 계산"""
        if camera_name not in self.camera_info or camera_name not in self.camera_extrinsics:
            return
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ArUco 마커 감지
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        # 결과 이미지 준비
        result_image = image.copy()
        
        if ids is not None:
            # 타겟 마커 찾기
            target_idx = None
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_marker_id:
                    target_idx = i
                    break
            
            if target_idx is not None:
                # 마커 포즈 추정
                cam_info = self.camera_info[camera_name]
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[target_idx]], self.marker_length,
                        cam_info['camera_matrix'], cam_info['dist_coeffs']
                    )
                    
                    # 카메라 좌표계 → 월드 좌표계 변환
                    cam_pos = tvecs[0].flatten()
                    extrinsic = self.camera_extrinsics[camera_name]
                    R_wc = extrinsic['rotation']
                    t_wc = extrinsic['position']
                    
                    # 월드 좌표 계산
                    world_pos = (R_wc @ cam_pos) + t_wc
                    self.marker_world_pose = world_pos
                    self.marker_timestamp = time.time()
                    
                    # 결과 발행
                    self.publish_marker_pose(world_pos)
                    
                    # 시각화
                    cv2.aruco.drawDetectedMarkers(result_image, corners, ids)
                    cv2.drawFrameAxes(result_image, cam_info['camera_matrix'], 
                                    cam_info['dist_coeffs'], rvecs[0], tvecs[0], 0.1)
                    
                    # 정보 표시
                    cv2.putText(result_image, f"Marker {self.target_marker_id} found", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_image, f"World: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    self.get_logger().error(f"Pose estimation error: {e}")
        
        # 이미지 표시
        cv2.imshow(f'{camera_name}_detection', result_image)
        cv2.waitKey(1)

    def verify_marker_projection(self, image, camera_name):
        """검증 카메라에서 계산된 월드 좌표를 이미지에 투영"""
        if (camera_name not in self.camera_info or 
            camera_name not in self.camera_extrinsics or
            self.marker_world_pose is None):
            cv2.imshow(f'{camera_name}_verification', image)
            cv2.waitKey(1)
            return
        
        # 시간 체크 (0.5초 이내의 데이터만 사용)
        if time.time() - self.marker_timestamp > 0.5:
            cv2.imshow(f'{camera_name}_verification', image)
            cv2.waitKey(1)
            return
        
        result_image = image.copy()
        
        # 월드 좌표를 이미지 좌표로 투영
        projected_point = self.project_world_to_image(self.marker_world_pose, camera_name)
        
        if projected_point is not None:
            x, y = projected_point
            
            # 투영된 위치에 십자가 그리기
            cv2.circle(result_image, (x, y), 10, (0, 0, 255), 3)
            cv2.line(result_image, (x-20, y), (x+20, y), (0, 0, 255), 3)
            cv2.line(result_image, (x, y-20), (x, y+20), (0, 0, 255), 3)
            
            # 정보 표시
            cv2.putText(result_image, f"Projected Marker {self.target_marker_id}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_image, f"Image: ({x}, {y})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(result_image, f"World: ({self.marker_world_pose[0]:.3f}, {self.marker_world_pose[1]:.3f}, {self.marker_world_pose[2]:.3f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 실제 마커도 감지해서 비교
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            # 타겟 마커가 실제로도 보이는지 확인
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_marker_id:
                    # 실제 마커 그리기
                    cv2.aruco.drawDetectedMarkers(result_image, [corners[i]], np.array([marker_id]))
                    
                    # 마커 중심 계산
                    marker_center = np.mean(corners[i][0], axis=0).astype(int)
                    cv2.circle(result_image, tuple(marker_center), 8, (0, 255, 0), 3)
                    
                    # 투영 위치와 실제 위치 차이 계산
                    if projected_point is not None:
                        diff = np.linalg.norm(np.array(projected_point) - marker_center)
                        cv2.putText(result_image, f"Error: {diff:.1f} pixels", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # 연결선 그리기
                        cv2.line(result_image, projected_point, tuple(marker_center), (255, 0, 0), 2)
                    
                    break
        
        cv2.imshow(f'{camera_name}_verification', result_image)
        cv2.waitKey(1)

    def project_world_to_image(self, world_point, camera_name):
        """월드 좌표를 이미지 좌표로 투영"""
        if camera_name not in self.camera_extrinsics or camera_name not in self.camera_info:
            return None
        
        extrinsic = self.camera_extrinsics[camera_name]
        R_wc = extrinsic['rotation']
        t_wc = extrinsic['position']
        
        # 월드 → 카메라 좌표 변환
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        
        world_pt = np.array(world_point).reshape(3, 1)
        cam_pt = R_cw @ world_pt + t_cw.reshape(3, 1)
        cam_pt = cam_pt.flatten()
        
        # 카메라 앞쪽에 있는지 확인
        if cam_pt[2] <= 0:
            return None
        
        # 이미지 좌표로 투영
        cam_info = self.camera_info[camera_name]
        try:
            img_pts, _ = cv2.projectPoints(
                cam_pt.reshape(1, 1, 3), 
                np.zeros(3), np.zeros(3),
                cam_info['camera_matrix'], 
                cam_info['dist_coeffs']
            )
            x, y = img_pts[0][0]
            
            # 이미지 범위 내에 있는지 확인
            if 0 <= x < cam_info['width'] and 0 <= y < cam_info['height']:
                return (int(x), int(y))
        except Exception as e:
            self.get_logger().error(f"Projection error: {e}")
        
        return None

    def publish_marker_pose(self, world_pos):
        """마커 월드 좌표 발행"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(world_pos[0])
        msg.pose.position.y = float(world_pos[1])
        msg.pose.position.z = float(world_pos[2])
        msg.pose.orientation.w = 1.0
        
        self.marker_pose_pub.publish(msg)
        self.get_logger().info(f'Marker {self.target_marker_id} world position: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})')

def main(args=None):
    rclpy.init(args=args)
    node = SingleMarkerTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()