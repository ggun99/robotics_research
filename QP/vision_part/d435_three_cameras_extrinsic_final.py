#!/usr/bin/env python3

import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import time
import tf2_ros
from scipy.spatial.transform import Rotation as R
import yaml
import os
import rclpy.time
import rclpy.duration

class D435ArucoDetectorWithExtrinsic(Node):
    def __init__(self):
        super().__init__('d435_aruco_detector_extrinsic')
        
        # 로봇 마커들 (0~3번) - 로컬 좌표계
        self.robot_markers_local = {
            0: np.array([0.2875, 0.1225, 0.0]),
            1: np.array([0.2875, -0.1225, 0.0]),
            2: np.array([-0.2875, -0.1225, 0.0]),
            3: np.array([-0.2875, 0.1225, 0.0])
        }
    
        self.marker_length = 0.075  # 마커 크기
        
        # 월드 좌표계 설정 상태
        self.world_established = False
        self.world_from_marker = False  # 마커로부터 설정되었는지 여부
        self.world_from_extrinsic = False  # extrinsic으로부터 설정되었는지 여부
        
        # 🆕 Extrinsic calibration 데이터
        self.extrinsic_data = {}
        self.load_extrinsic_calibration()
        
        # 카메라별 정보 저장
        self.cameras = {
            'camera1': {
                'frame_id': 'camera1_color_optical_frame',
                'camera_matrix': None,
                'dist_coeffs': None,
                'info_received': False,
                'H_world2cam': None,
                'latest_detections': {},
                'detection_timestamp': None
            },
            'camera2': {
                'frame_id': 'camera2_color_optical_frame', 
                'camera_matrix': None,
                'dist_coeffs': None,
                'info_received': False,
                'H_world2cam': None,
                'latest_detections': {},
                'detection_timestamp': None
            },
            'camera3': {
                'frame_id': 'camera3_color_optical_frame',
                'camera_matrix': None,
                'dist_coeffs': None, 
                'info_received': False,
                'H_world2cam': None,
                'latest_detections': {},
                'detection_timestamp': None
            }
        }
        
        # 로봇 중심 추적
        self.latest_robot_center = None
        self.center_timestamp = None

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 변환 행렬들
        self.H_cam2robot = None
        self.H_world2robot = None
        
        # 월드 기준 카메라 관련
        self.world_reference_camera = None
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # 🆕 TF 리스너 추가 (검증용)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 🆕 각 카메라별 구독자 설정
        self.setup_camera_subscriptions()
        
        # 발행자
        self.robot_center_pub = self.create_publisher(
            PoseStamped, '/robot_center', 10
        )
        
        # 🆕 주기적으로 extrinsic 기반 TF 브로드캐스트
        self.create_timer(0.1, self.broadcast_extrinsic_transforms)  # 10Hz
        
        # 🆕 Extrinsic vs TF 비교 검증 (5초마다)
        self.create_timer(0.5, self.validate_extrinsic_vs_tf)  # 0.2Hz
        self.validation_results = {}
        
        self.get_logger().info("D435 ArUco Detector with Extrinsic Support initialized")

    def load_extrinsic_calibration(self):
        """Extrinsic calibration 데이터 로드"""
        try:
            # 통합 파일 시도
            if os.path.exists('multi_camera_calibration.yaml'):
                with open('multi_camera_calibration.yaml', 'r') as f:
                    data = yaml.safe_load(f)
                
                if 'cameras' in data:
                    for camera_name, camera_data in data['cameras'].items():
                        position = np.array(camera_data['position'])
                        rotation = np.array(camera_data['rotation_matrix'])
                        
                        # 🔧 올바른 월드 → 카메라 변환 행렬 구성
                        # Extrinsic calibration에서 position, rotation은 월드에서 카메라의 위치/자세
                        # 따라서 H_cam2world를 먼저 만들고 역변환
                        H_cam2world = np.eye(4)
                        H_cam2world[0:3, 0:3] = rotation
                        H_cam2world[0:3, 3] = position
                        
                        H_world2cam = np.linalg.inv(H_cam2world)
                        
                        self.extrinsic_data[camera_name] = {
                            'position': position,
                            'rotation': rotation,
                            'H_world2cam': H_world2cam
                        }
                        
                        self.get_logger().info(f"📋 Loaded extrinsic data for {camera_name}")
                
                if len(self.extrinsic_data) > 0:
                    self.get_logger().info(f"✅ Loaded extrinsic calibration for {len(self.extrinsic_data)} cameras")
                    # extrinsic 데이터로 월드 좌표계 설정
                    self.setup_world_from_extrinsic()
                else:
                    self.get_logger().warn("⚠️ No camera data found in multi_camera_calibration.yaml")
                    
        except Exception as e:
            self.get_logger().warn(f"⚠️ Could not load extrinsic calibration: {e}")
            self.get_logger().info("Will use marker-based calibration when available")

    def setup_world_from_extrinsic(self):
        """Extrinsic 데이터를 사용해서 월드 좌표계 설정"""
        if len(self.extrinsic_data) > 0:
            self.world_established = True
            self.world_from_extrinsic = True
            
            # 각 카메라의 H_world2cam 설정
            for camera_name, extrinsic in self.extrinsic_data.items():
                if camera_name in ['camera1', 'camera2', 'camera3']:
                    self.cameras[camera_name]['H_world2cam'] = extrinsic['H_world2cam']
            
            self.get_logger().info("🌍 World coordinate system established from extrinsic calibration")

    def setup_camera_subscriptions(self):
        """각 카메라별 구독자 설정 - 클로저 문제 해결"""
        
        def make_image_callback(camera_name):
            """클로저 문제 해결을 위한 콜백 생성 함수"""
            return lambda msg: self.image_callback(msg, camera_name)
        
        def make_info_callback(camera_name):
            """클로저 문제 해결을 위한 정보 콜백 생성 함수"""
            return lambda msg: self.camera_info_callback(msg, camera_name)
        
        for camera_name in self.cameras.keys():
            # 이미지 구독 - 개별 콜백 함수 생성
            self.create_subscription(
                Image, 
                f'/{camera_name}/{camera_name}/color/image_raw', 
                make_image_callback(camera_name), 
                10
            )
            
            # 카메라 정보 구독 - 개별 콜백 함수 생성
            self.create_subscription(
                CameraInfo, 
                f'/{camera_name}/{camera_name}/color/camera_info',
                make_info_callback(camera_name), 
                10
            )
            
            self.get_logger().info(f"📷 Subscribed to {camera_name}")

    def camera_info_callback(self, msg, camera_name):
        """카메라 내부 매개변수 수신"""
        camera_config = self.cameras[camera_name]
        
        if not camera_config['info_received']:
            camera_config['camera_matrix'] = np.array(msg.k).reshape(3, 3)
            camera_config['dist_coeffs'] = np.array(msg.d)
            camera_config['info_received'] = True
            self.get_logger().info(f"📷 {camera_name} intrinsics received")

    def image_callback(self, msg, camera_name):
        """각 카메라별 이미지 처리 및 ArUco 검출"""
        
        # 🔧 카메라 이름 검증
        if camera_name not in self.cameras:
            self.get_logger().warn(f"❌ Unknown camera name: {camera_name}")
            return
            
        camera_config = self.cameras[camera_name]
        
        if not camera_config['info_received']:
            return
            
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ArUco 마커 검출
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            # 검출 실패시 타임스탬프만 업데이트
            camera_config['latest_detections'] = {}
            camera_config['detection_timestamp'] = time.time()
            self.visualize_results(cv_image, {}, camera_name)
            return

        # 포즈 추정 (해당 카메라 좌표계)
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, 
                camera_config['camera_matrix'], 
                camera_config['dist_coeffs']
            )
        except Exception as e:
            self.get_logger().error(f"{camera_name} pose estimation failed: {e}")
            return

        # 검출된 마커 정리
        detected_markers = {}
        for i, marker_id in enumerate(ids.flatten()):
            detected_markers[marker_id] = {
                'cam_tvec': tvecs[i].reshape(3),
                'cam_rvec': rvecs[i].reshape(3),
                'corners': corners[i],
                'camera_name': camera_name
            }

        # 🆕 10번 마커 기반 월드 좌표계 업데이트 (우선순위)
        self.update_world_from_marker(detected_markers, camera_config, camera_name, msg.header.stamp)

        # 🆕 검출 결과 저장
        camera_config['latest_detections'] = detected_markers
        camera_config['detection_timestamp'] = time.time()

        # 🤖 로봇 중심 계산
        self.calculate_robot_center(detected_markers, camera_config, camera_name, msg.header.stamp)

        # 시각화
        self.visualize_results(cv_image, detected_markers, camera_name)

    def update_world_from_marker(self, detected_markers, camera_config, camera_name, timestamp):
        """10번 마커를 발견했을 때 월드 좌표계 업데이트 (최고 우선순위)"""
        
        for marker_id, data in detected_markers.items():
            if marker_id == 10:
                rvec = data['cam_rvec']
                tvec = data['cam_tvec']
                
                # 마커 기준 월드 좌표계 계산
                H_cam2world = np.eye(4)
                R_matrix, _ = cv2.Rodrigues(rvec)
                H_cam2world[0:3, 0:3] = R_matrix
                H_cam2world[0:3, 3] = tvec
                
                H_world2cam = np.linalg.inv(H_cam2world)
                camera_config['H_world2cam'] = H_world2cam
                
                # 🔧 카메라 TF는 타이머에서 처리 (중복 방지)
                
                # 첫 번째 마커 기반 월드 설정
                if not self.world_from_marker:
                    self.world_established = True
                    self.world_from_marker = True
                    self.world_reference_camera = camera_name
                    
                    # 기존 extrinsic 기반 설정 무효화
                    if self.world_from_extrinsic:
                        self.get_logger().info("🔄 Switching from extrinsic to marker-based world coordinate system")
                        self.world_from_extrinsic = False
                    
                    self.get_logger().info(f"✅ Marker-based world coordinate system established by {camera_name}")
                
                break

    def calculate_robot_center(self, detected_markers, camera_config, camera_name, timestamp):
        """로봇 중심 계산"""
        
        if not self.world_established:
            return
            
        # 로봇 마커들 수집
        robot_markers_cam = {}
        for marker_id, data in detected_markers.items():
            if marker_id in self.robot_markers_local:
                rvec = data['cam_rvec']
                tvec = data['cam_tvec']
                R_matrix, _ = cv2.Rodrigues(rvec)
                robot_markers_cam[marker_id] = {
                    'position': tvec,
                    'rotation': R_matrix
                }

        if len(robot_markers_cam) > 0:
            # 하나의 마커를 기준으로 로봇 중심 계산
            robot_center_cam, robot_rotation_cam = self.calculate_robot_center_from_single_marker(robot_markers_cam)
            
            # H_cam2robot
            H_cam2robot = np.eye(4)
            H_cam2robot[0:3, 0:3] = robot_rotation_cam
            H_cam2robot[0:3, 3] = robot_center_cam
            self.H_cam2robot = H_cam2robot
            
            # 월드 좌표계로 변환 - 원본과 동일하게
            H_world2cam = camera_config['H_world2cam']
            if H_world2cam is not None:
                H_world2robot = H_world2cam @ H_cam2robot
                robot_center_world = H_world2robot[:3, 3]
                
                # 🔧 원본과 동일하게 timestamp와 camera_name 전달
                self.broadcast_robot_transform(H_world2robot, camera_name, timestamp)
                self.latest_robot_center = robot_center_world
                self.center_timestamp = time.time()
                
                self.get_logger().info(f"🤖 Robot center (cam): {robot_center_cam}")
                self.get_logger().info(f"🌍 Robot center (world): {robot_center_world}")
                self.publish_robot_center(robot_center_world, H_world2robot)

    def calculate_robot_center_from_single_marker(self, robot_markers_cam):
        """하나의 마커를 기준으로 로컬 좌표계 오프셋을 이용해 로봇 중심 계산"""
        
        # 가장 신뢰할만한 마커 선택
        best_marker_id = min(robot_markers_cam.keys())
        best_marker_data = robot_markers_cam[best_marker_id]
        
        marker_position_cam = best_marker_data['position']
        marker_rotation_cam = best_marker_data['rotation']
        
        # 로컬 좌표계에서의 오프셋
        marker_offset_local = self.robot_markers_local[best_marker_id]
        
        # 로컬 오프셋을 카메라 좌표계로 변환
        marker_offset_cam = marker_rotation_cam @ marker_offset_local
        
        # 로봇 중심 = 마커 위치 - 마커 오프셋
        robot_center_cam = marker_position_cam - marker_offset_cam
        
        return robot_center_cam, marker_rotation_cam

    def broadcast_extrinsic_transforms(self):
        """🔧 안정적인 TF 브로드캐스트 - 타임스탬프 통일"""
        
        if not self.world_established:
            return
        
        # 🔧 타임스탬프 통일
        timestamp = self.get_clock().now().to_msg()
        
        # 🔧 디버깅용 상태 확인
        current_time = time.time()
        active_cameras = []
        
        # 모든 카메라에 대해 TF 브로드캐스트
        for camera_name in ['camera1', 'camera2', 'camera3']:
            camera_config = self.cameras[camera_name]
            
            # 카메라별 상태 체크
            detection_age = float('inf')
            if camera_config.get('detection_timestamp'):
                detection_age = current_time - camera_config['detection_timestamp']
            
            # 우선순위 1: 마커 기반 데이터 (실시간) - 최근 데이터만 사용
            if (camera_config['H_world2cam'] is not None and 
                self.world_from_marker and 
                camera_config.get('detection_timestamp') is not None and
                detection_age < 0.3):  # 🔧 0.3초로 엄격하게
                
                # 🔧 유효성 검사
                H_world2cam = camera_config['H_world2cam']
                if not np.any(np.isnan(H_world2cam)) and not np.any(np.isinf(H_world2cam)):
                    self.broadcast_camera_transform_from_marker(
                        camera_name, 
                        H_world2cam, 
                        timestamp
                    )
                    active_cameras.append(f"{camera_name}(M)")
                else:
                    self.get_logger().warn(f"⚠️ Invalid H_world2cam for {camera_name}")
            
            # 우선순위 2: Extrinsic 데이터 (백업) - 안정적
            elif camera_name in self.extrinsic_data and self.world_from_extrinsic:
                self.broadcast_camera_transform_from_extrinsic(
                    camera_name, 
                    None, None,  # 사용하지 않음
                    timestamp
                )
                active_cameras.append(f"{camera_name}(E)")
        
        # 🔧 주기적 상태 로깅 (5초마다)
        if not hasattr(self, 'last_status_log'):
            self.last_status_log = 0
        
        if current_time - self.last_status_log > 5.0:
            self.get_logger().info(f"📊 Active cameras: {', '.join(active_cameras) if active_cameras else 'None'}")
            self.last_status_log = current_time

    def broadcast_camera_transform_from_extrinsic(self, camera_name, position, rotation, timestamp):
        """Extrinsic 데이터를 사용해서 카메라 TF 브로드캐스트 - 올바른 변환"""
        
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'{camera_name}_frame'
        
        # 🔧 올바른 변환: H_world2cam 사용 (마커 기반과 동일한 방식)
        H_world2cam = self.extrinsic_data[camera_name]['H_world2cam']
        
        t.transform.translation.x = float(H_world2cam[0, 3])
        t.transform.translation.y = float(H_world2cam[1, 3])
        t.transform.translation.z = float(H_world2cam[2, 3])
        
        R_mat = H_world2cam[0:3, 0:3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def broadcast_camera_transform_from_marker(self, camera_name, H_world2cam, timestamp):
        """마커 데이터를 사용해서 카메라 TF 브로드캐스트 - 작동하는 버전과 동일하게"""
        
        t = TransformStamped()
        t.header.stamp = timestamp  # 🔧 통일된 타임스탬프 사용
        t.header.frame_id = 'world'
        t.child_frame_id = f'{camera_name}_frame'
        
        # 🔧 작동하는 버전과 동일하게: H_world2cam을 직접 사용
        t.transform.translation.x = float(H_world2cam[0, 3])
        t.transform.translation.y = float(H_world2cam[1, 3])
        t.transform.translation.z = float(H_world2cam[2, 3])
        
        R_mat = H_world2cam[0:3, 0:3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def broadcast_robot_transform(self, H_world2robot, camera_name, timestamp):
        """로봇 좌표계 tf 브로드캐스트 - 원본과 동일"""
        
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'world'  # 모든 로봇은 같은 world를 기준으로
        t.child_frame_id = 'robot_center'
        
        # 위치
        t.transform.translation.x = float(H_world2robot[0, 3])
        t.transform.translation.y = float(H_world2robot[1, 3])
        t.transform.translation.z = float(H_world2robot[2, 3])
        
        # 회전 (H_world2robot을 그대로 사용)
        R_mat = H_world2robot[0:3, 0:3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
        
        self.get_logger().info(f"🤖 Broadcasting robot_center from world (detected by {camera_name})")

    def validate_extrinsic_vs_tf(self):
        """🔍 Extrinsic 데이터와 실제 TF 좌표계 비교 검증"""
        
        if not self.world_established:
            return
            
        # 월드 기준 카메라가 설정되어 있고, 마커 기반 월드가 활성화된 경우에만 검증
        if not self.world_from_marker or self.world_reference_camera is None:
            return
            
        try:
            validation_results = {}
            
            # 각 카메라에 대해 검증
            for camera_name in ['camera1', 'camera2', 'camera3']:
                if camera_name == self.world_reference_camera:
                    continue  # 기준 카메라는 제외
                    
                result = self.compare_camera_position(camera_name)
                if result is not None:
                    validation_results[camera_name] = result
            
            # 결과 저장 및 출력
            if validation_results:
                self.validation_results = validation_results
                self.log_validation_results(validation_results)
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ Validation failed: {e}")

    def compare_camera_position(self, target_camera):
        """특정 카메라의 extrinsic vs TF 위치 비교"""
        
        try:
            # 1. 현재 TF에서 카메라 위치 가져오기
            tf_transform = self.tf_buffer.lookup_transform(
                'world', f'{target_camera}_frame', 
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # 🔧 TF에서 가져온 것은 H_world2cam 형태의 변환
            tf_H_cam2world_translation = np.array([
                tf_transform.transform.translation.x,
                tf_transform.transform.translation.y,
                tf_transform.transform.translation.z
            ])
            
            tf_H_cam2world_rotation = R.from_quat([
                tf_transform.transform.rotation.x,
                tf_transform.transform.rotation.y,
                tf_transform.transform.rotation.z,
                tf_transform.transform.rotation.w
            ]).as_matrix()
            
            # 🔧 카메라의 실제 월드 좌표 계산: H_cam2world 구성
            tf_H_cam2world = np.eye(4)
            tf_H_cam2world[:3, :3] = tf_H_cam2world_rotation
            tf_H_cam2world[:3, 3] = tf_H_cam2world_translation
            
            tf_H_world2cam = np.linalg.inv(tf_H_cam2world)
            tf_position = tf_H_world2cam[:3, 3]  # 카메라의 실제 월드 좌표
            tf_rotation = tf_H_world2cam[:3, :3]
            
            # 2. Extrinsic 데이터에서 카메라 위치 계산
            if target_camera not in self.extrinsic_data:
                return None
                
            # 🔧 올바른 카메라 월드 좌표 계산: H_cam2world 사용
            extrinsic_H_world2cam = self.extrinsic_data[target_camera]['H_world2cam']
            extrinsic_position = extrinsic_H_world2cam[:3, 3]  # 카메라의 월드 좌표
            extrinsic_rotation = extrinsic_H_world2cam[:3, :3]
            
            # 3. 차이 계산 (둘 다 카메라의 월드 좌표계 상 위치/자세)
            position_diff = np.linalg.norm(tf_position - extrinsic_position)
            
            # 회전 차이 계산 (각도 차이) - 두 회전 행렬 간의 차이
            rotation_diff_matrix = tf_rotation.T @ extrinsic_rotation
            rotation_diff_angle = np.arccos(np.clip((np.trace(rotation_diff_matrix) - 1) / 2, -1, 1))
            rotation_diff_degrees = np.degrees(rotation_diff_angle)
            
            return {
                'tf_position': tf_position,
                'extrinsic_position': extrinsic_position,
                'position_diff': position_diff,
                'rotation_diff_degrees': rotation_diff_degrees,
                'tf_rotation': tf_rotation,
                'extrinsic_rotation': extrinsic_rotation
            }
            
        except Exception as e:
            self.get_logger().debug(f"Could not compare {target_camera}: {e}")
            return None

    def log_validation_results(self, results):
        """검증 결과 로깅"""
        
        self.get_logger().info("🔍 ========== Extrinsic vs TF Validation ==========")
        self.get_logger().info(f"📍 Reference camera: {self.world_reference_camera} (marker-based)")
        
        for camera_name, result in results.items():
            pos_diff = result['position_diff']
            rot_diff = result['rotation_diff_degrees']
            
            # 임계값 설정 (조정 가능)
            pos_threshold = 0.05  # 5cm
            rot_threshold = 5.0   # 5도
            
            pos_status = "✅ OK" if pos_diff < pos_threshold else "⚠️ LARGE"
            rot_status = "✅ OK" if rot_diff < rot_threshold else "⚠️ LARGE"
            
            self.get_logger().info(f"📷 {camera_name}:")
            self.get_logger().info(f"   Position diff: {pos_diff:.4f}m {pos_status}")
            self.get_logger().info(f"   Rotation diff: {rot_diff:.2f}° {rot_status}")
            
            # 자세한 위치 정보 (둘 다 카메라의 월드 좌표계 상 위치)
            tf_pos = result['tf_position']
            ext_pos = result['extrinsic_position']
            self.get_logger().info(f"   Marker-based cam pos: [{tf_pos[0]:.3f}, {tf_pos[1]:.3f}, {tf_pos[2]:.3f}]")
            self.get_logger().info(f"   Extrinsic cam pos:    [{ext_pos[0]:.3f}, {ext_pos[1]:.3f}, {ext_pos[2]:.3f}]")
        
        # 전체 요약
        max_pos_diff = max([r['position_diff'] for r in results.values()])
        max_rot_diff = max([r['rotation_diff_degrees'] for r in results.values()])
        
        overall_status = "✅ GOOD" if max_pos_diff < 0.05 and max_rot_diff < 5.0 else "⚠️ CHECK NEEDED"
        self.get_logger().info(f"📊 Overall: Max pos diff = {max_pos_diff:.4f}m, Max rot diff = {max_rot_diff:.2f}° {overall_status}")
        self.get_logger().info("🔍 ================================================")

    def get_camera_relative_position(self, source_camera, target_camera):
        """두 카메라 간의 상대적 위치 계산 (디버깅용)"""
        
        try:
            # TF에서 상대 변환 가져오기
            transform = self.tf_buffer.lookup_transform(
                f'{source_camera}_frame', f'{target_camera}_frame',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            relative_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y, 
                transform.transform.translation.z
            ])
            
            relative_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            
            return relative_pos, relative_quat
            
        except Exception as e:
            self.get_logger().debug(f"Could not get relative position: {e}")
            return None, None

    def publish_robot_center(self, center, H_world2robot):
        """로봇 중심 위치 발행"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])

        r_world2robot = H_world2robot[0:3, 0:3]
        rot = R.from_matrix(r_world2robot)
        quat = rot.as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]   
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        
        self.robot_center_pub.publish(msg)

    def project_world_to_image(self, world_point, camera_name):
        """월드 좌표를 카메라 이미지로 투영"""
        
        camera_config = self.cameras[camera_name]
        H_world2cam = camera_config.get('H_world2cam')
        
        if H_world2cam is None or not camera_config['info_received']:
            return None
        
        try:
            # 월드 → 카메라 변환
            world_point_homogeneous = np.append(world_point, 1)
            cam_point_homogeneous = H_world2cam @ world_point_homogeneous
            cam_point = cam_point_homogeneous[:3]
            
            if cam_point[2] <= 0:  # 카메라 뒤에 있음
                return None
            
            # 카메라 → 이미지 투영
            image_points, _ = cv2.projectPoints(
                cam_point.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                camera_config['camera_matrix'], camera_config['dist_coeffs']
            )
            
            return image_points[0][0]
            
        except Exception as e:
            return None

    def visualize_results(self, image, detected_markers, camera_name):
        """검출 결과 시각화"""
        
        # 검출된 마커들 표시
        for marker_id, data in detected_markers.items():
            corners = data['corners'][0]
            center = np.mean(corners, axis=0).astype(int)
            
            if marker_id == 10:
                # 10번 마커 - 월드 원점
                color = (255, 0, 0)
                label = f"World{marker_id}"
                cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(image, label, tuple(center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                self.draw_coordinate_axes(image, data['cam_rvec'], data['cam_tvec'], 0.05, camera_name)
            elif marker_id in self.robot_markers_local:
                # 로봇 마커들
                color = (0, 0, 255)
                label = f"R{marker_id}"
                cv2.polylines(image, [corners.astype(int)], True, (0, 255, 255), 2)
                cv2.putText(image, label, tuple(center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 로봇 중심 표시
        if (self.world_established and 
            self.latest_robot_center is not None and
            self.center_timestamp is not None and 
            time.time() - self.center_timestamp < 1.0):
            
            center_image = self.project_world_to_image(self.latest_robot_center, camera_name)
            if center_image is not None:
                center_pt = tuple(center_image.astype(int))
                cv2.circle(image, center_pt, 8, (0, 255, 255), -1)
                cv2.putText(image, "Robot", (center_pt[0]+15, center_pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 상태 정보 표시
        if self.world_from_marker:
            world_status = "World: Marker-based"
            world_color = (0, 255, 0)
        elif self.world_from_extrinsic:
            world_status = "World: Extrinsic-based"
            world_color = (0, 200, 255)
        else:
            world_status = "World: Not available"
            world_color = (0, 0, 255)
            
        cv2.putText(image, world_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, world_color, 2)
        
        # 로봇 좌표 표시
        if self.latest_robot_center is not None:
            coord_text = f"Robot: ({self.latest_robot_center[0]:.2f}, {self.latest_robot_center[1]:.2f}, {self.latest_robot_center[2]:.2f})"
            cv2.putText(image, coord_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 검증 상태 표시 (월드 기준 카메라가 아닌 경우)
        if (self.world_from_marker and 
            self.world_reference_camera is not None and 
            camera_name != self.world_reference_camera and
            hasattr(self, 'validation_results') and 
            camera_name in self.validation_results):
            
            result = self.validation_results[camera_name]
            pos_diff = result['position_diff']
            rot_diff = result['rotation_diff_degrees']
            
            validation_color = (0, 255, 0) if pos_diff < 0.05 and rot_diff < 5.0 else (0, 165, 255)
            validation_text = f"Validation: Pos {pos_diff:.3f}m, Rot {rot_diff:.1f}°"
            cv2.putText(image, validation_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, validation_color, 1)

        # 이미지 표시
        cv2.imshow(f'{camera_name} - ArUco Detection', image)
        cv2.waitKey(1)

    def draw_coordinate_axes(self, image, rvec, tvec, length, camera_name):
        """좌표축 그리기"""
        camera_config = self.cameras[camera_name]
        
        if not camera_config['info_received']:
            return
            
        try:
            axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                        camera_config['camera_matrix'], 
                                        camera_config['dist_coeffs'])
            
            imgpts = np.int32(imgpts).reshape(-1,2)
            origin = tuple(imgpts[0])
            
            # X축 (빨간색), Y축 (초록색), Z축 (파란색)
            cv2.arrowedLine(image, origin, tuple(imgpts[1]), (0,0,255), 2)
            cv2.arrowedLine(image, origin, tuple(imgpts[2]), (0,255,0), 2)
            cv2.arrowedLine(image, origin, tuple(imgpts[3]), (255,0,0), 2)
            
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    detector = D435ArucoDetectorWithExtrinsic()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
