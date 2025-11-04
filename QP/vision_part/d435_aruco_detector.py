#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from threading import Lock
import time
import yaml
import os
from scipy.spatial.transform import Rotation as R

class D435ArUcoDetector(Node):
    def __init__(self):
        super().__init__('d435_aruco_detector')
        
        # ROS2 관련 초기화
        self.bridge = CvBridge()
        self.rectangle_center_pub = self.create_publisher(PoseStamped, '/aruco_rectangle_center', 10)
        
        # ArUco 설정 - 버전 호환성
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.marker_length = 0.05  # 마커 크기 5cm
        
        # 카메라 데이터 저장
        self.detection_lock = Lock()
        self.latest_detections = {}
        self.camera_info = {}
        
        # ArUco 마커 사각형 배치 (로봇에 부착된 마커들의 로컬 좌표)
        # 0--1
        # |  |
        # 3--2
        self.marker_positions_local = {
            0: np.array([0.3, 0.135, 0.]),  # 왼쪽 위
            1: np.array([0.3, -0.135, 0.]),   # 오른쪽 위
            2: np.array([-0.3, -0.135, 0.]),    # 오른쪽 아래
            3: np.array([-0.3, 0.135, 0.])    # 왼쪽 아래
        }
        
        # 캘리브레이션 결과 로드
        self.load_camera_calibrations()
        
        # 카메라 토픽 구독
        self.setup_camera_subscribers()
        
        # 주기적으로 사각형 중심 계산
        self.timer = self.create_timer(0.1, self.calculate_rectangle_center)
        # 계산된 중심점 저장
        self.latest_rectangle_center = None
        self.center_timestamp = 0
        self.get_logger().info("D435 ArUco Detector initialized")
    
    def load_camera_calibrations(self):
        """저장된 카메라 캘리브레이션 결과 로드"""
        self.camera_extrinsics = {}
        
        # 통합 캘리브레이션 파일 시도
        if os.path.exists('multi_camera_calibration.yaml'):
            try:
                with open('multi_camera_calibration.yaml', 'r') as f:
                    calib_data = yaml.safe_load(f)
                
                for camera_name, camera_data in calib_data['cameras'].items():
                    self.camera_extrinsics[camera_name] = {
                        'position': np.array(camera_data['position']),
                        'rotation': np.array(camera_data['rotation_matrix'])
                    }
                    self.get_logger().info(f"Loaded calibration for {camera_name}")
                
                self.get_logger().info("Successfully loaded multi-camera calibration")
                return
                
            except Exception as e:
                self.get_logger().warn(f"Failed to load multi-camera calibration: {e}")
        
        # 개별 캘리브레이션 파일들 시도
        camera_names = ['camera1', 'camera2', 'camera3']
        loaded_cameras = 0
        
        for camera_name in camera_names:
            filename = f'{camera_name}_extrinsic.yaml'
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        calib_data = yaml.safe_load(f)
                    
                    self.camera_extrinsics[camera_name] = {
                        'position': np.array(calib_data['position']),
                        'rotation': np.array(calib_data['rotation_matrix'])
                    }
                    loaded_cameras += 1
                    self.get_logger().info(f"Loaded calibration for {camera_name}")
                    
                except Exception as e:
                    self.get_logger().warn(f"Failed to load {filename}: {e}")
        
        if loaded_cameras == 0:
            self.get_logger().warn("No calibration files found! Using default values")
            self.setup_default_extrinsics()
        else:
            self.get_logger().info(f"Loaded calibration for {loaded_cameras} cameras")
    
    def setup_default_extrinsics(self):
        """기본 외부 매개변수 설정 (캘리브레이션 파일이 없을 때)"""
        self.get_logger().warn("Using default camera extrinsics - please run calibration!")
        self.camera_extrinsics = {
            'camera1': {
                'position': np.array([0.0, -1.5, 1.2]),
                'rotation': np.eye(3)
            },
            'camera2': {
                'position': np.array([1.3, -0.75, 1.2]),
                'rotation': self.rotation_from_euler(0, 0, -np.pi/6)
            },
            'camera3': {
                'position': np.array([-1.3, -0.75, 1.2]),
                'rotation': self.rotation_from_euler(0, 0, np.pi/6)
            }
        }
    
    def rotation_from_euler(self, roll, pitch, yaw):
        """오일러 각도에서 회전 행렬 생성"""
        return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    def setup_camera_subscribers(self):
        """D435 카메라 토픽 구독"""
        # D435 토픽 구조 확인 및 구독
        available_cameras = list(self.camera_extrinsics.keys())
        
        for camera_name in available_cameras:
            # 이미지 토픽 구독 (여러 패턴 시도)
            possible_topics = [
                f'/{camera_name}/color/image_raw',
                f'/{camera_name}/{camera_name}/color/image_raw',
                f'/camera/{camera_name}/color/image_raw'
            ]
            
            # 카메라 정보 토픽
            possible_info_topics = [
                f'/{camera_name}/color/camera_info',
                f'/{camera_name}/{camera_name}/color/camera_info',
                f'/camera/{camera_name}/color/camera_info'
            ]
            
            # 첫 번째 패턴으로 구독 (실제 토픽명에 맞게 조정)
            image_topic = possible_topics[1]
            info_topic = possible_info_topics[1]
            
            self.create_subscription(
                Image,
                image_topic,
                lambda msg, name=camera_name: self.image_callback(msg, name),
                10
            )
            
            self.create_subscription(
                CameraInfo,
                info_topic,
                lambda msg, name=camera_name: self.camera_info_callback(msg, name),
                10
            )
            
            self.get_logger().info(f"Subscribed to {image_topic} and {info_topic}")
    
    def camera_info_callback(self, msg, camera_name):
        """카메라 정보 콜백"""
        self.camera_info[camera_name] = {
            'camera_matrix': np.array(msg.k).reshape(3, 3),
            'dist_coeffs': np.array(msg.d)
        }
    
    def image_callback(self, msg, camera_name):
        """이미지 콜백 및 ArUco 검출"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = self.detect_aruco_markers(cv_image, camera_name)
            # 중심점을 이미지에 투영해서 그리기
            vis_image = self.draw_detections_with_center(cv_image.copy(), detections, camera_name)
            
            with self.detection_lock:
                self.latest_detections[camera_name] = {
                    'detections': detections,
                    'timestamp': time.time(),
                    'image': cv_image,
                    'vis_image': vis_image
                }
            
            # 실시간 시각화
            cv2.imshow(f'{camera_name}_aruco_detection', vis_image)
            cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error in {camera_name}: {str(e)}")
    
    def detect_aruco_markers(self, image, camera_name):
        """ArUco 마커 검출"""
        if camera_name not in self.camera_info:
            return []
        
        cam_info = self.camera_info[camera_name]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ArUco 마커 검출 - 버전 호환성
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        detections = []
        if ids is not None:
            # 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 
                self.marker_length,
                cam_info['camera_matrix'],
                cam_info['dist_coeffs']
            )
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in [0, 1, 2, 3]:  # 로봇에 부착된 마커들
                    # 카메라 좌표계에서의 위치
                    cam_position = tvecs[i].flatten()
                    cam_rotation = rvecs[i].flatten()
                    
                    # 월드 좌표계로 변환
                    world_position = self.camera_to_world(cam_position, camera_name)
                    
                    detection = {
                        'id': int(marker_id),
                        'camera_position': cam_position,
                        'world_position': world_position,
                        'rotation': cam_rotation,
                        'corners': corners[i],
                        'camera_name': camera_name
                    }
                    detections.append(detection)
                    
                    # self.get_logger().info(
                    #     f"{camera_name}: Marker {marker_id} at world pos "
                    #     f"({world_position[0]:.3f}, {world_position[1]:.3f}, {world_position[2]:.3f})"
                    # )
        
        return detections
    
    def camera_to_world(self, cam_position, camera_name):
        """카메라 좌표를 월드 좌표로 변환"""
        if camera_name not in self.camera_extrinsics:
            self.get_logger().warn(f"No extrinsics for {camera_name}, using camera coordinates")
            return cam_position
        
        extrinsic = self.camera_extrinsics[camera_name]
        
        # 정확한 변환: 카메라 좌표 -> 월드 좌표
        # world_point = R_wc @ cam_point + t_wc
        R_wc = extrinsic['rotation']  # 카메라 -> 월드 회전행렬
        t_wc = extrinsic['position']  # 카메라 위치 (월드 좌표)
        
        world_position = R_wc @ cam_position.reshape(3, 1) + t_wc.reshape(3, 1)
        
        return world_position.flatten()
    
    def draw_detections_with_center(self, image, detections, camera_name):
        """검출된 마커와 계산된 중심점을 이미지에 그리기"""
        
        # 1. 기본 마커 검출 결과 그리기
        for detection in detections:
            marker_id = detection['id']
            corners = detection['corners']
            
            # 마커 그리기
            cv2.aruco.drawDetectedMarkers(image, [corners], np.array([marker_id]))
            
            # 마커 ID
            corner = corners[0][0].astype(int)
            cv2.putText(image, f"ID:{marker_id}", 
                       (corner[0], corner[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 월드 좌표
            world_pos = detection['world_position']
            pos_text = f"({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
            cv2.putText(image, pos_text,
                       (corner[0], corner[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 2. 계산된 rectangle center 투영해서 그리기
        if self.latest_rectangle_center is not None:
            current_time = time.time()
            
            # 최근 0.5초 이내의 중심점만 표시
            if current_time - self.center_timestamp < 0.5:
                center_image_point = self.project_world_to_image(
                    self.latest_rectangle_center, camera_name
                )
                
                if center_image_point is not None:
                    self.draw_rectangle_center_on_image(image, center_image_point, 
                                                      self.latest_rectangle_center)
        
        # 3. 마커들을 선으로 연결 (사각형 표시)
        if len(detections) >= 2:
            self.draw_rectangle_outline(image, detections)
        
        return image
    
    def project_world_to_image(self, world_point, camera_name):
        """월드 좌표를 이미지 좌표로 투영"""
        
        if camera_name not in self.camera_extrinsics or camera_name not in self.camera_info:
            return None
        
        try:
            # 1. 월드 좌표 -> 카메라 좌표 변환
            extrinsic = self.camera_extrinsics[camera_name]
            R_wc = extrinsic['rotation']  # 카메라 -> 월드
            t_wc = extrinsic['position']  # 카메라 위치
            
            # 역변환: 월드 -> 카메라
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            
            # 월드 점을 카메라 좌표로 변환
            world_point = np.array(world_point).reshape(3, 1)
            cam_point = R_cw @ world_point + t_cw.reshape(3, 1)
            cam_point = cam_point.flatten()
            
            # 카메라 뒤쪽에 있으면 투영하지 않음
            if cam_point[2] <= 0:
                return None
            
            # 2. 카메라 좌표 -> 이미지 좌표 투영
            cam_info = self.camera_info[camera_name]
            
            # projectPoints 사용 (왜곡 보정 포함)
            image_points, _ = cv2.projectPoints(
                cam_point.reshape(1, 1, 3),
                np.zeros(3),  # 회전 없음 (이미 카메라 좌표계)
                np.zeros(3),  # 평행이동 없음
                cam_info['camera_matrix'],
                cam_info['dist_coeffs']
            )
            
            image_point = image_points[0][0]
            
            # 이미지 범위 내에 있는지 확인
            height, width = 480, 640  # D435 기본 해상도 (실제로는 동적으로 가져와야 함)
            if 0 <= image_point[0] < width and 0 <= image_point[1] < height:
                return image_point
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"Projection error for {camera_name}: {e}")
            return None
    
    def draw_rectangle_center_on_image(self, image, center_point, world_center):
        """이미지에 rectangle center 그리기"""
        
        center_x, center_y = int(center_point[0]), int(center_point[1])
        
        # 1. 중심점 표시 (큰 빨간 원)
        cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  # 빨간 원
        cv2.circle(image, (center_x, center_y), 15, (255, 255, 255), 2)  # 흰 테두리
        
        # 2. 십자가 표시
        cross_size = 20
        cv2.line(image, 
                (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), 
                (0, 0, 255), 3)
        cv2.line(image, 
                (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), 
                (0, 0, 255), 3)
        
        # 3. 라벨 표시
        label = "Robot Center"
        cv2.putText(image, label, 
                   (center_x + 20, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 4. 월드 좌표 표시
        coord_text = f"({world_center[0]:.2f}, {world_center[1]:.2f}, {world_center[2]:.2f})"
        cv2.putText(image, coord_text,
                   (center_x + 20, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def draw_rectangle_outline(self, image, detections):
        """검출된 마커들을 선으로 연결해서 사각형 표시"""
        
        if len(detections) < 2:
            return
        
        # 마커 ID별로 중심점 계산
        marker_centers = {}
        for detection in detections:
            marker_id = detection['id']
            corners = detection['corners'][0]
            center = np.mean(corners, axis=0).astype(int)
            marker_centers[marker_id] = center
        
        # 사각형 연결 순서 (시계방향): 0->1->2->3->0
        connection_order = [0, 1, 2, 3, 0]
        
        for i in range(len(connection_order) - 1):
            id1, id2 = connection_order[i], connection_order[i + 1]
            
            if id1 in marker_centers and id2 in marker_centers:
                pt1 = tuple(marker_centers[id1])
                pt2 = tuple(marker_centers[id2])
                
                # 사각형 테두리 그리기
                cv2.line(image, pt1, pt2, (255, 0, 255), 2)  # 마젠타색 선
    
    def calculate_rectangle_center(self):
        """사각형의 중심점 계산 - 중복 마커 문제 해결"""
        with self.detection_lock:
            current_detections = self.latest_detections.copy()
        
        # 각 마커별로 가장 좋은 검출 결과 선택
        marker_positions = {}
        current_time = time.time()
        
        # 카메라 우선순위 (필요에 따라 조정)
        camera_priority = ['camera1', 'camera2', 'camera3']
        
        # 우선순위 순서대로 마커 수집
        for camera_name in camera_priority:
            if camera_name not in current_detections:
                continue
                
            data = current_detections[camera_name]
            if current_time - data['timestamp'] > 0.5:
                continue
            
            for detection in data['detections']:
                marker_id = detection['id']
                
                # 아직 해당 마커가 없으면 추가
                if marker_id not in marker_positions:
                    marker_positions[marker_id] = detection['world_position']
        
        # 중심 계산
        if len(marker_positions) >= 1:
            center = self.compute_rectangle_center(marker_positions)
            if center is not None:
                self.latest_rectangle_center = center
                self.center_timestamp = current_time
                self.publish_rectangle_center(center, len(marker_positions))
        
        # 디버깅 정보
        if marker_positions:
            self.get_logger().debug(f"Used markers: {list(marker_positions.keys())}")
    
    def compute_rectangle_center(self, marker_positions):
        """로봇 중심 계산 - 로컬 원점 [0,0,0]을 월드로 변환"""
        
        if len(marker_positions) < 1:
            return None
        
        # 로봇의 실제 중심점 (로컬 좌표계 원점)
        robot_center_local = np.array([0.0, 0.0, 0.0])
        
        # 로컬 원점을 월드 좌표로 변환
        robot_center_world = self.transform_local_center_to_world(
            robot_center_local, marker_positions
        )
        
        return robot_center_world

    def transform_local_center_to_world(self, local_center, marker_positions):
        """로컬 중심점 [0,0,0]을 월드 좌표로 변환"""
        
        detected_ids = list(marker_positions.keys())
        
        if len(detected_ids) == 1:
            # 1개 마커: 단순 변환
            return self.transform_with_single_marker(local_center, detected_ids[0], marker_positions)
        
        elif len(detected_ids) >= 2:
            # 2개 이상: Kabsch 알고리즘으로 정확한 변환
            return self.transform_with_multiple_markers(local_center, marker_positions)
        
        return None

    def transform_with_single_marker(self, local_center, marker_id, marker_positions):
        """단일 마커로 로봇 중심 변환 - 회전 고려"""
        
        world_marker_pos = marker_positions[marker_id]
        local_marker_pos = self.marker_positions_local[marker_id]
        
        # 방법 1: 마커의 자세 정보 사용 (가장 정확)
        if self.has_marker_rotation_info(marker_id):
            robot_center_world = self.transform_with_marker_pose(
                local_center, marker_id, world_marker_pos
            )
        else:
            # 방법 2: 단순 변환 (회전 추정 불가능한 경우)
            self.get_logger().warn(f"No rotation info for marker {marker_id}, using simple translation")
            robot_center_world = world_marker_pos - local_marker_pos
        
        self.get_logger().info(f"Robot center (single marker {marker_id}): ({robot_center_world[0]:.3f}, {robot_center_world[1]:.3f}, {robot_center_world[2]:.3f})")
        return robot_center_world

    def transform_with_marker_pose(self, local_center, marker_id, world_marker_pos):
        """마커의 포즈 정보를 사용한 정확한 변환"""
        
        # 마커의 회전 정보 가져오기 (최근 검출에서)
        marker_rotation = self.get_marker_rotation(marker_id)
        
        if marker_rotation is not None:
            # 마커의 회전행렬 계산
            R_marker = cv2.Rodrigues(marker_rotation)[0]  # rvec -> rotation matrix
            
            # 로컬 중심을 마커 기준으로 변환
            local_marker_pos = self.marker_positions_local[marker_id]
            local_offset = local_center - local_marker_pos  # [0,0,0] - marker_local_pos
            
            # 로컬 오프셋을 마커의 회전에 맞게 월드로 변환
            world_offset = R_marker @ local_offset
            
            # 최종 로봇 중심 위치
            robot_center_world = world_marker_pos + world_offset
            
            return robot_center_world
        else:
            # 회전 정보가 없으면 단순 변환
            local_marker_pos = self.marker_positions_local[marker_id]
            return world_marker_pos - local_marker_pos

    def get_marker_rotation(self, marker_id):
        """최근 검출에서 마커의 회전 정보 가져오기"""
        
        with self.detection_lock:
            current_detections = self.latest_detections.copy()
        
        current_time = time.time()
        
        # 모든 카메라에서 해당 마커 찾기
        for camera_name, data in current_detections.items():
            if current_time - data['timestamp'] > 0.5:
                continue
                
            for detection in data['detections']:
                if detection['id'] == marker_id:
                    return detection['rotation']  # rvec
        
        return None

    def has_marker_rotation_info(self, marker_id):
        """마커의 회전 정보가 있는지 확인"""
        return self.get_marker_rotation(marker_id) is not None

    def transform_with_multiple_markers(self, local_center, marker_positions):
        """다중 마커로 정확한 변환 (회전 포함)"""
        
        try:
            detected_ids = list(marker_positions.keys())
            
            # 로컬 및 월드 좌표 수집
            local_positions = np.array([self.marker_positions_local[mid] for mid in detected_ids])
            world_positions = np.array([marker_positions[mid] for mid in detected_ids])
            
            # Kabsch 알고리즘으로 최적 변환 계산
            R, t = self.compute_optimal_transformation(local_positions, world_positions)
            
            # 로컬 중심을 월드로 변환
            robot_center_world = R @ local_center + t
            
            self.get_logger().info(f"Robot center (Kabsch, {len(detected_ids)} markers): ({robot_center_world[0]:.3f}, {robot_center_world[1]:.3f}, {robot_center_world[2]:.3f})")
            
            # 변환 품질 검증
            self.validate_transformation(R, t, local_positions, world_positions)
            
            return robot_center_world
            
        except Exception as e:
            self.get_logger().error(f"Multi-marker transformation failed: {e}")
            # Fallback: 첫 번째 마커로 단순 변환
            first_id = list(marker_positions.keys())[0]
            return self.transform_with_single_marker(local_center, first_id, marker_positions)

    def compute_optimal_transformation(self, local_positions, world_positions):
        """Kabsch 알고리즘으로 최적 변환 계산"""
        
        # 1. 중심점들 계산
        local_centroid = np.mean(local_positions, axis=0)
        world_centroid = np.mean(world_positions, axis=0)
        
        # 2. 중심점 기준으로 정렬
        local_centered = local_positions - local_centroid
        world_centered = world_positions - world_centroid
        
        # 3. 공분산 행렬
        H = local_centered.T @ world_centered
        
        # 4. SVD 분해
        U, S, Vt = np.linalg.svd(H)
        
        # 5. 회전 행렬
        R = Vt.T @ U.T
        
        # 6. 반사 보정
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 7. 평행이동
        t = world_centroid - R @ local_centroid
        
        return R, t

    def validate_transformation(self, R, t, local_positions, world_positions):
        """변환 품질 검증"""
        
        errors = []
        for i, (local_pos, world_pos) in enumerate(zip(local_positions, world_positions)):
            transformed_pos = R @ local_pos + t
            error = np.linalg.norm(transformed_pos - world_pos)
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        self.get_logger().debug(f"Transformation quality - Mean error: {mean_error:.4f}m, Max error: {max_error:.4f}m")
        
        if max_error > 0.1:  # 10cm 이상 오차
            self.get_logger().warn(f"High transformation error: {max_error:.3f}m")
        
    def publish_rectangle_center(self, center, num_markers):
        """사각형 중심점 퍼블리시"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        
        msg.pose.orientation.w = 1.0
        
        self.rectangle_center_pub.publish(msg)
        
        self.get_logger().info(
            f'Rectangle center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) '
            f'from {num_markers} markers'
        )

def main(args=None):
    rclpy.init(args=args)
    
    detector = D435ArUcoDetector()
    
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