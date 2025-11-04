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
            
            # 디버깅 로그 추가
            time_diff = current_time - self.center_timestamp
            self.get_logger().debug(f"{camera_name}: Center age: {time_diff:.3f}s")
            
            if time_diff < 0.5:
                center_image_point = self.project_world_to_image(
                    self.latest_rectangle_center, camera_name
                )
                
                if center_image_point is not None:
                    self.get_logger().debug(f"{camera_name}: Center projected to ({center_image_point[0]:.1f}, {center_image_point[1]:.1f})")
                    self.draw_rectangle_center_on_image(image, center_image_point, 
                                                    self.latest_rectangle_center)
                else:
                    self.get_logger().debug(f"{camera_name}: Center projection failed")
            else:
                self.get_logger().debug(f"{camera_name}: Center too old ({time_diff:.3f}s)")
        else:
            self.get_logger().debug(f"{camera_name}: No center available")
        
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
    
    def validate_projection(self):
        """투영이 올바르게 되는지 검증"""
        if self.latest_rectangle_center is None:
            return
        
        center = self.latest_rectangle_center
        self.get_logger().info(f"Robot center (world): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        
        for camera_name in self.camera_extrinsics.keys():
            if camera_name in self.camera_info:
                projected = self.project_world_to_image(center, camera_name)
                if projected is not None:
                    self.get_logger().info(f"{camera_name}: Center projects to ({projected[0]:.1f}, {projected[1]:.1f})")
                else:
                    self.get_logger().warn(f"{camera_name}: Projection failed or out of view")

    def calculate_rectangle_center(self):
        """통합된 로봇 포즈 추정 및 중심점 계산"""
        with self.detection_lock:
            current_detections = self.latest_detections.copy()
        
        # 1. 모든 관측 데이터 수집
        all_observations = self.collect_all_observations(current_detections)
        
        if len(all_observations) < 1:
            return
        
        # 2. 로봇 포즈 추정 (위치 + 회전)
        robot_pose = self.estimate_robot_pose(all_observations)
        
        if robot_pose is not None:
            # 3. 로봇 중심점 계산 (포즈를 이용해 로컬 원점 변환)
            robot_position, robot_rotation = robot_pose
            center = robot_position  # 로컬 원점 [0,0,0]의 월드 위치
            
            self.latest_rectangle_center = center
            self.center_timestamp = time.time()
            self.publish_rectangle_center(center, len(all_observations))
            # 투영 검증 추가
            self.validate_projection()
            
            self.get_logger().info(f"Robot center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

    def collect_all_observations(self, current_detections):
        """모든 카메라에서 모든 마커 관측 수집"""
        observations = []
        current_time = time.time()
        
        for camera_name, data in current_detections.items():
            if current_time - data['timestamp'] > 0.5:
                continue
            
            for detection in data['detections']:
                marker_id = detection['id']
                
                if marker_id in self.marker_positions_local:
                    # 각 관측은 하나의 제약 방정식
                    obs = {
                        'marker_id': marker_id,
                        'camera_name': camera_name,
                        'image_point': np.mean(detection['corners'][0], axis=0),  # 마커 중심
                        'local_position': self.marker_positions_local[marker_id]
                    }
                    observations.append(obs)
        
        self.get_logger().debug(f"Collected {len(observations)} observations")
        return observations

    def estimate_robot_pose(self, observations):
        """PnP 또는 최적화를 통한 로봇 포즈 추정"""
        
        if len(observations) < 3:
            # 관측이 적으면 단순한 방법 사용
            return self.estimate_pose_simple(observations)
        else:
            # 충분한 관측이 있으면 최적화 방법 사용
            return self.estimate_pose_optimization(observations)

    def estimate_pose_simple(self, observations):
        """간단한 포즈 추정 (1-2개 관측)"""
        
        if len(observations) == 0:
            return None
        
        # 첫 번째 관측으로 초기 추정
        obs = observations[0]
        marker_id = obs['marker_id']
        camera_name = obs['camera_name']
        
        # 해당 카메라에서의 월드 위치 (기존 방법 사용)
        world_marker_pos = None
        with self.detection_lock:
            for detection in self.latest_detections[camera_name]['detections']:
                if detection['id'] == marker_id:
                    world_marker_pos = detection['world_position']
                    break
        
        if world_marker_pos is None:
            return None
        
        local_marker_pos = obs['local_position']
        
        # 단순 변환 (회전 무시)
        robot_position = world_marker_pos - local_marker_pos
        robot_rotation = np.eye(3)  # 단위 행렬
        
        return robot_position, robot_rotation

    def estimate_pose_optimization(self, observations):
        """최적화를 통한 정확한 포즈 추정"""
        
        try:
            from scipy.optimize import minimize
            
            # 초기 추정 (단순 방법으로)
            initial_pose = self.estimate_pose_simple(observations[:1])
            if initial_pose is None:
                return None
            
            initial_position, initial_rotation = initial_pose
            
            # 회전을 축-각 표현으로 변환
            from scipy.spatial.transform import Rotation as R
            initial_rvec = R.from_matrix(initial_rotation).as_rotvec()
            
            # 최적화 변수: [x, y, z, rx, ry, rz] (6DOF)
            x0 = np.concatenate([initial_position, initial_rvec])
            
            # 목적 함수: 재투영 오차의 제곱합
            def objective(params):
                position = params[:3]
                rvec = params[3:6]
                
                rotation_matrix = R.from_rotvec(rvec).as_matrix()
                
                total_error = 0
                
                for obs in observations:
                    # 로컬 마커 위치를 월드로 변환
                    local_pos = obs['local_position']
                    world_pos = rotation_matrix @ local_pos + position
                    
                    # 월드 위치를 이미지로 투영
                    projected_point = self.project_world_to_image_for_camera(
                        world_pos, obs['camera_name']
                    )
                    
                    if projected_point is not None:
                        # 재투영 오차
                        observed_point = obs['image_point']
                        error = np.linalg.norm(projected_point - observed_point)
                        total_error += error ** 2
                    else:
                        total_error += 1000  # 투영 실패 시 큰 페널티
                
                return total_error
            
            # 최적화 실행
            result = minimize(objective, x0, method='BFGS')
            
            if result.success:
                optimized_position = result.x[:3]
                optimized_rvec = result.x[3:6]
                optimized_rotation = R.from_rotvec(optimized_rvec).as_matrix()
                
                # 결과 검증
                final_error = result.fun
                if final_error < 100:  # 합리적인 오차
                    self.get_logger().info(f"Pose optimization succeeded, error: {final_error:.2f}")
                    return optimized_position, optimized_rotation
                else:
                    self.get_logger().warn(f"High optimization error: {final_error:.2f}")
            
        except Exception as e:
            self.get_logger().error(f"Pose optimization failed: {e}")
        
        # Fallback
        return self.estimate_pose_simple(observations)

    def project_world_to_image_for_camera(self, world_point, camera_name):
        """특정 카메라에 대한 월드-이미지 투영"""
        
        if camera_name not in self.camera_extrinsics or camera_name not in self.camera_info:
            return None
        
        try:
            # 월드 -> 카메라 변환
            extrinsic = self.camera_extrinsics[camera_name]
            R_wc = extrinsic['rotation']
            t_wc = extrinsic['position']
            
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            
            world_point = np.array(world_point).reshape(3, 1)
            cam_point = R_cw @ world_point + t_cw.reshape(3, 1)
            cam_point = cam_point.flatten()
            
            if cam_point[2] <= 0:
                return None
            
            # 카메라 -> 이미지 투영
            cam_info = self.camera_info[camera_name]
            image_points, _ = cv2.projectPoints(
                cam_point.reshape(1, 1, 3),
                np.zeros(3), np.zeros(3),
                cam_info['camera_matrix'],
                cam_info['dist_coeffs']
            )
            
            return image_points[0][0]
            
        except Exception:
            return None

    def estimate_pose_pnp(self, observations):
        """OpenCV PnP를 사용한 포즈 추정 (가장 정확)"""
        
        if len(observations) < 4:
            return self.estimate_pose_optimization(observations)
        
        # 각 카메라별로 PnP 실행 후 결과 융합
        pose_estimates = []
        
        # 카메라별로 관측 그룹화
        camera_observations = {}
        for obs in observations:
            camera_name = obs['camera_name']
            if camera_name not in camera_observations:
                camera_observations[camera_name] = []
            camera_observations[camera_name].append(obs)
        
        for camera_name, cam_obs in camera_observations.items():
            if len(cam_obs) < 3:  # PnP는 최소 3점 필요
                continue
            
            if camera_name not in self.camera_info:
                continue
            
            try:
                # 3D-2D 대응점 구성
                object_points = np.array([obs['local_position'] for obs in cam_obs], dtype=np.float32)
                image_points = np.array([obs['image_point'] for obs in cam_obs], dtype=np.float32)
                
                # PnP 실행
                cam_info = self.camera_info[camera_name]
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    cam_info['camera_matrix'],
                    cam_info['dist_coeffs']
                )
                
                if success:
                    # 카메라 좌표계에서의 로봇 포즈를 월드 좌표계로 변환
                    R_obj_to_cam = cv2.Rodrigues(rvec)[0]
                    t_obj_to_cam = tvec.flatten()
                    
                    # 월드 좌표계로 변환
                    extrinsic = self.camera_extrinsics[camera_name]
                    R_cam_to_world = extrinsic['rotation']
                    t_cam_to_world = extrinsic['position']
                    
                    # 로봇의 월드 포즈
                    R_obj_to_world = R_cam_to_world @ R_obj_to_cam
                    t_obj_to_world = R_cam_to_world @ t_obj_to_cam + t_cam_to_world
                    
                    pose_estimates.append((t_obj_to_world, R_obj_to_world))
                    
                    self.get_logger().debug(f"PnP successful for {camera_name} with {len(cam_obs)} points")
                    
            except Exception as e:
                self.get_logger().debug(f"PnP failed for {camera_name}: {e}")
                continue
        
        if len(pose_estimates) == 0:
            return self.estimate_pose_optimization(observations)
        elif len(pose_estimates) == 1:
            return pose_estimates[0]
        else:
            # 여러 추정의 평균 (더 정교한 융합 가능)
            positions = [pose[0] for pose in pose_estimates]
            rotations = [pose[1] for pose in pose_estimates]
            
            avg_position = np.mean(positions, axis=0)
            # 회전 행렬의 평균은 복잡하므로 첫 번째 사용
            avg_rotation = rotations[0]
            
            return avg_position, avg_rotation

    # def compute_weighted_average(self, valid_positions, valid_cameras):
    #     """가중치 제거 - 단순 평균만 사용"""
    #     return np.mean(valid_positions, axis=0)
   
        
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