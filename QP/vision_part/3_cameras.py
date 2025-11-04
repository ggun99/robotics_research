#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import threading
from threading import Lock
import time
from scipy.spatial.transform import Rotation as R

class RobotPoseEstimator(Node):
    def __init__(self):
        super().__init__('robot_pose_estimator')
        
        # ROS2 관련 초기화
        self.bridge = CvBridge()
        self.pose_publisher = self.create_publisher(PoseStamped, '/robot_pose_world', 10)
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.05  # 마커 크기 (미터)
        
        # 카메라 데이터 저장
        self.camera_data = {}
        self.detection_lock = Lock()
        self.latest_detections = {}  # 카메라별 최신 검출 결과
        
        # 로봇 마커 설정 (로봇 프레임에서의 4개 마커 위치)
        self.robot_marker_coords = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float64),      # 로봇 중심
            1: np.array([0.1, 0.0, 0.0], dtype=np.float64),     # 앞쪽
            2: np.array([0.1, 0.08, 0.0], dtype=np.float64),    # 앞쪽-오른쪽
            3: np.array([0.0, 0.08, 0.0], dtype=np.float64),    # 오른쪽
        }
        
        # 카메라 파라미터 로드
        self.load_camera_parameters()
        
        # 카메라 토픽 구독
        self.setup_camera_subscribers()
        
        # 주기적으로 로봇 위치 추정
        self.timer = self.create_timer(0.1, self.estimate_robot_pose)  # 10Hz
        
        self.get_logger().info("Robot Pose Estimator initialized")
    
    def load_camera_parameters(self):
        """카메라 파라미터 로드"""
        # 실제로는 파일에서 로드해야 함
        # 예시 데이터
        self.camera_params = {
            'camera1': {
                'camera_matrix': np.array([
                    [604.1236572265625, 0.0, 316.11456298828125],
                    [0.0, 604.129638671875, 234.6119842529297],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                'dist_coeffs': np.zeros(5, dtype=np.float64),
                'R_wc': np.eye(3, dtype=np.float64),  # 카메라 -> 월드 회전
                't_wc': np.array([0.0, 0.0, 1.0], dtype=np.float64).reshape(3, 1)  # 카메라 -> 월드 평행이동
            },
            'camera2': {
                'camera_matrix': np.array([
                    [604.7739868164062, 0.0, 325.9398193359375],
                    [0.0, 604.6397094726562, 233.66464233398438],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                'dist_coeffs': np.zeros(5, dtype=np.float64),
                'R_wc': self.rotation_matrix_from_euler(0, 0, -np.pi/4),
                't_wc': np.array([2.0, 0.0, 1.0], dtype=np.float64).reshape(3, 1)
            },
            'camera3': {
                'camera_matrix': np.array([
                    [606.121337890625, 0.0, 317.46221923828125],
                    [0.0, 606.2796020507812, 234.80279541015625],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                'dist_coeffs': np.zeros(5, dtype=np.float64),
                'R_wc': self.rotation_matrix_from_euler(0, 0, np.pi/2),
                't_wc': np.array([0.0, 2.0, 1.0], dtype=np.float64).reshape(3, 1)
            }
        }
        
        # 실제 파라미터 파일에서 로드하는 경우:
        # try:
        #     with open('camera_params.yaml', 'r') as f:
        #         data = yaml.safe_load(f)
        #         self.camera_params = self.parse_camera_yaml(data)
        # except FileNotFoundError:
        #     self.get_logger().warn("Camera parameter file not found, using default values")
    
    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        """오일러 각도에서 회전 행렬 생성"""
        return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    def setup_camera_subscribers(self):
        """카메라 토픽 구독 설정"""
        # 각 카메라별 토픽 구독
        camera_topics = [
            'camera1/camera1/color/image_raw',
            'camera2/camera2/color/image_raw', 
            'camera3/camera3/color/image_raw'
        ]
        
        for i, topic in enumerate(camera_topics):
            camera_name = f'camera_{i}'
            self.create_subscription(
                Image,
                topic,
                lambda msg, name=camera_name: self.image_callback(msg, name),
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")
    
    def image_callback(self, msg, camera_name):
        """카메라 이미지 콜백"""
        try:
            # ROS Image를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # ArUco 마커 검출
            detections = self.detect_aruco_markers(cv_image, camera_name)
            
            # 스레드 안전하게 결과 저장
            with self.detection_lock:
                self.latest_detections[camera_name] = {
                    'detections': detections,
                    'timestamp': time.time(),
                    'frame': cv_image
                }
                
        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {str(e)}")
    
    def detect_aruco_markers(self, image, camera_name):
        """ArUco 마커 검출 및 3D 위치 추정"""
        if camera_name not in self.camera_params:
            return []
        
        cam_param = self.camera_params[camera_name]
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ArUco 마커 검출
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None:
            return []
        
        # 마커 포즈 추정
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, 
            cam_param['camera_matrix'], 
            cam_param['dist_coeffs']
        )
        
        detections = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.robot_marker_coords:
                # 카메라 좌표계에서의 마커 위치
                cam_point = tvecs[i].reshape(3, 1)
                
                # 카메라 -> 월드 좌표 변환
                world_point = self.cam_to_world_point(
                    cam_param['R_wc'], 
                    cam_param['t_wc'], 
                    cam_point
                )
                
                detection = {
                    'id': int(marker_id),
                    'world_position': world_point.flatten(),
                    'camera_position': cam_point.flatten(),
                    'rvec': rvecs[i],
                    'tvec': tvecs[i],
                    'corner': corners[i]
                }
                detections.append(detection)
        
        return detections
    
    def cam_to_world_point(self, R_wc, t_wc, cam_point):
        """카메라 좌표를 월드 좌표로 변환"""
        return R_wc @ cam_point + t_wc
    
    def estimate_robot_pose(self):
        """로봇 위치 추정"""
        with self.detection_lock:
            current_detections = self.latest_detections.copy()
        
        # 모든 카메라에서 검출된 마커들 수집
        all_world_points = []
        current_time = time.time()
        
        for camera_name, data in current_detections.items():
            # 너무 오래된 검출 결과는 무시 (0.5초)
            if current_time - data['timestamp'] > 0.5:
                continue
                
            for detection in data['detections']:
                all_world_points.append((detection['id'], detection['world_position']))
        
        if len(all_world_points) < 3:
            # 최소 3개 마커가 필요
            return
        
        # 동일 ID 마커들의 평균 계산 (여러 카메라에서 관측된 경우)
        marker_world_positions = {}
        for marker_id, world_pos in all_world_points:
            if marker_id not in marker_world_positions:
                marker_world_positions[marker_id] = []
            marker_world_positions[marker_id].append(world_pos)
        
        # 평균 위치 계산
        avg_marker_positions = {}
        for marker_id, positions in marker_world_positions.items():
            avg_marker_positions[marker_id] = np.mean(positions, axis=0)
        
        # 로봇 포즈 계산 (Umeyama 알고리즘)
        robot_pose = self.calculate_robot_pose(avg_marker_positions)
        
        if robot_pose is not None:
            self.publish_robot_pose(robot_pose)
    
    def calculate_robot_pose(self, observed_markers):
        """관측된 마커들로부터 로봇 포즈 계산"""
        # 대응점 생성
        src_points = []  # 로봇 프레임에서의 마커 위치
        dst_points = []  # 월드 프레임에서의 관측된 마커 위치
        
        for marker_id, world_pos in observed_markers.items():
            if marker_id in self.robot_marker_coords:
                src_points.append(self.robot_marker_coords[marker_id])
                dst_points.append(world_pos)
        
        if len(src_points) < 3:
            return None
        
        src_points = np.array(src_points)
        dst_points = np.array(dst_points)
        
        # Umeyama 알고리즘으로 변환 행렬 계산
        R_robot_to_world, t_robot_to_world = self.umeyama_transform(src_points, dst_points)
        
        return {
            'position': t_robot_to_world.flatten(),
            'rotation_matrix': R_robot_to_world,
            'num_markers': len(src_points)
        }
    
    def umeyama_transform(self, src_points, dst_points):
        """Umeyama 알고리즘으로 3D 변환 계산"""
        src = np.array(src_points, dtype=np.float64)
        dst = np.array(dst_points, dtype=np.float64)
        
        # 중심점 계산
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        
        # 중심점으로부터의 편차
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid
        
        # 공분산 행렬
        H = src_centered.T @ dst_centered
        
        # SVD 분해
        U, S, Vt = np.linalg.svd(H)
        
        # 회전 행렬 계산
        R = Vt.T @ U.T
        
        # 반사 방지
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 평행이동 벡터 계산
        t = dst_centroid.reshape(3, 1) - R @ src_centroid.reshape(3, 1)
        
        return R, t
    
    def publish_robot_pose(self, robot_pose):
        """로봇 포즈 퍼블리시"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        # 위치 설정
        position = robot_pose['position']
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        # 회전을 쿼터니언으로 변환
        rotation_matrix = robot_pose['rotation_matrix']
        quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        self.pose_publisher.publish(msg)
        
        self.get_logger().info(
            f'Robot pose: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), '
            f'markers={robot_pose["num_markers"]}'
        )

def main(args=None):
    rclpy.init(args=args)
    
    robot_pose_estimator = RobotPoseEstimator()
    
    try:
        rclpy.spin(robot_pose_estimator)
    except KeyboardInterrupt:
        pass
    finally:
        robot_pose_estimator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()