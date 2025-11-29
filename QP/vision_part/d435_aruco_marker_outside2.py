#!/usr/bin/env python3
"""
ROS2를 사용하여 D435 카메라 토픽에서 ArUco 마커 6번과 7번의 위치를 추적하는 프로그램
- ArUco 마커 포즈 추정 기반 위치
- Depth 정보 기반 위치
- 두 방법의 결과를 비교 표시
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import time

class D435ArUcoTracker(Node):
    def __init__(self):
        super().__init__('d435_aruco_tracker')
        self.external_aruco_distances = []
        self.external_aruco_depth_distances = []

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # 마커 크기 (미터 단위)
        self.marker_size = 0.040  # 40mm
        
        # 추적할 마커 ID
        self.target_markers = [6, 7]
        
        # CV Bridge 초기화
        self.bridge = CvBridge()
        
        # 카메라 파라미터 저장용
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # 이미지 저장용 변수
        self.latest_color_image = None
        self.latest_depth_image = None
        self.color_timestamp = None
        self.depth_timestamp = None
        
        # 7번 위치 D435 카메라에서 받은 데이터 저장용
        self.d435_aruco_6_pose = None
        self.d435_depth_6_pose = None
        
        # 로컬 계산 결과 저장용
        self.local_marker_results = {}  # {marker_id: {'aruco': result, 'depth': result}}
        # 평균 데이터 저장용 (그래프 표시)
        self.avg_history = {
            'aruco_diff': [],  # ArUco vs ArUco 차이 평균들
            'aruco_depth_diff': [],  # ArUco vs Depth 차이 평균들
            'timestamps': []
        }
        # 구독자 설정 (개별 콜백 사용)
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', 
            self.camera_info_callback, 10
        )
        # 7번 위치 D435 카메라에서 발행하는 토픽 구독 (6번 마커만 볼 수 있음)
        self.d435_aruco_6_sub = self.create_subscription(
            Pose, '/aruco_pose',  # 7번 위치 D435에서 본 6번 마커
            self.d435_aruco_6_callback, 10
        )
        self.d435_depth_6_sub = self.create_subscription(
            Pose, '/depth_pose',  # 7번 위치 D435에서 본 6번 마커 (depth)
            self.d435_depth_6_callback, 10
        )
        # 퍼블리셔 설정 (각 마커별로 별도 토픽)
        # self.aruco_marker6_pub = self.create_publisher(
        #     Pose, '/aruco_marker_6_pose', 10
        # )
        # self.aruco_marker7_pub = self.create_publisher(
        #     Pose, '/aruco_marker_7_pose', 10
        # )
        # self.depth_marker6_pub = self.create_publisher(
        #     Pose, '/depth_marker_6_pose', 10
        # )
        # self.depth_marker7_pub = self.create_publisher(
        #     Pose, '/depth_marker_7_pose', 10
        # )
        
        # 시각화용 변수
        self.display_image = None
        self.show_display = True
        
        # self.get_logger().info("D435 ArUco Tracker 초기화 완료")
        # self.get_logger().info(f"추적 마커: {self.target_markers}")
        # self.get_logger().info(f"마커 크기: {self.marker_size * 1000}mm")
        
    def camera_info_callback(self, msg):
        """카메라 내부 파라미터 콜백"""
        if not self.camera_info_received:
            # 카메라 매트릭스 설정
            self.camera_matrix = np.array([
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]]
            ], dtype=np.float32)
            
            # 왜곡 계수 설정
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            
            self.camera_info_received = True
            # self.get_logger().info(f"카메라 매트릭스:\n{self.camera_matrix}")
            # self.get_logger().info(f"왜곡 계수: {self.dist_coeffs}")
    
    def color_callback(self, msg):
        """컬러 이미지 콜백"""
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_timestamp = msg.header.stamp
            self.try_process_images()
        except Exception as e:
            self.get_logger().error(f"컬러 이미지 변환 오류: {e}")
    
    def depth_callback(self, msg):
        """깊이 이미지 콜백"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.depth_timestamp = msg.header.stamp
            self.try_process_images()
        except Exception as e:
            self.get_logger().error(f"깊이 이미지 변환 오류: {e}")
    
    def try_process_images(self):
        """이미지가 모두 준비되면 처리 시작"""
        if (self.latest_color_image is not None and 
            self.latest_depth_image is not None and 
            self.camera_info_received):
            
            # 시간 차이 확인 (100ms 이내)
            if (abs((self.color_timestamp.sec + self.color_timestamp.nanosec * 1e-9) - 
                   (self.depth_timestamp.sec + self.depth_timestamp.nanosec * 1e-9)) < 0.5):
                
                self.process_images(self.latest_color_image, self.latest_depth_image)
        
    def get_position_from_aruco(self, corners, marker_id):
        """ArUco 마커 포즈 추정으로 위치 계산"""
        try:
            # 포즈 추정
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            if rvec is not None and tvec is not None:
                # 위치 벡터 (카메라 좌표계)
                position = tvec[0][0]  # [x, y, z]
                
                # 회전 벡터를 회전 행렬로 변환
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                
                return {
                    'position': position,
                    'rotation_vector': rvec[0],
                    'rotation_matrix': rotation_matrix,
                    'valid': True
                }
        except Exception as e:
            print(f"ArUco 포즈 추정 오류 (ID {marker_id}): {e}")
            
        return {'valid': False}
    
    def get_position_from_depth(self, corners, depth_image):
        """Depth 정보로 위치 계산"""
        try:
            # 마커의 코너 포인트들
            corner_points = corners[0].reshape(-1, 2)
            
            # 오른쪽 아래 코너 (index 2) 사용
            bottom_right_corner = (corner_points[2]+corner_points[1]+corner_points[0]+corner_points[3])/4  # [x, y]
            center_x = int(bottom_right_corner[0])+5
            center_y = int(bottom_right_corner[1])+5
            
            # 오른쪽 아래 코너 주변의 depth 값들 평균 계산 (노이즈 감소)
            window_size = 3  # 더 작은 윈도우 사용 (코너는 더 정확하므로)
            depth_values = []
            
            height, width = depth_image.shape
            
            for dy in range(-window_size, window_size + 1):
                for dx in range(-window_size, window_size + 1):
                    x = center_x + dx
                    y = center_y + dy
                    if 0 <= x < width and 0 <= y < height:
                        depth = depth_image[y, x]  # depth는 mm 단위
                        if depth > 0:  # 유효한 depth 값만
                            depth_values.append(depth / 1000.0)  # mm를 m로 변환
            
            if not depth_values:
                return {'valid': False}
            
            # 평균 depth 계산
            avg_depth = np.mean(depth_values)
            
            # 카메라 좌표계로 변환
            # 픽셀 좌표를 정규화된 카메라 좌표로 변환
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            
            x = (center_x - cx) * avg_depth / fx
            y = (center_y - cy) * avg_depth / fy
            z = avg_depth
            
            return {
                'position': np.array([x, y, z]),
                'depth_values': depth_values,
                'pixel_coords': (center_x, center_y),
                'corner_type': 'bottom_right',  # 사용한 코너 정보 추가
                'valid': True
            }
            
        except Exception as e:
            self.get_logger().error(f"Depth 기반 위치 계산 오류: {e}")            
            return {'valid': False}
    
    def process_images(self, color_image, depth_image):
        """이미지 처리 및 마커 감지"""
        # ArUco 마커 감지
        corners, ids, _ = cv2.aruco.detectMarkers(
            color_image, self.aruco_dict, parameters=self.aruco_params
        )
        
        # 결과 표시용 이미지 복사
        self.display_image = color_image.copy()
        
        # 현재 타임스탬프
        current_stamp = self.get_clock().now().to_msg()
        frame_id = "camera_color_optical_frame"
        
        # 감지된 마커 처리
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.target_markers:
                    corner = corners[i]
                    
                    # ArUco 기반 위치 계산
                    aruco_result = self.get_position_from_aruco(corner, marker_id)
                    
                    # Depth 기반 위치 계산
                    depth_result = self.get_position_from_depth(corner, depth_image)
                    
                    # 로컬 결과 저장
                    self.local_marker_results[marker_id] = {
                        'aruco': aruco_result,
                        'depth': depth_result
                    }
                    
                    # 결과 그리기
                    self.draw_results(self.display_image, marker_id, corner, 
                                    aruco_result, depth_result)
                    
                    # ROS2 메시지 생성 및 발행 (각 마커별로)
                    if aruco_result['valid']:
                        pose = Pose()
                        pos = aruco_result['position']
                        pose.position.x = float(pos[0])
                        pose.position.y = float(pos[1])
                        pose.position.z = float(pos[2])
                        
                        # 회전 행렬을 쿼터니언으로 변환
                        rot_matrix = aruco_result['rotation_matrix']
                        r = R.from_matrix(rot_matrix)
                        quat = r.as_quat()  # [x, y, z, w]
                        pose.orientation.x = float(quat[0])
                        pose.orientation.y = float(quat[1])
                        pose.orientation.z = float(quat[2])
                        pose.orientation.w = float(quat[3])
                        
                        # # 마커 ID에 따라 다른 토픽으로 발행
                        # if marker_id == 6:
                        #     self.aruco_marker6_pub.publish(pose)
                        # elif marker_id == 7:
                        #     self.aruco_marker7_pub.publish(pose)
                    
                    if depth_result['valid']:
                        pose = Pose()
                        pos = depth_result['position']
                        pose.position.x = float(pos[0])
                        pose.position.y = float(pos[1])
                        pose.position.z = float(pos[2])
                        # Depth 기반에서는 방향 정보가 없으므로 identity quaternion
                        pose.orientation.w = 1.0
                        
                        # # 마커 ID에 따라 다른 토픽으로 발행
                        # if marker_id == 6:
                        #     self.depth_marker6_pub.publish(pose)
                        # elif marker_id == 7:
                        #     self.depth_marker7_pub.publish(pose)
                    
                    # 콘솔에 결과 출력
                    # self.get_logger().info(f"\n=== 마커 ID {marker_id} ===")
                    if aruco_result['valid']:
                        pos = aruco_result['position']
                        # self.get_logger().info(f"ArUco 위치: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
                    
                    if depth_result['valid']:
                        pos = depth_result['position']
                        # self.get_logger().info(f"Depth 위치: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
                        # self.get_logger().info(f"사용된 depth 값 개수: {len(depth_result['depth_values'])}")
                    
                    if aruco_result['valid'] and depth_result['valid']:
                        diff = np.linalg.norm(aruco_result['position'] - depth_result['position'])
                        # self.get_logger().info(f"위치 차이: {diff:.4f}m")
                    
            # 6번과 7번 마커가 모두 감지되었을 때 마커 간 거리 비교
            self.compare_marker_distances()
        
        # 시각화 (선택적)
        if self.show_display:
            self.show_results()
    
    def d435_aruco_6_callback(self, msg):
        """7번 위치 D435에서 본 6번 마커 ArUco 위치 콜백"""
        self.d435_aruco_6_pose = msg
        pos = msg.position
        # self.get_logger().info(f"D435에서 본 6번 ArUco 위치: x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")

    def d435_depth_6_callback(self, msg):
        """7번 위치 D435에서 본 6번 마커 Depth 위치 콜백"""
        self.d435_depth_6_pose = msg
        pos = msg.position
        # self.get_logger().info(f"D435에서 본 6번 Depth 위치: x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")

    def compare_marker_distances(self):
        """6번과 7번 마커 간의 거리 비교 (외부 카메라 vs 7번 위치 D435)"""
        # 6번과 7번 마커가 모두 외부 카메라에서 감지되었는지 확인
        if 6 not in self.local_marker_results or 7 not in self.local_marker_results:
            return
        
        external_6_aruco = self.local_marker_results[6]['aruco']  # 외부 카메라에서 본 6번
        external_7_aruco = self.local_marker_results[7]['aruco']  # 외부 카메라에서 본 7번
        external_6_depth = self.local_marker_results[6]['depth']
        external_7_depth = self.local_marker_results[7]['depth']
        
        # ArUco 기반 거리 비교 부분 수정
        if (external_6_aruco['valid'] and external_7_aruco['valid'] and 
            self.d435_aruco_6_pose is not None):
            
            # 외부 카메라에서 본 6번-7번 간 거리 (ArUco)
            external_6_7_vec = external_6_aruco['position'] - external_7_aruco['position']
            external_aruco_distance = np.linalg.norm(external_6_7_vec)
            
            # 7번 위치에서 본 6번 마커까지의 거리 (ArUco)
            d435_6_pos = np.array([
                self.d435_aruco_6_pose.position.x,
                self.d435_aruco_6_pose.position.y,
                self.d435_aruco_6_pose.position.z
            ])
            d435_aruco_distance = np.linalg.norm(d435_6_pos)
            
            # 거리 차이 계산
            aruco_distance_diff = abs(external_aruco_distance - d435_aruco_distance)
            self.external_aruco_distances.append(aruco_distance_diff)
            
            # 100개마다 평균 계산 및 저장
            if len(self.external_aruco_distances) >= 100:
                ext_aruco_avg = np.mean(self.external_aruco_distances)
                self.avg_history['aruco_diff'].append(ext_aruco_avg)
                self.avg_history['timestamps'].append(time.time())
                
                sample_count = len(self.avg_history['aruco_diff'])
                print(f"ArUco Distance Difference Average #{sample_count}: {ext_aruco_avg:.4f}m")
                
                # 리스트 초기화
                self.external_aruco_distances = []
        
        # ArUco vs Depth 비교 부분 수정
        if (external_6_aruco['valid'] and external_7_aruco['valid'] and 
            self.d435_depth_6_pose is not None):
            
            # 외부 카메라에서 본 6번-7번 간 거리 (ArUco)
            external_6_7_vec = external_6_aruco['position'] - external_7_aruco['position']
            external_aruco_distance = np.linalg.norm(external_6_7_vec)
            
            # 7번 위치에서 본 6번 마커까지의 거리 (Depth)
            d435_6_pos = np.array([
                self.d435_depth_6_pose.position.x,
                self.d435_depth_6_pose.position.y,
                self.d435_depth_6_pose.position.z
            ])
            d435_depth_distance = np.linalg.norm(d435_6_pos)
            
            # ArUco vs Depth 차이 계산
            aruco_depth_distance_diff = abs(external_aruco_distance - d435_depth_distance)
            self.external_aruco_depth_distances.append(aruco_depth_distance_diff)
            
        # 100개마다 평균 계산 및 저장
        if len(self.external_aruco_depth_distances) >= 100:
            ext_aruco_depth_avg = np.mean(self.external_aruco_depth_distances)
            self.avg_history['aruco_depth_diff'].append(ext_aruco_depth_avg)
            
            sample_count = len(self.avg_history['aruco_depth_diff'])
            print(f"ArUco-Depth Distance Difference Average #{sample_count}: {ext_aruco_depth_avg:.4f}m")
            
            # 리스트 초기화
            self.external_aruco_depth_distances = []
            
            # self.get_logger().info(f"\n=== Depth Method Distance Comparison ===")
            # self.get_logger().info(f"External camera 6-7 distance: {external_depth_distance:.4f}m")
            # self.get_logger().info(f"External camera 6-7 vector: x={external_6_7_vec[0]:.4f}, y={external_6_7_vec[1]:.4f}, z={external_6_7_vec[2]:.4f}")
            # self.get_logger().info(f"D435 camera to marker 6 distance: {d435_depth_distance:.4f}m")
            # self.get_logger().info(f"D435 camera to marker 6 position: x={d435_6_pos[0]:.4f}, y={d435_6_pos[1]:.4f}, z={d435_6_pos[2]:.4f}")
            # self.get_logger().info(f"TOTAL DISTANCE DIFFERENCE: {depth_distance_diff:.4f}m")
            # self.get_logger().info(f"Coordinate differences: dx={coord_diff[0]:.4f}, dy={coord_diff[1]:.4f}, dz={coord_diff[2]:.4f}")
        
        # ArUco vs Depth 방법 직접 비교
        if (external_6_aruco['valid'] and external_7_aruco['valid'] and 
            external_6_depth['valid'] and external_7_depth['valid'] and 
            self.d435_aruco_6_pose is not None and self.d435_depth_6_pose is not None):
            
            # 외부 카메라의 ArUco vs Depth 차이
            ext_aruco_vec = external_6_aruco['position'] - external_7_aruco['position']
            ext_depth_vec = external_6_depth['position'] - external_7_depth['position']
            ext_aruco_dist = np.linalg.norm(ext_aruco_vec)
            ext_depth_dist = np.linalg.norm(ext_depth_vec)
            ext_method_diff = abs(ext_aruco_dist - ext_depth_dist)
            
            # D435 카메라의 ArUco vs Depth 차이
            d435_aruco_pos = np.array([
                self.d435_aruco_6_pose.position.x,
                self.d435_aruco_6_pose.position.y,
                self.d435_aruco_6_pose.position.z
            ])
            d435_depth_pos = np.array([
                self.d435_depth_6_pose.position.x,
                self.d435_depth_6_pose.position.y,
                self.d435_depth_6_pose.position.z
            ])
            d435_aruco_dist = np.linalg.norm(d435_aruco_pos)
            d435_depth_dist = np.linalg.norm(d435_depth_pos)
            d435_method_diff = abs(d435_aruco_dist - d435_depth_dist)
            
            # self.get_logger().info(f"\n=== Method Comparison (ArUco vs Depth) ===")
            # self.get_logger().info(f"External camera method difference: {ext_method_diff:.4f}m (ArUco: {ext_aruco_dist:.4f}m, Depth: {ext_depth_dist:.4f}m)")
            # self.get_logger().info(f"D435 camera method difference: {d435_method_diff:.4f}m (ArUco: {d435_aruco_dist:.4f}m, Depth: {d435_depth_dist:.4f}m)")
            # self.get_logger().info(f"="*50)
    def show_final_graph(self):
        """최종 그래프 표시"""
        import matplotlib.pyplot as plt
        
        # 데이터 확인
        aruco_data = self.avg_history['aruco_diff']
        aruco_depth_data = self.avg_history['aruco_depth_diff']
        
        if len(aruco_data) == 0 and len(aruco_depth_data) == 0:
            print("표시할 그래프 데이터가 없습니다. (100개 이상의 샘플이 필요)")
            return
        
        # 그래프 설정
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # ArUco vs ArUco 차이 그래프
        if len(aruco_data) > 0:
            x_aruco = list(range(1, len(aruco_data) + 1))
            ax1.plot(x_aruco, aruco_data, 'b-o', linewidth=2, markersize=8, 
                    label=f'ArUco vs ArUco (average of {len(aruco_data)} data points)')
            ax1.set_title('ArUco Distance Difference Averages\n(External Camera vs D435 Camera)', 
                        fontsize=14, fontweight='bold')
            ax1.set_ylabel('Distance Difference (m)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=11)
            
            # 통계 정보 추가
            mean_aruco = np.mean(aruco_data)
            std_aruco = np.std(aruco_data)
            max_aruco = np.max(aruco_data)
            min_aruco = np.min(aruco_data)
            
            stats_text = f'Mean: {mean_aruco:.4f}m\nStd: {std_aruco:.4f}m\nMax: {max_aruco:.4f}m\nMin: {min_aruco:.4f}m'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # 평균선 표시
            ax1.axhline(y=mean_aruco, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_aruco:.4f}m')
            ax1.legend(fontsize=11)
        else:
            ax1.text(0.5, 0.5, 'No ArUco data available\n(Need 100+ samples)', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=12)
            ax1.set_title('ArUco Distance Difference Averages - No Data')
        
        # ArUco vs Depth 차이 그래프
        if len(aruco_depth_data) > 0:
            x_aruco_depth = list(range(1, len(aruco_depth_data) + 1))
            ax2.plot(x_aruco_depth, aruco_depth_data, 'r-s', linewidth=2, markersize=8,
                    label=f'ArUco vs Depth (average of {len(aruco_depth_data)} data points)')
            ax2.set_title('ArUco vs Depth Distance Difference Averages\n(External ArUco vs D435 Depth)', 
                        fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sample Number (×100 measurements)', fontsize=12)
            ax2.set_ylabel('Distance Difference (m)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=11)
            
            # 통계 정보 추가
            mean_aruco_depth = np.mean(aruco_depth_data)
            std_aruco_depth = np.std(aruco_depth_data)
            max_aruco_depth = np.max(aruco_depth_data)
            min_aruco_depth = np.min(aruco_depth_data)
            
            stats_text = f'Mean: {mean_aruco_depth:.4f}m\nStd: {std_aruco_depth:.4f}m\nMax: {max_aruco_depth:.4f}m\nMin: {min_aruco_depth:.4f}m'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            # 평균선 표시
            ax2.axhline(y=mean_aruco_depth, color='blue', linestyle='--', alpha=0.7, label=f'Mean: {mean_aruco_depth:.4f}m')
            ax2.legend(fontsize=11)
        else:
            ax2.text(0.5, 0.5, 'No ArUco-Depth data available\n(Need 100+ samples)', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('ArUco vs Depth Distance Difference Averages - No Data')
            ax2.set_xlabel('Sample Number (×100 measurements)', fontsize=12)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 전체 통계 출력
        print("\n" + "="*60)
        print("=== FINAL STATISTICS ===")
        if len(aruco_data) > 0:
            print(f"ArUco vs ArUco:")
            print(f"  - Total averages: {len(aruco_data)}")
            print(f"  - Total samples: {len(aruco_data) * 100}")
            print(f"  - Overall mean: {np.mean(aruco_data):.6f}m")
            print(f"  - Standard deviation: {np.std(aruco_data):.6f}m")
            print(f"  - Min: {np.min(aruco_data):.6f}m")
            print(f"  - Max: {np.max(aruco_data):.6f}m")
        
        if len(aruco_depth_data) > 0:
            print(f"ArUco vs Depth:")
            print(f"  - Total averages: {len(aruco_depth_data)}")
            print(f"  - Total samples: {len(aruco_depth_data) * 100}")
            print(f"  - Overall mean: {np.mean(aruco_depth_data):.6f}m")
            print(f"  - Standard deviation: {np.std(aruco_depth_data):.6f}m")
            print(f"  - Min: {np.min(aruco_depth_data):.6f}m")
            print(f"  - Max: {np.max(aruco_depth_data):.6f}m")
        print("="*60)
        
        # 그래프 표시
        plt.show()

    def draw_results(self, frame, marker_id, corners, aruco_result, depth_result):
        """결과 시각화"""
        if self.display_image is not None:
            # 정보 패널 추가
            info_text = [
                f"Target Markers: {self.target_markers}",
                f"Marker Size: {self.marker_size * 1000}mm",
                "Press 'q' to quit and show graph"  # 메시지 수정
            ]
            
            # 현재 수집 상태 표시
            aruco_count = len(self.external_aruco_distances)
            aruco_depth_count = len(self.external_aruco_depth_distances)
            total_aruco_avgs = len(self.avg_history['aruco_diff'])
            total_aruco_depth_avgs = len(self.avg_history['aruco_depth_diff'])
            
            info_text.extend([
                f"ArUco samples: {aruco_count}/100 (Avgs: {total_aruco_avgs})",
                f"ArUco-Depth samples: {aruco_depth_count}/100 (Avgs: {total_aruco_depth_avgs})"
            ])
            
            for i, text in enumerate(info_text):
                cv2.putText(self.display_image, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 이미지 표시
            cv2.imshow('D435 ArUco Tracker', self.display_image)
            
            # 종료 조건 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_display = False
                cv2.destroyAllWindows()
                
                # 그래프 표시
                self.get_logger().info("이미지 창을 닫고 그래프를 표시합니다...")
                self.show_final_graph()
                
                # ROS2 종료
                rclpy.shutdown()
        
    def show_results(self):
        """결과 시각화"""
        if self.display_image is not None:
            # 정보 패널 추가
            info_text = [
                f"Target Markers: {self.target_markers}",
                f"Marker Size: {self.marker_size * 1000}mm",
                "Press 'q' to quit and show graph"  # 메시지 수정
            ]
            
            # 현재 수집 상태 표시
            aruco_count = len(self.external_aruco_distances)
            aruco_depth_count = len(self.external_aruco_depth_distances)
            total_aruco_avgs = len(self.avg_history['aruco_diff'])
            total_aruco_depth_avgs = len(self.avg_history['aruco_depth_diff'])
            
            info_text.extend([
                f"ArUco samples: {aruco_count}/100 (Avgs: {total_aruco_avgs})",
                f"ArUco-Depth samples: {aruco_depth_count}/100 (Avgs: {total_aruco_depth_avgs})"
            ])
            
            for i, text in enumerate(info_text):
                cv2.putText(self.display_image, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 이미지 표시
            cv2.imshow('D435 ArUco Tracker', self.display_image)
            
            # 종료 조건 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_display = False
                cv2.destroyAllWindows()
                
                # 그래프 표시
                self.get_logger().info("이미지 창을 닫고 그래프를 표시합니다...")
                self.show_final_graph()
                
                # ROS2 종료
                rclpy.shutdown()
    
    def run(self):
        """메인 실행 루프"""
        self.get_logger().info("ArUco 마커 추적 시작... 'q'를 눌러 종료하고 그래프 표시")
        
        try:
            while rclpy.ok() and self.show_display:
                rclpy.spin_once(self, timeout_sec=0.1)
                
        except KeyboardInterrupt:
            self.get_logger().info("프로그램이 중단되었습니다.")
            # 중단되어도 그래프 표시
            if len(self.avg_history['aruco_diff']) > 0 or len(self.avg_history['aruco_depth_diff']) > 0:
                self.show_final_graph()
        
        finally:
            # 리소스 정리
            cv2.destroyAllWindows()
            self.get_logger().info("프로그램 종료")

def main():
    """메인 함수"""
    try:
        rclpy.init()
        tracker = D435ArUcoTracker()
        tracker.run()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
