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
        
        # ArUco 기반 거리 비교
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
            if len(self.external_aruco_distances) >= 100:
                ext_aruco_avg = np.mean(self.external_aruco_distances)
                print(f"External ArUco Distance Difference Average over 100 samples: {ext_aruco_avg:.4f}m")
                self.external_aruco_distances = []
            # 좌표별 차이 (외부 6-7 벡터 vs D435 6번 위치)
            coord_diff = external_6_7_vec - d435_6_pos
            
            # self.get_logger().info(f"\n=== ArUco Method Distance Comparison ===")
            # self.get_logger().info(f"External camera 6-7 distance: {external_aruco_distance:.4f}m")
            # self.get_logger().info(f"External camera 6-7 vector: x={external_6_7_vec[0]:.4f}, y={external_6_7_vec[1]:.4f}, z={external_6_7_vec[2]:.4f}")
            # self.get_logger().info(f"D435 camera to marker 6 distance: {d435_aruco_distance:.4f}m")
            # self.get_logger().info(f"D435 camera to marker 6 position: x={d435_6_pos[0]:.4f}, y={d435_6_pos[1]:.4f}, z={d435_6_pos[2]:.4f}")
            # self.get_logger().info(f"TOTAL DISTANCE DIFFERENCE: {aruco_distance_diff:.4f}m")
            # self.get_logger().info(f"Coordinate differences: dx={coord_diff[0]:.4f}, dy={coord_diff[1]:.4f}, dz={coord_diff[2]:.4f}")
        
        # Depth 기반 거리 비교
        if (external_6_depth['valid'] and external_7_depth['valid'] and 
            self.d435_depth_6_pose is not None):
            
            # 외부 카메라에서 본 6번-7번 간 거리 (Depth)
            external_6_7_vec = external_6_depth['position'] - external_7_depth['position']
            external_depth_distance = np.linalg.norm(external_6_7_vec)
            
            # 7번 위치에서 본 6번 마커까지의 거리 (Depth)
            d435_6_pos = np.array([
                self.d435_depth_6_pose.position.x,
                self.d435_depth_6_pose.position.y,
                self.d435_depth_6_pose.position.z
            ])
            d435_depth_distance = np.linalg.norm(d435_6_pos)
            
            # 거리 차이 계산
            depth_distance_diff = abs(external_depth_distance - d435_depth_distance)
            aruco_depth_distance_diff = abs(external_aruco_distance - d435_depth_distance)
            self.external_aruco_depth_distances.append(aruco_depth_distance_diff)
            if len(self.external_aruco_depth_distances) >= 100:
                ext_aruco_depth_avg = np.mean(self.external_aruco_depth_distances)
                print(f"External ArUco-Depth Distance Difference Average over 100 samples: {ext_aruco_depth_avg:.4f}m")
                self.external_aruco_depth_distances = []
            # 좌표별 차이
            coord_diff = external_6_7_vec - d435_6_pos
            
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

    def draw_results(self, frame, marker_id, corners, aruco_result, depth_result):
        """결과를 프레임에 그리기"""
        # 마커 테두리 그리기
        cv2.aruco.drawDetectedMarkers(frame, [corners])
        
        # 마커 ID 표시
        corner_points = corners.reshape(-1, 2)
        center = tuple(map(int, np.mean(corner_points, axis=0)))
        
        # 마커 ID에 따라 텍스트 위치 조정 (겹치지 않도록)
        if marker_id == 6:
            text_x = center[0] + 70  # 6번 마커는 오른쪽
            text_y_start = center[1] - 60  
        elif marker_id == 7:
            text_x = center[0] - 200  # 7번 마커는 왼쪽
            text_y_start = center[1] - 60
        else:
            text_x = center[0] + 50  # 기본값
            text_y_start = center[1] - 40
        
        # ID 텍스트
        cv2.putText(frame, f"ID: {marker_id}", 
                   (text_x, text_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ArUco 결과 표시
        if aruco_result['valid']:
            pos = aruco_result['position']
            text = f"ArUco: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            cv2.putText(frame, text, 
                       (text_x, text_y_start + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 좌표축 그리기
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                              aruco_result['rotation_vector'], 
                              aruco_result['position'], self.marker_size)
        
        # draw_results 함수의 Depth 결과 표시 부분 수정
        if depth_result['valid']:
            pos = depth_result['position']
            corner_type = depth_result.get('corner_type', 'center')
            text = f"Depth({corner_type}): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            cv2.putText(frame, text, 
                    (text_x, text_y_start + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 사용한 코너점 강조 표시
            pixel_coords = depth_result['pixel_coords']
            cv2.circle(frame, pixel_coords, 5, (0, 0, 255), -1)  # 빨간 원
            cv2.circle(frame, pixel_coords, 8, (255, 255, 255), 2)  # 흰색 테두리
            
            # 코너 라벨 표시
            cv2.putText(frame, "BR", (pixel_coords[0] + 10, pixel_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 차이 계산 및 표시
        if aruco_result['valid'] and depth_result['valid']:
            diff = np.linalg.norm(aruco_result['position'] - depth_result['position'])
            text = f"Local Diff: {diff:.3f}m"
            cv2.putText(frame, text, 
                       (text_x, text_y_start + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 마커 간 거리 차이 표시 (화면 상단에)
        if (6 in self.local_marker_results and 7 in self.local_marker_results and
            marker_id == 6):  # 6번 마커일 때만 한 번 표시
            
            external_6_aruco = self.local_marker_results[6]['aruco']
            external_7_aruco = self.local_marker_results[7]['aruco']
            external_6_depth = self.local_marker_results[6]['depth']
            external_7_depth = self.local_marker_results[7]['depth']
            
            y_pos = 120
            
            # === ArUco 기반 비교 ===
            cv2.putText(frame, "=== ArUco Method ===", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            
            if (external_6_aruco['valid'] and external_7_aruco['valid'] and 
                self.d435_aruco_6_pose is not None):
                
                # 외부 카메라에서 본 6번-7번 간 벡터와 거리
                external_6_7_vec = external_6_aruco['position'] - external_7_aruco['position']
                external_aruco_distance = np.linalg.norm(external_6_7_vec)
                
                # 7번 위치에서 본 6번 마커 위치와 거리
                d435_6_pos = np.array([
                    self.d435_aruco_6_pose.position.x,
                    self.d435_aruco_6_pose.position.y,
                    self.d435_aruco_6_pose.position.z
                ])
                d435_aruco_distance = np.linalg.norm(d435_6_pos)
                
                # 거리 차이와 좌표별 차이
                aruco_distance_diff = abs(external_aruco_distance - d435_aruco_distance)
                coord_diff = external_6_7_vec - d435_6_pos
                
                # ArUco 결과 표시
                cv2.putText(frame, f"External 6-7 distance: {external_aruco_distance:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"D435 to marker 6: {d435_aruco_distance:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"Total distance diff: {aruco_distance_diff:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_pos += 20
                cv2.putText(frame, f"Coord diff: dx={coord_diff[0]:.3f}, dy={coord_diff[1]:.3f}, dz={coord_diff[2]:.3f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y_pos += 35
            else:
                cv2.putText(frame, "ArUco data not available", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                y_pos += 35
            
            # === Depth 기반 비교 ===
            cv2.putText(frame, "=== Depth Method ===", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_pos += 30
            
            if (external_6_depth['valid'] and external_7_depth['valid'] and 
                self.d435_depth_6_pose is not None):
                
                # 외부 카메라에서 본 6번-7번 간 벡터와 거리
                external_6_7_vec = external_6_depth['position'] - external_7_depth['position']
                external_depth_distance = np.linalg.norm(external_6_7_vec)
                
                # 7번 위치에서 본 6번 마커 위치와 거리
                d435_6_pos = np.array([
                    self.d435_depth_6_pose.position.x,
                    self.d435_depth_6_pose.position.y,
                    self.d435_depth_6_pose.position.z
                ])
                d435_depth_distance = np.linalg.norm(d435_6_pos)
                
                # 거리 차이와 좌표별 차이
                depth_distance_diff = abs(external_depth_distance - d435_depth_distance)
                coord_diff = external_6_7_vec - d435_6_pos
                
                # Depth 결과 표시
                cv2.putText(frame, f"External 6-7 distance: {external_depth_distance:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"D435 to marker 6: {d435_depth_distance:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"Total distance diff: {depth_distance_diff:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                y_pos += 20
                cv2.putText(frame, f"Coord diff: dx={coord_diff[0]:.3f}, dy={coord_diff[1]:.3f}, dz={coord_diff[2]:.3f}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            else:
                cv2.putText(frame, "Depth data not available", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                y_pos += 25
            
            # === 방법별 비교 (ArUco vs Depth) ===
            if (external_6_aruco['valid'] and external_7_aruco['valid'] and 
                external_6_depth['valid'] and external_7_depth['valid'] and 
                self.d435_aruco_6_pose is not None and self.d435_depth_6_pose is not None):
                
                y_pos += 10
                cv2.putText(frame, "=== Method Comparison ===", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
                
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
                
                cv2.putText(frame, f"External ArUco vs Depth: {ext_method_diff:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 18
                cv2.putText(frame, f"D435 ArUco vs Depth: {d435_method_diff:.3f}m", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def show_results(self):
        """결과 시각화"""
        if self.display_image is not None:
            # 정보 패널 추가
            info_text = [
                f"Target Markers: {self.target_markers}",
                f"Marker Size: {self.marker_size * 1000}mm",
                "Press 'q' to quit"
            ]
            
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
                rclpy.shutdown()
    
    def run(self):
        """메인 실행 루프"""
        self.get_logger().info("ArUco 마커 추적 시작... 'q'를 눌러 종료")
        
        try:
            while rclpy.ok() and self.show_display:
                rclpy.spin_once(self, timeout_sec=0.1)
                
        except KeyboardInterrupt:
            self.get_logger().info("프로그램이 중단되었습니다.")
        
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
