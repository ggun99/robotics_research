#!/usr/bin/env python3
"""
ROS2를 사용하여 좌표 변환 정확도를 검증하는 프로그램
- 외부 카메라에서 본 6번, 7번 마커 위치 구독
- 7번 위치 카메라에서 6번 마커를 직접 감지
- 좌표 변환을 통해 계산된 6번 위치와 실제 6번 위치 비교
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import time
from collections import deque

class ArUcoPositionComparison(Node):
    def __init__(self):
        super().__init__('aruco_position_comparison')
        
        # CV Bridge 초기화
        self.bridge = CvBridge()
        
        # 위치 데이터 저장용
        self.aruco_pose = None
        self.depth_pose = None
        self.aruco_timestamp = None
        self.depth_timestamp = None
        
        # 히스토리 저장용 (차이 그래프 그리기)
        self.position_history = deque(maxlen=100)  # 최근 100개 데이터
        self.time_history = deque(maxlen=100)
        
        # 카메라 이미지 구독 (시각화용)
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.color_callback, 10
        )
        
        # ArUco 마커 6번 위치 구독
        self.aruco_marker6_sub = self.create_subscription(
            Pose, '/aruco_pose',
            self.aruco_callback, 10
        )
        
        # Depth 마커 6번 위치 구독 (만약 발행한다면)
        self.depth_marker6_sub = self.create_subscription(
            Pose, '/depth_pose',
            self.depth_callback, 10
        )
        
        # 시각화용 변수
        self.display_image = None
        self.show_display = True
        
        # 통계 변수
        self.total_measurements = 0
        self.total_difference = 0.0
        self.max_difference = 0.0
        self.min_difference = float('inf')
        
        self.get_logger().info("ArUco Position Comparison 초기화 완료")
        self.get_logger().info("구독 토픽:")
        self.get_logger().info("  - /aruco_marker_6_pose")
        self.get_logger().info("  - /depth_marker_6_pose")
        self.get_logger().info("  - /camera/camera/color/image_raw")
        
    def color_callback(self, msg):
        """컬러 이미지 콜백 (시각화용)"""
        try:
            self.display_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.visualize_comparison()
        except Exception as e:
            self.get_logger().error(f"컬러 이미지 변환 오류: {e}")
    
    def aruco_callback(self, msg):
        """ArUco 마커 6번 위치 콜백"""
        self.aruco_pose = msg
        self.aruco_timestamp = self.get_clock().now()
        self.calculate_difference()
        
        # 로그 출력
        pos = msg.position
        self.get_logger().info(f"ArUco 6번 위치: x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")
    
    def depth_callback(self, msg):
        """Depth 마커 6번 위치 콜백"""
        self.depth_pose = msg
        self.depth_timestamp = self.get_clock().now()
        self.calculate_difference()
        
        # 로그 출력
        pos = msg.position
        self.get_logger().info(f"Depth 6번 위치: x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")
    
    def calculate_difference(self):
        """두 위치 간의 차이 계산"""
        if self.aruco_pose is None or self.depth_pose is None:
            return
        
        # 시간 동기화 확인 (1초 이내)
        if (abs((self.aruco_timestamp.nanoseconds - self.depth_timestamp.nanoseconds) * 1e-9) > 1.0):
            return
        
        # 위치 차이 계산
        aruco_pos = np.array([
            self.aruco_pose.position.x,
            self.aruco_pose.position.y,
            self.aruco_pose.position.z
        ])
        
        depth_pos = np.array([
            self.depth_pose.position.x,
            self.depth_pose.position.y,
            self.depth_pose.position.z
        ])
        
        # 유클리드 거리 계산
        difference = np.linalg.norm(aruco_pos - depth_pos)
        
        # 히스토리에 추가
        current_time = time.time()
        self.position_history.append({
            'aruco_pos': aruco_pos,
            'depth_pos': depth_pos,
            'difference': difference,
            'timestamp': current_time
        })
        self.time_history.append(current_time)
        
        # 통계 업데이트
        self.total_measurements += 1
        self.total_difference += difference
        self.max_difference = max(self.max_difference, difference)
        self.min_difference = min(self.min_difference, difference)
        
        # 차이 로그 출력
        self.get_logger().info(f"위치 차이: {difference:.4f}m")
        self.get_logger().info(f"평균 차이: {self.total_difference/self.total_measurements:.4f}m")
    
    def visualize_comparison(self):
        """비교 결과를 이미지에 시각화"""
        if self.display_image is None:
            return
        
        # 이미지 복사
        vis_image = self.display_image.copy()
        
        # 정보 패널 배경
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (600, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
        
        # 텍스트 정보 표시
        y_offset = 40
        line_height = 25
        
        # 제목
        cv2.putText(vis_image, "ArUco vs Depth Position Comparison", 
                   (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height * 2
        
        # ArUco 위치 정보
        if self.aruco_pose is not None:
            pos = self.aruco_pose.position
            cv2.putText(vis_image, f"ArUco Position:", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += line_height
            cv2.putText(vis_image, f"  X: {pos.x:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(vis_image, f"  Y: {pos.y:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(vis_image, f"  Z: {pos.z:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # 카메라로부터의 거리
            aruco_distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            cv2.putText(vis_image, f"  Distance: {aruco_distance:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height * 1.5
        
        # Depth 위치 정보
        if self.depth_pose is not None:
            pos = self.depth_pose.position
            cv2.putText(vis_image, f"Depth Position:", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
            cv2.putText(vis_image, f"  X: {pos.x:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(vis_image, f"  Y: {pos.y:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(vis_image, f"  Z: {pos.z:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # 카메라로부터의 거리
            depth_distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            cv2.putText(vis_image, f"  Distance: {depth_distance:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height * 1.5
        
        # 차이 정보
        if len(self.position_history) > 0:
            latest = self.position_history[-1]
            cv2.putText(vis_image, f"Position Difference:", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            cv2.putText(vis_image, f"  Current: {latest['difference']:.4f}m", 
                       (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            if self.total_measurements > 0:
                avg_diff = self.total_difference / self.total_measurements
                cv2.putText(vis_image, f"  Average: {avg_diff:.4f}m", 
                           (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(vis_image, f"  Max: {self.max_difference:.4f}m", 
                           (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(vis_image, f"  Min: {self.min_difference:.4f}m", 
                           (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(vis_image, f"  Samples: {self.total_measurements}", 
                           (20, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 차이 히스토리 그래프 그리기
        self.draw_difference_graph(vis_image)
        
        # 3D 위치 시각화
        self.draw_3d_positions(vis_image)
        
        # 이미지 표시
        cv2.imshow('ArUco Position Comparison', vis_image)
        
        # 종료 조건 확인
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.show_display = False
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    def draw_difference_graph(self, image):
        """차이 히스토리를 그래프로 그리기"""
        if len(self.position_history) < 2:
            return
        
        # 그래프 영역 설정
        graph_x = 650
        graph_y = 50
        graph_width = 300
        graph_height = 150
        
        # 그래프 배경
        cv2.rectangle(image, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (50, 50, 50), -1)
        cv2.rectangle(image, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (255, 255, 255), 2)
        
        # 제목
        cv2.putText(image, "Difference History", 
                   (graph_x + 10, graph_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 데이터 준비
        differences = [entry['difference'] for entry in self.position_history]
        max_diff = max(differences) if differences else 1.0
        min_diff = min(differences) if differences else 0.0
        diff_range = max_diff - min_diff if max_diff > min_diff else 1.0
        
        # 그래프 그리기
        points = []
        for i, diff in enumerate(differences):
            x = graph_x + int((i / len(differences)) * graph_width)
            y = graph_y + graph_height - int(((diff - min_diff) / diff_range) * graph_height)
            points.append((x, y))
        
        # 선 그리기
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(image, points[i-1], points[i], (0, 255, 0), 2)
        
        # 축 레이블
        cv2.putText(image, f"{max_diff:.3f}", 
                   (graph_x - 50, graph_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"{min_diff:.3f}", 
                   (graph_x - 50, graph_y + graph_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_3d_positions(self, image):
        """3D 위치를 2D로 투영해서 시각화"""
        if self.aruco_pose is None or self.depth_pose is None:
            return
        
        # 3D 시각화 영역
        viz_x = 650
        viz_y = 220
        viz_size = 200
        
        # 배경
        cv2.rectangle(image, (viz_x, viz_y), 
                     (viz_x + viz_size, viz_y + viz_size), (30, 30, 30), -1)
        cv2.rectangle(image, (viz_x, viz_y), 
                     (viz_x + viz_size, viz_y + viz_size), (255, 255, 255), 2)
        
        # 제목
        cv2.putText(image, "3D Positions (Top View)", 
                   (viz_x + 10, viz_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 중심점
        center_x = viz_x + viz_size // 2
        center_y = viz_y + viz_size // 2
        
        # 좌표축 그리기
        cv2.line(image, (center_x, viz_y), (center_x, viz_y + viz_size), (100, 100, 100), 1)
        cv2.line(image, (viz_x, center_y), (viz_x + viz_size, center_y), (100, 100, 100), 1)
        
        # 스케일 계산 (최대 거리 기준)
        aruco_pos = np.array([self.aruco_pose.position.x, self.aruco_pose.position.y, self.aruco_pose.position.z])
        depth_pos = np.array([self.depth_pose.position.x, self.depth_pose.position.y, self.depth_pose.position.z])
        
        max_distance = max(np.linalg.norm(aruco_pos), np.linalg.norm(depth_pos))
        scale = (viz_size // 2 - 20) / max(max_distance, 0.1)
        
        # ArUco 위치 (X-Z 평면)
        aruco_x = center_x + int(self.aruco_pose.position.x * scale)
        aruco_z = center_y - int(self.aruco_pose.position.z * scale)  # Y축 뒤집기
        cv2.circle(image, (aruco_x, aruco_z), 5, (255, 0, 0), -1)
        cv2.putText(image, "ArUco", (int(aruco_x + 10), int(aruco_z)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Depth 위치 (X-Z 평면)
        depth_x = center_x + int(self.depth_pose.position.x * scale)
        depth_z = center_y - int(self.depth_pose.position.z * scale)  # Y축 뒤집기
        cv2.circle(image, (depth_x, depth_z), 5, (0, 0, 255), -1)
        cv2.putText(image, "Depth", (int(depth_x + 10), int(depth_z)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 두 점 사이 연결선
        cv2.line(image, (aruco_x, aruco_z), (depth_x, depth_z), (0, 255, 0), 2)
        
        # 카메라 위치 (원점)
        cv2.circle(image, (center_x, center_y), 3, (255, 255, 255), -1)
        cv2.putText(image, "Camera", (int(center_x + 5), int(center_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def run(self):
        """메인 실행 루프"""
        self.get_logger().info("ArUco Position Comparison 시작... 'q'를 눌러 종료")
        
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
        comparator = ArUcoPositionComparison()
        comparator.run()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
