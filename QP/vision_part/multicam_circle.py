import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import os
from multicam_aruco import ARUCOBoardPose, ARUCORobotPose
import cv2
import matplotlib.pyplot as plt

class World_Robot(Node):
    def __init__(self):
        super().__init__('world_robot')
        self.bridge = CvBridge()
        self.world_aruco_detector = ARUCOBoardPose()
        self.robot_aruco_detector = ARUCORobotPose()

        self.camera_1_sub = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.image_1_callback,
            10)
        self.camera_2_sub = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.image_2_callback,
            10)
        self.camera_3_sub = self.create_subscription(
            Image,
            '/camera3/camera3/color/image_raw',
            self.image_3_callback,
            10)
        self.camera_info_1 = self.create_subscription(
            CameraInfo,
            '/camera1/camera1/color/camera_info',
            self.camera1_info_callback,
            1)
        self.camera_info_2 = self.create_subscription(
            CameraInfo,
            '/camera2/camera2/color/camera_info',
            self.camera2_info_callback,
            1)
        self.camera_info_3 = self.create_subscription(
            CameraInfo,
            '/camera3/camera3/color/camera_info',
            self.camera3_info_callback,
            1)
        
        self.create_timer(0.1, self.loop)

        self.x_pose = []
        self.y_pose = []
        self.timestamps = []  # 시간 추적용

        self.camera1_info = None
        self.camera2_info = None
        self.camera3_info = None
        self.camera1_image = None
        self.camera2_image = None
        self.camera2_image_robot = None
        self.camera3_image = None

        self.H_cam1_to_cam2 = None
        self.H_cam2_to_cam1 = None
        self.H_cam2_to_cam3 = None

        
    def image_1_callback(self, msg:Image):
        self.camera1_image = msg
    def image_2_callback(self, msg:Image):
        self.camera2_image = msg
    def image_3_callback(self, msg:Image):
        self.camera3_image = msg
    def camera1_info_callback(self, msg:CameraInfo):
        if self.camera1_info is None:
            self.camera1_info = msg
    def camera2_info_callback(self, msg:CameraInfo):
        if self.camera2_info is None:
            self.camera2_info = msg
    def camera3_info_callback(self, msg:CameraInfo):
        if self.camera3_info is None:
            self.camera3_info = msg

    def loop(self):
        if self.H_cam1_to_cam2 is None:
            H_cam1_to_cam2, baseline_12, ts_12 = self.load_transformation_matrix('stereo_calibration_results/H_camera1_camera2_current.yaml')
            self.H_cam1_to_cam2 = H_cam1_to_cam2
            self.H_cam2_to_cam1 = self.inverse_homogeneous_matrix(self.H_cam1_to_cam2)

        if self.H_cam2_to_cam3 is None:
            H_cam2_to_cam3, baseline_23, ts_23 = self.load_transformation_matrix('stereo_calibration_results/H_camera2_camera3_current.yaml')
            self.H_cam2_to_cam3 = H_cam2_to_cam3

        if self.camera1_image is None or self.camera2_image is None or self.camera3_image is None:
            return

        if self.camera1_info is None or self.camera2_info is None or self.camera3_info is None:
            return
        
        img_1 = self.bridge.imgmsg_to_cv2(self.camera1_image, desired_encoding='bgr8')
        img_2_world = self.bridge.imgmsg_to_cv2(self.camera2_image, desired_encoding='bgr8')
        img_2_robot = img_2_world.copy()
        img_3 = self.bridge.imgmsg_to_cv2(self.camera3_image, desired_encoding='bgr8')

        cam_k_1 = np.array(self.camera1_info.k).reshape(3, 3)
        cam_d_1 = np.array(self.camera1_info.d)
        cam_k_2 = np.array(self.camera2_info.k).reshape(3, 3)
        cam_d_2 = np.array(self.camera2_info.d)
        cam_k_3 = np.array(self.camera3_info.k).reshape(3, 3)
        cam_d_3 = np.array(self.camera3_info.d)

        world_result_cam2 = self.world_aruco_detector.run(cam_k_2, cam_d_2, img_2_world)
        robot_result_cam1 = self.robot_aruco_detector.run(cam_k_1, cam_d_1, img_1)
        robot_result_cam3 = self.robot_aruco_detector.run(cam_k_3, cam_d_3, img_3)
        robot_result_cam2 = self.robot_aruco_detector.run(cam_k_2, cam_d_2, img_2_robot)


        if world_result_cam2 is None:
                cv2.putText(img_2_world, "World ArUco Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                return
        
        t_world_cam2, R_world_cam2 = world_result_cam2
        H_cam2_to_world = self._conver_tR_to_H(t_world_cam2, R_world_cam2)  # cam2 -> world
        H_world_to_cam2 = self.inverse_homogeneous_matrix(H_cam2_to_world)  # world -> cam2
        
        # 사용 가능한 결과들만 저장
        available_results = {}
        H_world_to_robot_results = {}

        # 각 카메라별로 결과가 있는지 확인하고 계산
        if robot_result_cam1 is not None:
            t_robot_cam1, R_robot_cam1 = robot_result_cam1
            H_cam1_to_robot = self._conver_tR_to_H(t_robot_cam1, R_robot_cam1)  # cam1 -> robot
            H_world_to_robot_results['cam1'] = H_world_to_cam2 @ self.H_cam2_to_cam1 @ H_cam1_to_robot
            
        if robot_result_cam2 is not None:
            t_robot_cam2, R_robot_cam2 = robot_result_cam2
            H_cam2_to_robot = self._conver_tR_to_H(t_robot_cam2, R_robot_cam2)  # cam2 -> robot
            H_world_to_robot_results['cam2'] = H_world_to_cam2 @ H_cam2_to_robot
            
        if robot_result_cam3 is not None:
            t_robot_cam3, R_robot_cam3 = robot_result_cam3
            H_cam3_to_robot = self._conver_tR_to_H(t_robot_cam3, R_robot_cam3)  # cam3 -> robot
            H_world_to_robot_results['cam3'] = H_world_to_cam2 @ self.H_cam2_to_cam3 @ H_cam3_to_robot

        # 결과 비교 (2개 이상 있을 때만)
        if len(H_world_to_robot_results) >= 2:
            self.compare_transformations(H_world_to_robot_results)

        self.display_images(img_1, img_2_world, img_2_robot, img_3, 
                       robot_result_cam1, robot_result_cam2, robot_result_cam3, 
                       world_result_cam2, H_world_to_robot_results)
        
        # if robot_result_cam1 is None or robot_result_cam3 is None or robot_result_cam2 is None:
            
        #     if robot_result_cam1 is None:
        #         cv2.putText(img_1, "Robot ArUco Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #     if robot_result_cam3 is None:
        #         cv2.putText(img_3, "Robot ArUco Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        #     if robot_result_cam2 is None:
        #         cv2.putText(img_2_robot, "Robot ArUco Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # else:
        #     t_world_cam2, R_world_cam2 = world_result_cam2
        #     t_robot_cam1, R_robot_cam1 = robot_result_cam1
        #     t_robot_cam3, R_robot_cam3 = robot_result_cam3
        #     t_robot_cam2, R_robot_cam2 = robot_result_cam2

        #     H_cam2_to_world = self._conver_tR_to_H(t_world_cam2, R_world_cam2)  # cam2 -> world
        #     H_world_to_cam2 = self.inverse_homogeneous_matrix(H_cam2_to_world)  # world -> cam2

        #     H_cam1_to_robot = self._conver_tR_to_H(t_robot_cam1, R_robot_cam1)
        #     H_cam3_to_robot = self._conver_tR_to_H(t_robot_cam3, R_robot_cam3)
        #     H_cam2_to_robot = self._conver_tR_to_H(t_robot_cam2, R_robot_cam2)

        #     H_cam1_to_cam2, baseline_12, ts_12 = self.load_transformation_matrix('stereo_calibration_results/H_camera1_camera2_current.yaml')
        #     H_cam2_to_cam1 = self.inverse_homogeneous_matrix(H_cam1_to_cam2)
        #     H_cam2_to_cam3, baseline_23, ts_23 = self.load_transformation_matrix('stereo_calibration_results/H_camera2_camera3_current.yaml')

        #     H_world_to_robot_from_cam1 = H_world_to_cam2 @ H_cam2_to_cam1 @ H_cam1_to_robot
        #     H_world_to_robot_from_cam3 = H_world_to_cam2 @ H_cam2_to_cam3 @ H_cam3_to_robot
        #     H_world_to_robot_from_cam2 = H_world_to_cam2 @ H_cam2_to_robot

            # self.get_logger().info(f"World to Robot from Cam1: {H_world_to_robot_from_cam1}")
            # self.get_logger().info(f"World to Robot from Cam3: {H_world_to_robot_from_cam3}")
            # self.get_logger().info(f"World to Robot from Cam2: {H_world_to_robot_from_cam2}")            

    def _conver_tR_to_H(self, t, R):
        H = np.eye(4)
        H[0:3, 0:3] = R
        H[0:3, 3] = t.ravel()
        return H
    
    def inverse_homogeneous_matrix(self, H):
        """동차 변환 행렬의 효율적이고 안정적인 역행렬 계산"""
        R = H[:3, :3]
        t = H[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        H_inv = np.eye(4)
        H_inv[:3, :3] = R_inv
        H_inv[:3, 3] = t_inv
        return H_inv
    
    def load_transformation_matrix(self, yaml_file):
        """YAML 파일에서 변환 행렬 로드"""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        H = np.array(data['stereo_transformation']['transformation_matrix'])
        baseline = data['stereo_transformation']['baseline_distance_m']
        timestamp = data['stereo_transformation']['timestamp']
        
        return H, baseline, timestamp
    
    def compare_transformations(self, results):
        """사용 가능한 변환 행렬들 비교"""
        cam_names = list(results.keys())
        
        for i, cam1 in enumerate(cam_names):
            for cam2 in cam_names[i+1:]:
                H1 = results[cam1]
                H2 = results[cam2]
                
                # 위치 차이
                pos_diff = np.linalg.norm(H1[:3, 3] - H2[:3, 3]) * 1000  # mm
                
                # 회전 차이 (간단한 방법)
                R_diff = H1[:3, :3] @ H2[:3, :3].T
                rot_diff = np.degrees(np.arccos((np.trace(R_diff) - 1) / 2))
                
                self.get_logger().info(f"{cam1} vs {cam2}: 위치차이 {pos_diff:.1f}mm, 회전차이 {rot_diff:.2f}도")
    
    def display_images(self, img_1, img_2_world, img_2_robot, img_3, 
                  robot_result_cam1, robot_result_cam2, robot_result_cam3, 
                  world_result_cam2, H_world_to_robot_results):
                
                """이미지들과 상태 정보 표시"""
                
                    # 이미지 복사 (원본 보존)
                display_img_1 = img_1.copy()
                display_img_2_world = img_2_world.copy()
                display_img_2_robot = img_2_robot.copy()
                display_img_3 = img_3.copy()
                
                # 카메라별 상태 표시
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Camera 1 상태
                if robot_result_cam1 is not None:
                    cv2.putText(display_img_1, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
                    if 'cam1' in H_world_to_robot_results:
                        pos = H_world_to_robot_results['cam1'][:3, 3]
                        cv2.putText(display_img_1, f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", 
                                (10, 60), font, font_scale, (0, 255, 0), thickness)
                else:
                    cv2.putText(display_img_1, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
                
                # Camera 2 World 상태
                if world_result_cam2 is not None:
                    cv2.putText(display_img_2_world, "World ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
                else:
                    cv2.putText(display_img_2_world, "World ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
                
                # ✅ 저장된 점수만 표시
                cv2.putText(display_img_2_world, f"Saved Points: {len([x for x in self.x_pose if x is not None])}", 
                        (10, 60), font, 0.6, (255, 255, 255), 2)
                cv2.putText(display_img_2_world, "Press SPACE to capture position", 
                        (10, 80), font, 0.5, (200, 200, 200), 2)
                
                # Camera 2 Robot 상태
                if robot_result_cam2 is not None:
                    cv2.putText(display_img_2_robot, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
                    if 'cam2' in H_world_to_robot_results:
                        pos = H_world_to_robot_results['cam2'][:3, 3]
                        cv2.putText(display_img_2_robot, f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", 
                                (10, 60), font, font_scale, (0, 255, 0), thickness)
                else:
                    cv2.putText(display_img_2_robot, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
                
                # Camera 3 상태
                if robot_result_cam3 is not None:
                    cv2.putText(display_img_3, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
                    if 'cam3' in H_world_to_robot_results:
                        pos = H_world_to_robot_results['cam3'][:3, 3]
                        cv2.putText(display_img_3, f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", 
                                (10, 60), font, font_scale, (0, 255, 0), thickness)
                else:
                    cv2.putText(display_img_3, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
                
                # ✅ 현재 위치 실시간 표시 (저장되지 않음)
                x_positions = []
                y_positions = []
                
                for cam_name, H_matrix in H_world_to_robot_results.items():
                    if H_matrix is not None:
                        x_positions.append(H_matrix[0, 3])
                        y_positions.append(H_matrix[1, 3])
                
                # 현재 위치 표시 (Camera 3에)
                if len(x_positions) > 0:
                    current_x = np.mean(x_positions)
                    current_y = np.mean(y_positions)
                    cv2.putText(display_img_3, f"Current: [{current_x:.3f}, {current_y:.3f}]", 
                            (10, 90), font, 0.5, (255, 255, 0), 2)
                
                # 이미지 크기 조정 (화면에 맞게)
                scale = 0.6
                height, width = display_img_1.shape[:2]
                new_width, new_height = int(width * scale), int(height * scale)
                
                display_img_1 = cv2.resize(display_img_1, (new_width, new_height))
                display_img_2_world = cv2.resize(display_img_2_world, (new_width, new_height))
                display_img_2_robot = cv2.resize(display_img_2_robot, (new_width, new_height))
                display_img_3 = cv2.resize(display_img_3, (new_width, new_height))
                
                # 이미지 표시
                cv2.imshow('Camera 1 (Robot Detection)', display_img_1)
                cv2.imshow('Camera 2 (World Detection)', display_img_2_world)
                cv2.imshow('Camera 2 (Robot Detection)', display_img_2_robot)
                cv2.imshow('Camera 3 (Robot Detection)', display_img_3)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("종료 요청됨")
                    self.plot_trajectory()  # 그래프 그리기
                    cv2.destroyAllWindows()
                    rclpy.shutdown()
                elif key == ord('s'):
                    # 스크린샷 저장
                    timestamp = self.get_clock().now().to_msg()
                    cv2.imwrite(f'cam1_{timestamp.sec}.jpg', display_img_1)
                    cv2.imwrite(f'cam2_world_{timestamp.sec}.jpg', display_img_2_world)
                    cv2.imwrite(f'cam2_robot_{timestamp.sec}.jpg', display_img_2_robot)
                    cv2.imwrite(f'cam3_{timestamp.sec}.jpg', display_img_3)
                    self.get_logger().info("스크린샷 저장됨")
                elif key == ord('p'):  # 'p' 키로 중간에도 그래프 보기
                    self.plot_trajectory(show_only=True)
                elif key == ord(' '):  # ✅ 스페이스바로 위치 캡처 (한 번에 한 번만)
                    if len(x_positions) > 0:
                        self.x_pose.append(np.mean(x_positions))
                        self.y_pose.append(np.mean(y_positions))
                        self.get_logger().info(f"위치 캡처됨: [{np.mean(x_positions):.3f}, {np.mean(y_positions):.3f}] (총 {len([x for x in self.x_pose if x is not None])}개)")
                    else:
                        self.get_logger().info("캡처할 유효한 위치 데이터가 없습니다")
                elif key == ord('r'):  # ✅ 'r' 키로 데이터 초기화
                    point_count = len([x for x in self.x_pose if x is not None])
                    self.x_pose.clear()
                    self.y_pose.clear()
                    self.timestamps.clear()
                    self.get_logger().info(f"저장된 {point_count}개 위치 데이터 초기화됨")
                
                # ✅ 자동 캡처 코드 제거 - 오직 수동 캡처만

    def plot_trajectory(self, show_only=False):
        """로봇의 trajectory 그래프 그리기"""
        
        if len(self.x_pose) == 0:
            self.get_logger().info("저장된 위치 데이터가 없습니다.")
            return
        
        # None 값 제거하고 유효한 데이터만 추출
        valid_indices = []
        valid_x = []
        valid_y = []
        
        for i, (x, y) in enumerate(zip(self.x_pose, self.y_pose)):
            if x is not None and y is not None:
                valid_indices.append(i)
                valid_x.append(x)
                valid_y.append(y)
        
        if len(valid_x) == 0:
            self.get_logger().info("유효한 위치 데이터가 없습니다.")
            return
        
        # 그래프 생성
        # plt.figure(figsize=(15, 10))
        
        # 그래프 생성
        plt.figure(figsize=(15, 10))
    
        # 1. 2D Trajectory (X-Y 평면)
        plt.subplot(2, 2, 1)
        plt.plot(valid_x, valid_y, 'b-', linewidth=2, alpha=0.7, label='Robot Path')
        plt.scatter(valid_x, valid_y, c=range(len(valid_x)), cmap='viridis', s=20, alpha=0.8)
        
        # 시작점과 끝점 표시
        if len(valid_x) > 0:
            plt.scatter(valid_x[0], valid_y[0], c='green', s=100, marker='o', label='Start')
            plt.scatter(valid_x[-1], valid_y[-1], c='red', s=100, marker='x', label='End')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot 2D Trajectory (Top View)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Time Progress')
        
        # 2. X Position over Time
        plt.subplot(2, 2, 2)
        plt.plot(valid_indices, valid_x, 'r-', linewidth=2, label='X Position')
        plt.scatter(valid_indices, valid_x, c='red', s=20, alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('X Position (m)')
        plt.title('X Position over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Y Position over Time
        plt.subplot(2, 2, 3)
        plt.plot(valid_indices, valid_y, 'g-', linewidth=2, label='Y Position')
        plt.scatter(valid_indices, valid_y, c='green', s=20, alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. 통계 정보
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # 통계 계산
        total_points = len(self.x_pose)
        valid_points = len(valid_x)
        detection_rate = (valid_points / total_points) * 100 if total_points > 0 else 0
        
        if len(valid_x) > 1:
            # 이동 거리 계산
            distances = []
            for i in range(1, len(valid_x)):
                dist = np.sqrt((valid_x[i] - valid_x[i-1])**2 + (valid_y[i] - valid_y[i-1])**2)
                distances.append(dist)
            total_distance = sum(distances)
            avg_speed = np.mean(distances) if distances else 0
            max_speed = max(distances) if distances else 0
        else:
            total_distance = 0
            avg_speed = 0
            max_speed = 0
        
        # 통계 텍스트
        stats_text = f"""
        📊 Trajectory Statistics
        
        📍 Total Data Points: {total_points}
        ✅ Valid Detections: {valid_points} ({detection_rate:.1f}%)
        ❌ Lost Tracking: {total_points - valid_points}
        
        📏 Total Distance: {total_distance:.3f} m
        🚀 Average Step Size: {avg_speed:.4f} m
        ⚡ Max Step Size: {max_speed:.4f} m
        
        📍 Position Range:
        X: [{min(valid_x):.3f}, {max(valid_x):.3f}] m
        Y: [{min(valid_y):.3f}, {max(valid_y):.3f}] m
        
        📍 Final Position:
        X: {valid_x[-1]:.3f} m
        Y: {valid_y[-1]:.3f} m
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 파일 저장 (종료 시에만)
        if not show_only:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'robot_trajectory_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.get_logger().info(f"Trajectory 저장됨: {filename}")
            
            # 데이터도 CSV로 저장
            self.save_trajectory_data(timestamp)
        
        plt.show()
    
    def save_trajectory_data(self, timestamp):
        """trajectory 데이터를 CSV로 저장"""
        try:
            import pandas as pd
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                'step': range(len(self.x_pose)),
                'x_position': self.x_pose,
                'y_position': self.y_pose
            })
            
            filename = f'robot_trajectory_data_{timestamp}.csv'
            df.to_csv(filename, index=False)
            self.get_logger().info(f"Trajectory 데이터 저장됨: {filename}")
            
        except ImportError:
            # pandas가 없으면 기본 CSV로 저장
            filename = f'robot_trajectory_data_{timestamp}.csv'
            with open(filename, 'w') as f:
                f.write("step,x_position,y_position\n")
                for i, (x, y) in enumerate(zip(self.x_pose, self.y_pose)):
                    f.write(f"{i},{x},{y}\n")
            self.get_logger().info(f"Trajectory 데이터 저장됨: {filename}")

    def __del__(self):
        """소멸자에서도 그래프 그리기 (안전장치)"""
        if hasattr(self, 'x_pose') and len(self.x_pose) > 0:
            self.plot_trajectory()

    def main(args=None):
        rclpy.init(args=args)
        world_robot_node = World_Robot()
        
        try:
            rclpy.spin(world_robot_node)
        except KeyboardInterrupt:
            print("\n프로그램 종료 중...")
        finally:
            # 종료 시 그래프 그리기
            world_robot_node.plot_trajectory()
            world_robot_node.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    World_Robot.main()