import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String, Bool  # ✅ 추가
from cv_bridge import CvBridge
import os
from multicam_aruco import ARUCOBoardPose, ARUCORobotPose
import cv2
import matplotlib.pyplot as plt
import time
import threading
import pandas as pd
from datetime import datetime

class World_Robot(Node):
    def __init__(self):
        super().__init__('world_robot')
        self.bridge = CvBridge()
        self.world_aruco_detector = ARUCOBoardPose()
        self.robot_aruco_detector = ARUCORobotPose()

        # 기존 구독자들
        self.camera_1_sub = self.create_subscription(
            Image, '/camera1/camera1/color/image_raw', self.image_1_callback, 10)
        self.camera_2_sub = self.create_subscription(
            Image, '/camera2/camera2/color/image_raw', self.image_2_callback, 10)
        self.camera_3_sub = self.create_subscription(
            Image, '/camera3/camera3/color/image_raw', self.image_3_callback, 10)
        self.camera_info_1 = self.create_subscription(
            CameraInfo, '/camera1/camera1/color/camera_info', self.camera1_info_callback, 1)
        self.camera_info_2 = self.create_subscription(
            CameraInfo, '/camera2/camera2/color/camera_info', self.camera2_info_callback, 1)
        self.camera_info_3 = self.create_subscription(
            CameraInfo, '/camera3/camera3/color/camera_info', self.camera3_info_callback, 1)
        
        # ✅ 로봇 자동 제어 관련 구독자/발행자
        self.robot_ready_sub = self.create_subscription(
            Bool, '/robot_auto/robot_ready', self.robot_ready_callback, 10)
        self.robot_status_sub = self.create_subscription(
            String, '/robot_auto/status', self.robot_status_callback, 10)
        
        # ✅ 데이터 수집 완료 신호 발행자
        self.data_collected_pub = self.create_publisher(Bool, '/robot_auto/data_collected', 10)
        
        self.create_timer(0.1, self.loop)

        # 데이터 구조
        self.trajectory_data = []
        self.collecting_data = False
        self.collection_thread = None

        # 기존 변수들
        self.camera1_info = None
        self.camera2_info = None
        self.camera3_info = None
        self.camera1_image = None
        self.camera2_image = None
        self.camera3_image = None
        self.H_cam1_to_cam2 = None
        self.H_cam2_to_cam1 = None
        self.H_cam2_to_cam3 = None
        self.current_frame_data = {}
        
        # ✅ 자동 모드 관련 변수
        self.auto_mode = False
        self.robot_ready = False
        
        self.get_logger().info("📊 자동 데이터 수집기 초기화 완료")
        self.get_logger().info("   - 로봇 준비 신호 대기 중: /robot_auto/robot_ready")
        
    def image_1_callback(self, msg: Image):
        self.camera1_image = msg
    def image_2_callback(self, msg: Image):
        self.camera2_image = msg
    def image_3_callback(self, msg: Image):
        self.camera3_image = msg
    def camera1_info_callback(self, msg: CameraInfo):
        if self.camera1_info is None:
            self.camera1_info = msg
    def camera2_info_callback(self, msg: CameraInfo):
        if self.camera2_info is None:
            self.camera2_info = msg
    def camera3_info_callback(self, msg: CameraInfo):
        if self.camera3_info is None:
            self.camera3_info = msg

    # ✅ 로봇 컨트롤러와의 통신 콜백
    def robot_ready_callback(self, msg: Bool):
        """로봇이 데이터 수집 준비 완료되었다는 신호 받음"""
        if msg.data and not self.collecting_data:
            self.get_logger().info("🤖 로봇 준비 신호 받음 - 자동 데이터 수집 시작")
            self.robot_ready = True
            self.start_auto_data_collection()
        elif msg.data and self.collecting_data:
            self.get_logger().info("⚠️  이미 데이터 수집 중입니다...")
    
    def robot_status_callback(self, msg: String):
        """로봇 상태 정보 받음"""
        status = msg.data
        if status == "auto_started":
            self.auto_mode = True
            self.get_logger().info("🚀 자동 모드 활성화됨")
        elif status == "auto_stopped":
            self.auto_mode = False
            self.get_logger().info("⏹️  자동 모드 비활성화됨")
        elif status.startswith("ready_for_data_"):
            position_num = status.split("_")[-1]
            self.get_logger().info(f"📍 로봇 위치 {position_num} 준비 완료")

    def start_auto_data_collection(self):
        """자동 데이터 수집 시작"""
        if self.collecting_data:
            self.get_logger().warn("⚠️  이미 데이터 수집 중입니다.")
            return
        
        # 데이터 수집 가능 여부 확인
        if self.current_frame_data is None or len(self.current_frame_data) == 0:
            self.get_logger().warn("⚠️  수집할 데이터가 없습니다 - 로봇에게 재시도 신호 전송")
            # 3초 후 재시도 신호 전송
            self.create_timer(3.0, self.send_retry_signal, single_shot=True)
            return
        
        # 별도 스레드에서 데이터 수집
        self.collection_thread = threading.Thread(target=self.collect_data_sequence_auto)
        self.collection_thread.start()
    
    def send_retry_signal(self):
        """재시도 신호 전송 (데이터가 없을 때)"""
        retry_msg = Bool()
        retry_msg.data = False  # False로 보내서 로봇이 재시도하도록
        self.data_collected_pub.publish(retry_msg)
        self.get_logger().info("🔄 재시도 신호 전송됨")

    def collect_data_sequence_auto(self):
        """자동 모드용 데이터 수집 (0.5초 간격으로 5개)"""
        self.collecting_data = True
        collected_samples = []
        
        self.get_logger().info("📊 자동 데이터 수집 시작... (5개 샘플, 0.5초 간격)")
        
        for i in range(5):
            # 현재 프레임 데이터 수집
            if self.current_frame_data:
                sample_data = {}
                
                for cam_name, cam_data in self.current_frame_data.items():
                    transform = cam_data['transform']
                    sample_data[cam_name] = {
                        'x': transform[0, 3],
                        'y': transform[1, 3],
                        'z': transform[2, 3],
                        'num_markers': cam_data['num_markers']
                    }
                
                collected_samples.append(sample_data)
                self.get_logger().info(f"  샘플 {i+1}/5 수집됨 (카메라 {len(sample_data)}개)")
            else:
                self.get_logger().warn(f"  샘플 {i+1}/5 수집 실패 - 데이터 없음")
                collected_samples.append({})
            
            if i < 4:  # 마지막 샘플 후에는 대기하지 않음
                time.sleep(0.5)
        
        # 수집된 데이터 분석 및 저장
        self.analyze_and_store_samples_auto(collected_samples)
        self.collecting_data = False
        
        # ✅ 로봇에게 데이터 수집 완료 신호 전송
        self.send_data_collection_complete()

    def send_data_collection_complete(self):
        """데이터 수집 완료 신호를 로봇에게 전송"""
        complete_msg = Bool()
        complete_msg.data = True
        self.data_collected_pub.publish(complete_msg)
        
        self.get_logger().info("✅ 데이터 수집 완료 - 로봇에게 신호 전송됨")
        
    def analyze_and_store_samples_auto(self, samples):
        """자동 모드용 샘플 분석 및 저장"""
        
        # 각 카메라별로 데이터 분리
        camera_data = {'cam1': [], 'cam2': [], 'cam3': []}
        
        for sample in samples:
            for cam_name in ['cam1', 'cam2', 'cam3']:
                if cam_name in sample:
                    camera_data[cam_name].append(sample[cam_name])
        
        # 각 카메라별 통계 계산
        camera_stats = {}
        all_x_coords = []
        all_y_coords = []
        
        for cam_name, cam_samples in camera_data.items():
            if len(cam_samples) > 0:
                x_coords = [s['x'] for s in cam_samples]
                y_coords = [s['y'] for s in cam_samples]
                z_coords = [s['z'] for s in cam_samples]
                marker_counts = [s['num_markers'] for s in cam_samples]
                
                camera_stats[cam_name] = {
                    'sample_count': len(cam_samples),
                    'x_mean': np.mean(x_coords),
                    'x_std': np.std(x_coords) if len(x_coords) > 1 else 0.0,
                    'y_mean': np.mean(y_coords),
                    'y_std': np.std(y_coords) if len(y_coords) > 1 else 0.0,
                    'z_mean': np.mean(z_coords),
                    'z_std': np.std(z_coords) if len(z_coords) > 1 else 0.0,
                    'avg_markers': np.mean(marker_counts),
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'z_coords': z_coords,
                    'marker_counts': marker_counts
                }
                
                # 전체 평균 계산용 데이터 수집
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
        
        # 전체 평균 계산
        if len(all_x_coords) > 0:
            overall_x_mean = np.mean(all_x_coords)
            overall_y_mean = np.mean(all_y_coords)
            overall_x_std = np.std(all_x_coords) if len(all_x_coords) > 1 else 0.0
            overall_y_std = np.std(all_y_coords) if len(all_y_coords) > 1 else 0.0
        else:
            overall_x_mean = overall_y_mean = overall_x_std = overall_y_std = 0.0
        
        # 최종 데이터 저장
        position_data = {
            'timestamp': time.time(),
            'position_id': len(self.trajectory_data) + 1,
            'overall_mean': {
                'x': overall_x_mean,
                'y': overall_y_mean,
                'x_std': overall_x_std,
                'y_std': overall_y_std
            },
            'camera_stats': camera_stats,
            'total_samples': len([s for s in samples if s]),
            'participating_cameras': list(camera_stats.keys())
        }
        
        self.trajectory_data.append(position_data)
        
        # 로그 출력
        self.get_logger().info(f"✅ 위치 {position_data['position_id']} 데이터 저장:")
        self.get_logger().info(f"   전체 평균: X={overall_x_mean:.4f}±{overall_x_std:.4f}m, Y={overall_y_mean:.4f}±{overall_y_std:.4f}m")
        self.get_logger().info(f"   참여 카메라: {len(camera_stats)}개 ({', '.join(camera_stats.keys())})")

    # ✅ 기존 수동 데이터 수집도 유지
    def collect_data_sequence(self):
        """수동 모드용 데이터 수집 (스페이스바 입력 시)"""
        self.collecting_data = True
        collected_samples = []
        
        self.get_logger().info("📊 수동 데이터 수집 시작... (5개 샘플, 0.5초 간격)")
        
        for i in range(5):
            if self.current_frame_data:
                sample_data = {}
                
                for cam_name, cam_data in self.current_frame_data.items():
                    transform = cam_data['transform']
                    sample_data[cam_name] = {
                        'x': transform[0, 3],
                        'y': transform[1, 3],
                        'z': transform[2, 3],
                        'num_markers': cam_data['num_markers']
                    }
                
                collected_samples.append(sample_data)
                self.get_logger().info(f"  샘플 {i+1}/5 수집됨 (카메라 {len(sample_data)}개)")
            else:
                self.get_logger().warn(f"  샘플 {i+1}/5 수집 실패 - 데이터 없음")
                collected_samples.append({})
            
            if i < 4:
                time.sleep(0.5)
        
        self.analyze_and_store_samples(collected_samples)
        self.collecting_data = False

    def analyze_and_store_samples(self, samples):
        """기존 수동 모드용 샘플 분석 (기존 코드 유지)"""
        # ... 기존 analyze_and_store_samples 코드와 동일 ...
        
        # 각 카메라별로 데이터 분리
        camera_data = {'cam1': [], 'cam2': [], 'cam3': []}
        
        for sample in samples:
            for cam_name in ['cam1', 'cam2', 'cam3']:
                if cam_name in sample:
                    camera_data[cam_name].append(sample[cam_name])
        
        # 각 카메라별 통계 계산
        camera_stats = {}
        all_x_coords = []
        all_y_coords = []
        
        for cam_name, cam_samples in camera_data.items():
            if len(cam_samples) > 0:
                x_coords = [s['x'] for s in cam_samples]
                y_coords = [s['y'] for s in cam_samples]
                z_coords = [s['z'] for s in cam_samples]
                marker_counts = [s['num_markers'] for s in cam_samples]
                
                camera_stats[cam_name] = {
                    'sample_count': len(cam_samples),
                    'x_mean': np.mean(x_coords),
                    'x_std': np.std(x_coords) if len(x_coords) > 1 else 0.0,
                    'y_mean': np.mean(y_coords),
                    'y_std': np.std(y_coords) if len(y_coords) > 1 else 0.0,
                    'z_mean': np.mean(z_coords),
                    'z_std': np.std(z_coords) if len(z_coords) > 1 else 0.0,
                    'avg_markers': np.mean(marker_counts),
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'z_coords': z_coords,
                    'marker_counts': marker_counts
                }
                
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
        
        # 전체 평균 계산
        if len(all_x_coords) > 0:
            overall_x_mean = np.mean(all_x_coords)
            overall_y_mean = np.mean(all_y_coords)
            overall_x_std = np.std(all_x_coords) if len(all_x_coords) > 1 else 0.0
            overall_y_std = np.std(all_y_coords) if len(all_y_coords) > 1 else 0.0
        else:
            overall_x_mean = overall_y_mean = overall_x_std = overall_y_std = 0.0
        
        # 최종 데이터 저장
        position_data = {
            'timestamp': time.time(),
            'position_id': len(self.trajectory_data) + 1,
            'overall_mean': {
                'x': overall_x_mean,
                'y': overall_y_mean,
                'x_std': overall_x_std,
                'y_std': overall_y_std
            },
            'camera_stats': camera_stats,
            'total_samples': len([s for s in samples if s]),
            'participating_cameras': list(camera_stats.keys())
        }
        
        self.trajectory_data.append(position_data)
        
        # 로그 출력
        self.get_logger().info(f"✅ 위치 데이터 저장 완료:")
        self.get_logger().info(f"   전체 평균: X={overall_x_mean:.4f}±{overall_x_std:.4f}m, Y={overall_y_mean:.4f}±{overall_y_std:.4f}m")
        self.get_logger().info(f"   참여 카메라: {len(camera_stats)}개 ({', '.join(camera_stats.keys())})")

    def loop(self):
        # ... 기존 loop 코드는 그대로 유지 ...
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
        H_cam2_to_world = self._conver_tR_to_H(t_world_cam2, R_world_cam2)
        H_world_to_cam2 = self.inverse_homogeneous_matrix(H_cam2_to_world)
        
        H_world_to_robot_results = {}
        
        if robot_result_cam1 is not None:
            t_robot_cam1, R_robot_cam1, num_markers_cam1 = robot_result_cam1
            H_cam1_to_robot = self._conver_tR_to_H(t_robot_cam1, R_robot_cam1)
            H_world_to_robot_results['cam1'] = {
                'transform': H_world_to_cam2 @ self.H_cam2_to_cam1 @ H_cam1_to_robot,
                'num_markers': num_markers_cam1
            }
            
        if robot_result_cam2 is not None:
            t_robot_cam2, R_robot_cam2, num_markers_cam2 = robot_result_cam2
            H_cam2_to_robot = self._conver_tR_to_H(t_robot_cam2, R_robot_cam2)
            H_world_to_robot_results['cam2'] = {
                'transform': H_world_to_cam2 @ H_cam2_to_robot,
                'num_markers': num_markers_cam2
            }
            
        if robot_result_cam3 is not None:
            t_robot_cam3, R_robot_cam3, num_markers_cam3 = robot_result_cam3
            H_cam3_to_robot = self._conver_tR_to_H(t_robot_cam3, R_robot_cam3)
            H_world_to_robot_results['cam3'] = {
                'transform': H_world_to_cam2 @ self.H_cam2_to_cam3 @ H_cam3_to_robot,
                'num_markers': num_markers_cam3
            }

        self.current_frame_data = H_world_to_robot_results.copy()

        self.display_images(img_1, img_2_world, img_2_robot, img_3, 
                       robot_result_cam1, robot_result_cam2, robot_result_cam3, 
                       world_result_cam2, H_world_to_robot_results)

    def display_images(self, img_1, img_2_world, img_2_robot, img_3, 
              robot_result_cam1, robot_result_cam2, robot_result_cam3, 
              world_result_cam2, H_world_to_robot_results):
        """이미지들과 상태 정보 표시 (ArUco 보드 중심점과 축 표시 추가)"""
        
        display_img_1 = img_1.copy()
        display_img_2_world = img_2_world.copy()
        display_img_2_robot = img_2_robot.copy()
        display_img_3 = img_3.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # ✅ ArUco 보드 중심점과 축 시각화 함수 (수정)
        def draw_board_center_and_axes(img, aruco_result, camera_k, camera_d, label=""):
            """ArUco 보드의 중심점과 좌표축을 이미지에 그리기"""
            if aruco_result is None:
                return
            
            # ✅ 결과 형태에 따라 처리 (world는 2개, robot은 3개 반환)
            if len(aruco_result) == 2:
                t, R = aruco_result  # World ArUco 결과
            elif len(aruco_result) == 3:
                t, R, num_markers = aruco_result  # Robot ArUco 결과
            else:
                return
            
            rvec = cv2.Rodrigues(R)[0]
            
            # ✅ 1. 좌표축 그리기 (기존)
            cv2.drawFrameAxes(img, camera_k, camera_d, rvec, t, 0.1, 3)
            
            # ✅ 2. 보드 중심점 투영 및 표시
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)  # 보드 중심점 (0,0,0)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, t, camera_k, camera_d)
            center_2d = center_2d.reshape(-1, 2).astype(int)
            
            # 중심점에 큰 원 그리기
            cv2.circle(img, tuple(center_2d[0]), 8, (0, 255, 255), -1)  # 노란색 원
            cv2.circle(img, tuple(center_2d[0]), 12, (0, 0, 0), 2)     # 검정 테두리
            
            # ✅ 4. 보드 중심점에 라벨 표시
            cv2.putText(img, f"{label} CENTER", 
                (center_2d[0][0] + 15, center_2d[0][1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # ✅ 5. 개별 ArUco 마커들의 중심점도 표시 (선택적)
            # 5x7 격자의 각 마커 중심점 계산
            marker_size = 0.06
            marker_separation = 0.005
            start_x = -2 * (marker_size + marker_separation)  # 5개 마커의 중심 기준
            start_y = -3 * (marker_size + marker_separation)  # 7개 마커의 중심 기준
            
            for i in range(7):  # 세로 7개
                for j in range(5):  # 가로 5개
                    marker_x = start_x + j * (marker_size + marker_separation)
                    marker_y = start_y + i * (marker_size + marker_separation)
                    
                    marker_center_3d = np.array([[marker_x, marker_y, 0]], dtype=np.float32)
                    marker_center_2d, _ = cv2.projectPoints(marker_center_3d, rvec, t, camera_k, camera_d)
                    marker_center_2d = marker_center_2d.reshape(-1, 2).astype(int)
                    
                    # 작은 점으로 각 마커 중심 표시
                    cv2.circle(img, tuple(marker_center_2d[0]), 2, (100, 100, 100), -1)
        
        # ✅ Camera 1 - Robot ArUco 시각화
        if robot_result_cam1 is not None:
            cv2.putText(display_img_1, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            
            # 보드 중심점과 축 그리기
            cam_k_1 = np.array(self.camera1_info.k).reshape(3, 3)
            cam_d_1 = np.array(self.camera1_info.d)
            draw_board_center_and_axes(display_img_1, robot_result_cam1, cam_k_1, cam_d_1, "ROBOT")
            
            if 'cam1' in H_world_to_robot_results:
                pos = H_world_to_robot_results['cam1']['transform'][:3, 3]
                markers = H_world_to_robot_results['cam1']['num_markers']
                cv2.putText(display_img_1, f"World Pos: [{pos[0]:.3f}, {pos[1]:.3f}]", 
                        (10, 60), font, 0.5, (0, 255, 0), 2)
                cv2.putText(display_img_1, f"Markers: {markers}", 
                        (10, 80), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_img_1, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
        
        # ✅ Camera 2 - World ArUco 시각화
        if world_result_cam2 is not None:
            cv2.putText(display_img_2_world, "World ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            
            # 보드 중심점과 축 그리기
            cam_k_2 = np.array(self.camera2_info.k).reshape(3, 3)
            cam_d_2 = np.array(self.camera2_info.d)
            draw_board_center_and_axes(display_img_2_world, world_result_cam2, cam_k_2, cam_d_2, "WORLD")
        else:
            cv2.putText(display_img_2_world, "World ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
        
        # ✅ Camera 2 - Robot ArUco 시각화
        if robot_result_cam2 is not None:
            cv2.putText(display_img_2_robot, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            
            # 보드 중심점과 축 그리기
            cam_k_2 = np.array(self.camera2_info.k).reshape(3, 3)
            cam_d_2 = np.array(self.camera2_info.d)
            draw_board_center_and_axes(display_img_2_robot, robot_result_cam2, cam_k_2, cam_d_2, "ROBOT")
            
            if 'cam2' in H_world_to_robot_results:
                pos = H_world_to_robot_results['cam2']['transform'][:3, 3]
                markers = H_world_to_robot_results['cam2']['num_markers']
                cv2.putText(display_img_2_robot, f"World Pos: [{pos[0]:.3f}, {pos[1]:.3f}]", 
                        (10, 60), font, 0.5, (0, 255, 0), 2)
                cv2.putText(display_img_2_robot, f"Markers: {markers}", 
                        (10, 80), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_img_2_robot, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
        
        # ✅ Camera 3 - Robot ArUco 시각화
        if robot_result_cam3 is not None:
            cv2.putText(display_img_3, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            
            # 보드 중심점과 축 그리기
            cam_k_3 = np.array(self.camera3_info.k).reshape(3, 3)
            cam_d_3 = np.array(self.camera3_info.d)
            draw_board_center_and_axes(display_img_3, robot_result_cam3, cam_k_3, cam_d_3, "ROBOT")
            
            if 'cam3' in H_world_to_robot_results:
                pos = H_world_to_robot_results['cam3']['transform'][:3, 3]
                markers = H_world_to_robot_results['cam3']['num_markers']
                cv2.putText(display_img_3, f"World Pos: [{pos[0]:.3f}, {pos[1]:.3f}]", 
                        (10, 60), font, 0.5, (0, 255, 0), 2)
                cv2.putText(display_img_3, f"Markers: {markers}", 
                        (10, 80), font, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_img_3, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
        
        # ✅ 자동/수동 모드 상태 표시
        cv2.putText(display_img_2_world, f"Saved Positions: {len(self.trajectory_data)}", 
                (10, 60), font, 0.6, (255, 255, 255), 2)
        
        if self.auto_mode:
            cv2.putText(display_img_2_world, "AUTO MODE", 
                    (10, 90), font, 0.7, (0, 255, 0), 2)
            if self.collecting_data:
                cv2.putText(display_img_2_world, "AUTO COLLECTING...", 
                        (10, 110), font, 0.5, (0, 255, 255), 2)
            elif self.robot_ready:
                cv2.putText(display_img_2_world, "Robot Ready - Waiting...", 
                        (10, 110), font, 0.5, (255, 255, 0), 2)
        else:
            if self.collecting_data:
                cv2.putText(display_img_2_world, "COLLECTING DATA...", 
                        (10, 90), font, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_img_2_world, "Press SPACE for manual collection", 
                        (10, 90), font, 0.5, (200, 200, 200), 2)
        
        # 현재 위치 실시간 표시
        if len(H_world_to_robot_results) > 0:
            all_x = [data['transform'][0, 3] for data in H_world_to_robot_results.values()]
            all_y = [data['transform'][1, 3] for data in H_world_to_robot_results.values()]
            current_x = np.mean(all_x)
            current_y = np.mean(all_y)
            cv2.putText(display_img_3, f"Live: [{current_x:.3f}, {current_y:.3f}]", 
                    (10, 100), font, 0.5, (255, 255, 0), 2)
            cv2.putText(display_img_3, f"Cameras: {len(H_world_to_robot_results)}/3", 
                    (10, 120), font, 0.5, (255, 255, 0), 2)
        
        # ✅ 범례 추가 (Camera 1에)
        legend_y = 140
        cv2.putText(display_img_1, "Legend:", (10, legend_y), font, 0.4, (255, 255, 255), 1)
        cv2.circle(display_img_1, (20, legend_y + 20), 8, (0, 255, 255), -1)
        cv2.putText(display_img_1, "Board Center", (35, legend_y + 25), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display_img_1, "RGB Axes: X(R) Y(G) Z(B)", (10, legend_y + 40), font, 0.3, (255, 255, 255), 1)
        cv2.putText(display_img_1, "Gray dots: Marker centers", (10, legend_y + 55), font, 0.3, (255, 255, 255), 1)
        
        # 이미지 크기 조정 및 표시
        scale = 0.6
        height, width = display_img_1.shape[:2]
        new_width, new_height = int(width * scale), int(height * scale)
        
        display_img_1 = cv2.resize(display_img_1, (new_width, new_height))
        display_img_2_world = cv2.resize(display_img_2_world, (new_width, new_height))
        display_img_2_robot = cv2.resize(display_img_2_robot, (new_width, new_height))
        display_img_3 = cv2.resize(display_img_3, (new_width, new_height))
        
        cv2.imshow('Camera 1 (Robot Detection)', display_img_1)
        cv2.imshow('Camera 2 (World Detection)', display_img_2_world)
        cv2.imshow('Camera 2 (Robot Detection)', display_img_2_robot)
        cv2.imshow('Camera 3 (Robot Detection)', display_img_3)
        
        # 키 입력 처리 (기존과 동일)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("종료 요청됨")
            self.plot_trajectory()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord('p'):
            self.plot_trajectory(show_only=True)
        elif key == ord(' '):  # 수동 데이터 수집
            if not self.collecting_data and not self.auto_mode:
                self.collection_thread = threading.Thread(target=self.collect_data_sequence)
                self.collection_thread.start()
            elif self.auto_mode:
                self.get_logger().info("⚠️  자동 모드에서는 수동 수집이 불가능합니다.")
            else:
                self.get_logger().info("⚠️  이미 데이터 수집 중입니다...")
        elif key == ord('r'):  # 데이터 초기화
            self.trajectory_data.clear()
            self.get_logger().info(f"🗑️  저장된 데이터 초기화됨")

    # ... 기존 함수들 (plot_trajectory, _conver_tR_to_H 등) 그대로 유지 ...
    def plot_trajectory(self, show_only=False):
        """개선된 trajectory 그래프 그리기"""
        
        if len(self.trajectory_data) == 0:
            self.get_logger().info("저장된 위치 데이터가 없습니다.")
            return
        
        # 전체 평균 위치 추출 (그래프용)
        x_positions = [data['overall_mean']['x'] for data in self.trajectory_data]
        y_positions = [data['overall_mean']['y'] for data in self.trajectory_data]
        x_stds = [data['overall_mean']['x_std'] for data in self.trajectory_data]
        y_stds = [data['overall_mean']['y_std'] for data in self.trajectory_data]
        
        # 그래프 생성
        plt.figure(figsize=(16, 12))
        
        # 1. 메인 2D Trajectory with Error Bars
        plt.subplot(2, 3, 1)
        plt.errorbar(x_positions, y_positions, xerr=x_stds, yerr=y_stds, 
                    fmt='o-', linewidth=2, markersize=6, alpha=0.8, capsize=3)
        plt.scatter(x_positions, y_positions, c=range(len(x_positions)), 
                   cmap='viridis', s=50, alpha=0.9, zorder=5)
        
        if len(x_positions) > 0:
            plt.scatter(x_positions[0], y_positions[0], c='green', s=150, 
                       marker='o', label='Start', zorder=10)
            plt.scatter(x_positions[-1], y_positions[-1], c='red', s=150, 
                       marker='x', label='End', zorder=10)
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Trajectory with Uncertainty')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        # 2. X Position over Time
        plt.subplot(2, 3, 2)
        indices = range(len(x_positions))
        plt.errorbar(indices, x_positions, yerr=x_stds, 
                    fmt='r-o', linewidth=2, markersize=4, capsize=3)
        plt.xlabel('Position Index')
        plt.ylabel('X Position (m)')
        plt.title('X Position with Std Dev')
        plt.grid(True, alpha=0.3)
        
        # 3. Y Position over Time
        plt.subplot(2, 3, 3)
        plt.errorbar(indices, y_positions, yerr=y_stds, 
                    fmt='g-o', linewidth=2, markersize=4, capsize=3)
        plt.xlabel('Position Index')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position with Std Dev')
        plt.grid(True, alpha=0.3)
        
        # 4. 카메라별 참여도 분석
        plt.subplot(2, 3, 4)
        camera_participation = {'cam1': 0, 'cam2': 0, 'cam3': 0}
        for data in self.trajectory_data:
            for cam in data['participating_cameras']:
                camera_participation[cam] += 1
        
        cameras = list(camera_participation.keys())
        counts = list(camera_participation.values())
        colors = ['red', 'green', 'blue']
        
        plt.bar(cameras, counts, color=colors, alpha=0.7)
        plt.xlabel('Camera')
        plt.ylabel('Participation Count')
        plt.title('Camera Participation')
        plt.grid(True, alpha=0.3)
        
        # 5. 정확도 분석 (표준편차 기반)
        plt.subplot(2, 3, 5)
        position_errors = [np.sqrt(x_std**2 + y_std**2) for x_std, y_std in zip(x_stds, y_stds)]
        plt.plot(indices, position_errors, 'purple', marker='o', linewidth=2, markersize=4)
        plt.xlabel('Position Index')
        plt.ylabel('Position Uncertainty (m)')
        plt.title('Measurement Uncertainty')
        plt.grid(True, alpha=0.3)
        
        # 6. 통계 정보
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # 상세 통계 계산
        total_positions = len(self.trajectory_data)
        avg_uncertainty = np.mean(position_errors) if position_errors else 0
        max_uncertainty = max(position_errors) if position_errors else 0
        
        # 각 카메라별 평균 마커 수
        camera_marker_stats = {}
        for cam_name in ['cam1', 'cam2', 'cam3']:
            marker_counts = []
            for data in self.trajectory_data:
                if cam_name in data['camera_stats']:
                    marker_counts.extend(data['camera_stats'][cam_name]['marker_counts'])
            camera_marker_stats[cam_name] = {
                'avg': np.mean(marker_counts) if marker_counts else 0,
                'samples': len(marker_counts)
            }
        
        stats_text = f"""
        📊 Detailed Trajectory Analysis
        
        📍 Total Positions: {total_positions}
        🎯 Average Uncertainty: {avg_uncertainty:.4f} m
        ⚠️  Max Uncertainty: {max_uncertainty:.4f} m
        
        📹 Camera Performance:
        CAM1: {camera_marker_stats['cam1']['samples']} samples, 
              {camera_marker_stats['cam1']['avg']:.1f} avg markers
        CAM2: {camera_marker_stats['cam2']['samples']} samples, 
              {camera_marker_stats['cam2']['avg']:.1f} avg markers  
        CAM3: {camera_marker_stats['cam3']['samples']} samples, 
              {camera_marker_stats['cam3']['avg']:.1f} avg markers
        
        📏 Position Range:
        X: [{min(x_positions):.3f}, {max(x_positions):.3f}] m
        Y: [{min(y_positions):.3f}, {max(y_positions):.3f}] m
        
        📍 Final Position:
        X: {x_positions[-1]:.4f} ± {x_stds[-1]:.4f} m
        Y: {y_positions[-1]:.4f} ± {y_stds[-1]:.4f} m
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 파일 저장 (종료 시에만)
        if not show_only:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'robot_trajectory_auto_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.get_logger().info(f"자동 수집 Trajectory 저장됨: {filename}")
        
        plt.show()
    
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

    def save_trajectory_to_csv(self, filename=None):
        """궤적 데이터를 CSV 파일로 저장"""
        if len(self.trajectory_data) == 0:
            self.get_logger().warn("저장할 궤적 데이터가 없습니다.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'robot_trajectory_auto_{timestamp}.csv'
        
        # CSV용 데이터 리스트 준비
        csv_data = []
        
        for data in self.trajectory_data:
            position_id = data['position_id']
            timestamp = data['timestamp']
            overall_mean = data['overall_mean']
            camera_stats = data['camera_stats']
            
            # 기본 행 (전체 평균 데이터)
            base_row = {
                'position_id': position_id,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'overall_x_mean': overall_mean['x'],
                'overall_y_mean': overall_mean['y'],
                'overall_x_std': overall_mean['x_std'],
                'overall_y_std': overall_mean['y_std'],
                'position_uncertainty': np.sqrt(overall_mean['x_std']**2 + overall_mean['y_std']**2),
                'total_samples': data['total_samples'],
                'participating_cameras': ', '.join(data['participating_cameras']),
                'num_participating_cameras': len(data['participating_cameras'])
            }
            
            # 각 카메라별 상세 데이터 추가
            for cam_name in ['cam1', 'cam2', 'cam3']:
                if cam_name in camera_stats:
                    cam_stat = camera_stats[cam_name]
                    base_row.update({
                        f'{cam_name}_sample_count': cam_stat['sample_count'],
                        f'{cam_name}_x_mean': cam_stat['x_mean'],
                        f'{cam_name}_x_std': cam_stat['x_std'],
                        f'{cam_name}_y_mean': cam_stat['y_mean'],
                        f'{cam_name}_y_std': cam_stat['y_std'],
                        f'{cam_name}_z_mean': cam_stat['z_mean'],
                        f'{cam_name}_z_std': cam_stat['z_std'],
                        f'{cam_name}_avg_markers': cam_stat['avg_markers']
                    })
                else:
                    # 해당 카메라에서 데이터가 없는 경우
                    base_row.update({
                        f'{cam_name}_sample_count': 0,
                        f'{cam_name}_x_mean': np.nan,
                        f'{cam_name}_x_std': np.nan,
                        f'{cam_name}_y_mean': np.nan,
                        f'{cam_name}_y_std': np.nan,
                        f'{cam_name}_z_mean': np.nan,
                        f'{cam_name}_z_std': np.nan,
                        f'{cam_name}_avg_markers': np.nan
                    })
            
            csv_data.append(base_row)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(csv_data)
        
        # 컬럼 순서 정리
        column_order = [
            'position_id', 'timestamp', 'datetime',
            'overall_x_mean', 'overall_y_mean', 'overall_x_std', 'overall_y_std', 'position_uncertainty',
            'total_samples', 'num_participating_cameras', 'participating_cameras'
        ]
        
        # 카메라별 컬럼 추가
        for cam_name in ['cam1', 'cam2', 'cam3']:
            column_order.extend([
                f'{cam_name}_sample_count', f'{cam_name}_x_mean', f'{cam_name}_x_std',
                f'{cam_name}_y_mean', f'{cam_name}_y_std', f'{cam_name}_z_mean', f'{cam_name}_z_std',
                f'{cam_name}_avg_markers'
            ])
        
        df = df[column_order]
        
        # CSV 저장
        try:
            df.to_csv(filename, index=False, float_format='%.6f')
            self.get_logger().info(f"✅ 궤적 데이터 CSV 저장 완료: {filename}")
            self.get_logger().info(f"   총 {len(df)} 개 위치 데이터 저장됨")
            return filename
        except Exception as e:
            self.get_logger().error(f"❌ CSV 저장 실패: {e}")
            return None

    def save_detailed_samples_to_csv(self, filename=None):
        """개별 샘플 데이터를 상세 CSV로 저장"""
        if len(self.trajectory_data) == 0:
            self.get_logger().warn("저장할 샘플 데이터가 없습니다.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'robot_samples_detailed_{timestamp}.csv'
        
        # 상세 샘플 데이터 리스트 준비
        detailed_data = []
        
        for data in self.trajectory_data:
            position_id = data['position_id']
            timestamp = data['timestamp']
            camera_stats = data['camera_stats']
            
            # 각 카메라의 개별 샘플들을 행으로 추가
            for cam_name, cam_stat in camera_stats.items():
                for i, (x, y, z, markers) in enumerate(zip(
                    cam_stat['x_coords'], cam_stat['y_coords'], 
                    cam_stat['z_coords'], cam_stat['marker_counts'])):
                    
                    detailed_data.append({
                        'position_id': position_id,
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'camera': cam_name,
                        'sample_index': i + 1,
                        'x': x,
                        'y': y,
                        'z': z,
                        'num_markers': markers
                    })
        
        # DataFrame 생성 및 저장
        df_detailed = pd.DataFrame(detailed_data)
        
        try:
            df_detailed.to_csv(filename, index=False, float_format='%.6f')
            self.get_logger().info(f"✅ 상세 샘플 데이터 CSV 저장 완료: {filename}")
            self.get_logger().info(f"   총 {len(df_detailed)} 개 개별 샘플 저장됨")
            return filename
        except Exception as e:
            self.get_logger().error(f"❌ 상세 CSV 저장 실패: {e}")
            return None

    def save_summary_statistics_to_csv(self, filename=None):
        """전체 통계 요약을 CSV로 저장"""
        if len(self.trajectory_data) == 0:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'robot_trajectory_summary_{timestamp}.csv'
        
        # 전체 통계 계산
        x_positions = [data['overall_mean']['x'] for data in self.trajectory_data]
        y_positions = [data['overall_mean']['y'] for data in self.trajectory_data]
        x_stds = [data['overall_mean']['x_std'] for data in self.trajectory_data]
        y_stds = [data['overall_mean']['y_std'] for data in self.trajectory_data]
        
        # 카메라별 참여도 통계
        camera_participation = {'cam1': 0, 'cam2': 0, 'cam3': 0}
        camera_marker_stats = {'cam1': [], 'cam2': [], 'cam3': []}
        
        for data in self.trajectory_data:
            for cam in data['participating_cameras']:
                camera_participation[cam] += 1
            
            for cam_name, cam_stat in data['camera_stats'].items():
                camera_marker_stats[cam_name].extend(cam_stat['marker_counts'])
        
        # 요약 통계 생성
        summary_stats = [{
            'metric': 'experiment_info',
            'total_positions': len(self.trajectory_data),
            'data_collection_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rotation_step_degrees': 5,  # 하드코딩된 값
            'samples_per_position': 5,   # 하드코딩된 값
            'value': '',
            'unit': ''
        }]
        
        # 위치 통계
        position_stats = [
            {'metric': 'x_position_mean', 'value': np.mean(x_positions), 'unit': 'm'},
            {'metric': 'x_position_std', 'value': np.std(x_positions), 'unit': 'm'},
            {'metric': 'x_position_range', 'value': max(x_positions) - min(x_positions), 'unit': 'm'},
            {'metric': 'y_position_mean', 'value': np.mean(y_positions), 'unit': 'm'},
            {'metric': 'y_position_std', 'value': np.std(y_positions), 'unit': 'm'},
            {'metric': 'y_position_range', 'value': max(y_positions) - min(y_positions), 'unit': 'm'},
            {'metric': 'average_position_uncertainty', 'value': np.mean([np.sqrt(x**2 + y**2) for x, y in zip(x_stds, y_stds)]), 'unit': 'm'},
            {'metric': 'max_position_uncertainty', 'value': max([np.sqrt(x**2 + y**2) for x, y in zip(x_stds, y_stds)]), 'unit': 'm'}
        ]
        
        # 카메라 통계
        camera_stats = []
        for cam_name in ['cam1', 'cam2', 'cam3']:
            camera_stats.extend([
                {'metric': f'{cam_name}_participation_count', 'value': camera_participation[cam_name], 'unit': 'positions'},
                {'metric': f'{cam_name}_participation_rate', 'value': camera_participation[cam_name] / len(self.trajectory_data), 'unit': 'ratio'},
                {'metric': f'{cam_name}_avg_markers', 'value': np.mean(camera_marker_stats[cam_name]) if camera_marker_stats[cam_name] else 0, 'unit': 'markers'},
                {'metric': f'{cam_name}_total_samples', 'value': len(camera_marker_stats[cam_name]), 'unit': 'samples'}
            ])
        
        # 모든 통계 합치기
        all_stats = summary_stats + position_stats + camera_stats
        
        df_summary = pd.DataFrame(all_stats)
        
        try:
            df_summary.to_csv(filename, index=False, float_format='%.6f')
            self.get_logger().info(f"✅ 요약 통계 CSV 저장 완료: {filename}")
            return filename
        except Exception as e:
            self.get_logger().error(f"❌ 요약 통계 CSV 저장 실패: {e}")
            return None

    # analyze_and_store_samples_auto 함수 수정 (CSV 저장 추가)
    def analyze_and_store_samples_auto(self, samples):
        """자동 모드용 샘플 분석 및 저장 (CSV 저장 추가)"""
        
        # 기존 분석 코드...
        camera_data = {'cam1': [], 'cam2': [], 'cam3': []}
        
        for sample in samples:
            for cam_name in ['cam1', 'cam2', 'cam3']:
                if cam_name in sample:
                    camera_data[cam_name].append(sample[cam_name])
        
        camera_stats = {}
        all_x_coords = []
        all_y_coords = []
        
        for cam_name, cam_samples in camera_data.items():
            if len(cam_samples) > 0:
                x_coords = [s['x'] for s in cam_samples]
                y_coords = [s['y'] for s in cam_samples]
                z_coords = [s['z'] for s in cam_samples]
                marker_counts = [s['num_markers'] for s in cam_samples]
                
                camera_stats[cam_name] = {
                    'sample_count': len(cam_samples),
                    'x_mean': np.mean(x_coords),
                    'x_std': np.std(x_coords) if len(x_coords) > 1 else 0.0,
                    'y_mean': np.mean(y_coords),
                    'y_std': np.std(y_coords) if len(y_coords) > 1 else 0.0,
                    'z_mean': np.mean(z_coords),
                    'z_std': np.std(z_coords) if len(z_coords) > 1 else 0.0,
                    'avg_markers': np.mean(marker_counts),
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'z_coords': z_coords,
                    'marker_counts': marker_counts
                }
                
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
        
        if len(all_x_coords) > 0:
            overall_x_mean = np.mean(all_x_coords)
            overall_y_mean = np.mean(all_y_coords)
            overall_x_std = np.std(all_x_coords) if len(all_x_coords) > 1 else 0.0
            overall_y_std = np.std(all_y_coords) if len(all_y_coords) > 1 else 0.0
        else:
            overall_x_mean = overall_y_mean = overall_x_std = overall_y_std = 0.0
        
        position_data = {
            'timestamp': time.time(),
            'position_id': len(self.trajectory_data) + 1,
            'overall_mean': {
                'x': overall_x_mean,
                'y': overall_y_mean,
                'x_std': overall_x_std,
                'y_std': overall_y_std
            },
            'camera_stats': camera_stats,
            'total_samples': len([s for s in samples if s]),
            'participating_cameras': list(camera_stats.keys())
        }
        
        self.trajectory_data.append(position_data)
        
        # 로그 출력
        self.get_logger().info(f"✅ 위치 {position_data['position_id']} 데이터 저장:")
        self.get_logger().info(f"   전체 평균: X={overall_x_mean:.4f}±{overall_x_std:.4f}m, Y={overall_y_mean:.4f}±{overall_y_std:.4f}m")
        self.get_logger().info(f"   참여 카메라: {len(camera_stats)}개 ({', '.join(camera_stats.keys())})")
        
        # ✅ 주기적으로 CSV 저장 (10개 위치마다)
        if len(self.trajectory_data) % 10 == 0:
            self.save_trajectory_to_csv()
            self.get_logger().info(f"📊 중간 CSV 저장 완료 ({len(self.trajectory_data)} 위치)")


def main(args=None):
    rclpy.init(args=args)
    world_robot_node = World_Robot()
    
    try:
        rclpy.spin(world_robot_node)
    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    finally:
        # ✅ 종료 시 모든 데이터를 CSV로 저장
        if len(world_robot_node.trajectory_data) > 0:
            print("📊 데이터를 CSV 파일로 저장 중...")
            
            # 1. 메인 궤적 데이터
            trajectory_file = world_robot_node.save_trajectory_to_csv()
            
            # 2. 상세 샘플 데이터  
            detailed_file = world_robot_node.save_detailed_samples_to_csv()
            
            # 3. 요약 통계
            summary_file = world_robot_node.save_summary_statistics_to_csv()
            
            print("✅ CSV 파일 저장 완료:")
            if trajectory_file:
                print(f"   - 궤적 데이터: {trajectory_file}")
            if detailed_file:
                print(f"   - 상세 샘플: {detailed_file}")
            if summary_file:
                print(f"   - 요약 통계: {summary_file}")
        
        # 종료 시 그래프 그리기
        world_robot_node.plot_trajectory()
        world_robot_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()