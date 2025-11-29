import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, TransformStamped
from cv_bridge import CvBridge
import os
from multicam_aruco import ARUCOBoardPose, ARUCORobotPose
import cv2
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster  # ✅ TF2 추가
# import tf_transformations  # ✅ TF 변환 유틸리티

class World_Robot(Node):
    def __init__(self):
        super().__init__('world_robot')
        self.bridge = CvBridge()
        self.world_aruco_detector = ARUCOBoardPose()
        self.robot_aruco_detector = ARUCORobotPose()

        # ✅ TF Broadcaster 초기화
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

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
        
        self.robot_pose_pub = self.create_publisher(Pose, '/mobile_base/pose', 10)

        self.create_timer(0.1, self.loop)

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
        

    def publish_transforms(self, H_world_to_robot_results):
        """동적 TF 변환 발행"""
        current_time = self.get_clock().now().to_msg()
        
        # ✅ 1. World ArUco 프레임 발행 (카메라2에서 감지된 경우)
        if hasattr(self, '_world_detected') and self._world_detected:
            # World ArUco는 이미 world 프레임으로 설정되어 있으므로 
            # 여기서는 world 프레임을 기준으로 함
            pass
        
        # ✅ 2. 각 카메라에서 감지된 로봇 위치 발행
        for cam_name, H_matrix in H_world_to_robot_results.items():
            if H_matrix is not None:
                H_matrix = H_matrix @ self.create_x_rotation_matrix(180) @ self.create_z_rotation_matrix(90)  # X축 기준 180도 회전 적용
                # 위치 추출
                translation = H_matrix[:3, 3]
                
                # 회전 행렬에서 quaternion 변환
                rotation_matrix = H_matrix[:3, :3]
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                
                # TF 메시지 생성
                transform = TransformStamped()
                transform.header.stamp = current_time
                transform.header.frame_id = 'world'
                transform.child_frame_id = f'robot_from_{cam_name}'
                
                transform.transform.translation.x = float(translation[0])
                transform.transform.translation.y = float(translation[1])
                transform.transform.translation.z = float(translation[2])
                
                transform.transform.rotation.x = float(quaternion[0])
                transform.transform.rotation.y = float(quaternion[1])
                transform.transform.rotation.z = float(quaternion[2])
                transform.transform.rotation.w = float(quaternion[3])
                
                self.tf_broadcaster.sendTransform(transform)
        
        # ✅ 3. 평균 로봇 위치 발행
        if len(H_world_to_robot_results) > 0:
            # 평균 계산
            x_positions = []
            y_positions = []
            z_positions = []
            rotation_matrices = []
            
            for cam_name, H_matrix in H_world_to_robot_results.items():
                if H_matrix is not None:
                    H_matrix = H_matrix @ self.create_x_rotation_matrix(180) @ self.create_z_rotation_matrix(90)
                    x_positions.append(H_matrix[0, 3])
                    y_positions.append(H_matrix[1, 3])
                    z_positions.append(H_matrix[2, 3])
                    rotation_matrices.append(H_matrix[:3, :3])
            
            if len(x_positions) > 0:
                # 평균 위치
                avg_x = np.mean(x_positions)
                avg_y = np.mean(y_positions)
                avg_z = np.mean(z_positions)
                
                # 평균 회전
                avg_rotation_matrix = self.average_rotations(rotation_matrices)
                avg_quaternion = self.rotation_matrix_to_quaternion(avg_rotation_matrix)
                
                # 평균 로봇 위치 TF 발행
                robot_transform = TransformStamped()
                robot_transform.header.stamp = current_time
                robot_transform.header.frame_id = 'world'
                robot_transform.child_frame_id = 'robot_base'
                
                robot_transform.transform.translation.x = float(avg_x)
                robot_transform.transform.translation.y = float(avg_y)
                robot_transform.transform.translation.z = float(avg_z)
                
                robot_transform.transform.rotation.x = float(avg_quaternion[0])
                robot_transform.transform.rotation.y = float(avg_quaternion[1])
                robot_transform.transform.rotation.z = float(avg_quaternion[2])
                robot_transform.transform.rotation.w = float(avg_quaternion[3])
                
                self.tf_broadcaster.sendTransform(robot_transform)

                # ✅ Pose 메시지 생성
                mobile_base_pose = Pose()
                mobile_base_pose.position.x = float(avg_x)
                mobile_base_pose.position.y = float(avg_y)
                mobile_base_pose.position.z = float(avg_z)

                # ✅ 평균 orientation 설정
                mobile_base_pose.orientation.x = float(avg_quaternion[0])  # x
                mobile_base_pose.orientation.y = float(avg_quaternion[1])  # y
                mobile_base_pose.orientation.z = float(avg_quaternion[2])  # z
                mobile_base_pose.orientation.w = float(avg_quaternion[3])  # w
                
                self.robot_pose_pub.publish(mobile_base_pose)


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
                self._world_detected = False
                return
        
        # ✅ World 감지됨 표시
        self._world_detected = True

        t_world_cam2, R_world_cam2 = world_result_cam2
        H_cam2_to_world = self._conver_tR_to_H(t_world_cam2, R_world_cam2)  # cam2 -> world

        # ✅ X축 기준 180도 회전 적용
        R_x_180 = self.create_x_rotation_matrix(180)  # X축 기준 180도 회전
        # ✅ World 좌표계를 X축 기준 180도 회전
        H_cam2_to_world_rotated = H_cam2_to_world @ R_x_180
        H_world_to_cam2 = self.inverse_homogeneous_matrix(H_cam2_to_world_rotated)  # world -> cam2
        # ✅ World ArUco 프레임 발행
        current_time = self.get_clock().now().to_msg()
        world_aruco_transform = TransformStamped()
        world_aruco_transform.header.stamp = current_time
        world_aruco_transform.header.frame_id = 'cam2_frame'
        world_aruco_transform.child_frame_id = 'world'
        
        # World ArUco의 위치와 회전
        world_translation = H_cam2_to_world_rotated[:3, 3]
        world_quaternion = self.rotation_matrix_to_quaternion(H_cam2_to_world_rotated[:3, :3])
        
        world_aruco_transform.transform.translation.x = float(world_translation[0])
        world_aruco_transform.transform.translation.y = float(world_translation[1])
        world_aruco_transform.transform.translation.z = float(world_translation[2])
        
        world_aruco_transform.transform.rotation.x = float(world_quaternion[0])
        world_aruco_transform.transform.rotation.y = float(world_quaternion[1])
        world_aruco_transform.transform.rotation.z = float(world_quaternion[2])
        world_aruco_transform.transform.rotation.w = float(world_quaternion[3])
        
        self.tf_broadcaster.sendTransform(world_aruco_transform)

        # ✅ 신뢰성 임계값 설정
        MIN_MARKERS_DETECTED = 3  # 최소 감지된 마커 수
        
        # 사용 가능한 결과들만 저장
        available_results = {}
        H_world_to_robot_results = {}

        # 각 카메라별로 결과가 있는지 확인하고 계산
        if robot_result_cam1 is not None:
            t_robot_cam1, R_robot_cam1, num_aruco_cam1 = robot_result_cam1
            if num_aruco_cam1 <= MIN_MARKERS_DETECTED:
                pass
                # available_results['cam1'] = (t_robot_cam1, R_robot_cam1)
            else:
                H_cam1_to_robot_original = self._conver_tR_to_H(t_robot_cam1, R_robot_cam1)  # cam1 -> robot (원본)
                # ✅ X축 기준 180도 회전 적용
                H_cam1_to_robot = H_cam1_to_robot_original #@ R_x_180  # 회전된 robot 좌표계
                H_world_to_robot_results['cam1'] = H_world_to_cam2 @ self.H_cam2_to_cam1 @ H_cam1_to_robot
            
        if robot_result_cam2 is not None:
            t_robot_cam2, R_robot_cam2, num_aruco_cam2 = robot_result_cam2
            if num_aruco_cam2 <= MIN_MARKERS_DETECTED:
                pass
            else:
                H_cam2_to_robot_original = self._conver_tR_to_H(t_robot_cam2, R_robot_cam2)  # cam2 -> robot (원본)
                # ✅ X축 기준 180도 회전 적용
                H_cam2_to_robot = H_cam2_to_robot_original #@ R_x_180  # 회전된 robot 좌표계
                H_world_to_robot_results['cam2'] = H_world_to_cam2 @ H_cam2_to_robot
            
        if robot_result_cam3 is not None:
            t_robot_cam3, R_robot_cam3, num_aruco_cam3 = robot_result_cam3
            if num_aruco_cam3 <= MIN_MARKERS_DETECTED:
                pass
            else:
                H_cam3_to_robot_original = self._conver_tR_to_H(t_robot_cam3, R_robot_cam3)  # cam3 -> robot (원본)
                # ✅ X축 기준 180도 회전 적용
                H_cam3_to_robot = H_cam3_to_robot_original #@ R_x_180  # 회전된 robot 좌표계
                H_world_to_robot_results['cam3'] = H_world_to_cam2 @ self.H_cam2_to_cam3 @ H_cam3_to_robot

        # ✅ TF 변환 발행
        self.publish_transforms(H_world_to_robot_results)

        self.display_images(img_1, img_2_world, img_2_robot, img_3, 
                       robot_result_cam1, robot_result_cam2, robot_result_cam3, 
                       world_result_cam2, H_world_to_robot_results)          

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
    
    def rotation_matrix_to_quaternion(self, R_matrix):
        """회전 행렬을 quaternion으로 변환"""
        r = R.from_matrix(R_matrix)
        quat = r.as_quat()  # [x, y, z, w] 순서
        return quat

    def average_rotations(self, rotation_matrices):
        """여러 회전 행렬들의 평균 계산 (Quaternion 평균 방법)"""
        if len(rotation_matrices) == 1:
            return rotation_matrices[0]
        
        # 모든 회전 행렬을 quaternion으로 변환
        quaternions = []
        for R_mat in rotation_matrices:
            quat = self.rotation_matrix_to_quaternion(R_mat)
            quaternions.append(quat)
        
        # Quaternion 평균 계산 (첫 번째를 기준으로)
        q_mean = quaternions[0].copy()
        
        for i in range(1, len(quaternions)):
            q = quaternions[i]
            # Quaternion의 부호 일치 확인 (shortest path)
            if np.dot(q_mean, q) < 0:
                q = -q
            q_mean = q_mean + q
        
        # 정규화
        q_mean = q_mean / np.linalg.norm(q_mean)
        
        # 다시 회전 행렬로 변환
        r_avg = R.from_quat(q_mean)
        return r_avg.as_matrix()
    
    def create_x_rotation_matrix(self, angle_degrees):
        """X축 기준 회전 행렬 생성 (4x4 동차 변환 행렬)"""
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        R_x = np.array([
            [1,     0,      0,     0],
            [0,  cos_a, -sin_a,  0],
            [0,  sin_a,  cos_a,  0],
            [0,     0,      0,     1]
        ])
        return R_x
    
    def create_z_rotation_matrix(self, angle_degrees):
        """Z축 기준 회전 행렬 생성 (4x4 동차 변환 행렬)"""
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        R_z = np.array([
            [cos_a, -sin_a, 0,  0],
            [sin_a,  cos_a, 0,  0],
            [    0,      0, 1,  0],
            [    0,      0, 0,  1]
        ])
        return R_z

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
        
        
        # Camera 2 World 상태
        if world_result_cam2 is not None:
            cv2.putText(display_img_2_world, "World ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            # t_world_cam2, R_world_cam2 = world_result_cam2
            # H_cam2_to_world = self._conver_tR_to_H(t_world_cam2, R_world_cam2)  # cam2 -> world
            # H_world_to_cam2 = self.inverse_homogeneous_matrix(H_cam2_to_world)  # world -> cam2
            
        else:
            cv2.putText(display_img_2_world, "World ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
            
        
        # Camera 1 상태
        if robot_result_cam1 is not None:
            cv2.putText(display_img_1, "Robot ArUco: DETECTED", (10, 30), font, font_scale, (0, 255, 0), thickness)
            if 'cam1' in H_world_to_robot_results:
                pos = H_world_to_robot_results['cam1'][:3, 3]
                cv2.putText(display_img_1, f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", 
                        (10, 60), font, font_scale, (0, 255, 0), thickness)

        else:
            cv2.putText(display_img_1, "Robot ArUco: NOT DETECTED", (10, 30), font, font_scale, (0, 0, 255), thickness)
        
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
        
        # # 비교 결과 표시 (Camera 2 World 이미지에)
        # y_offset = 90

        # # ✅ 평균 위치 계산 및 표시
        # if len(H_world_to_robot_results) >= 1:
        #     # 모든 유효한 카메라의 위치 및 회전 수집
        #     x_positions = []
        #     y_positions = []
        #     z_positions = []
        #     rotation_matrices = []
            
        #     for cam_name, H_matrix in H_world_to_robot_results.items():
        #         if H_matrix is not None:
        #             x_positions.append(H_matrix[0, 3])
        #             y_positions.append(H_matrix[1, 3])
        #             z_positions.append(H_matrix[2, 3])
        #             rotation_matrices.append(H_matrix[:3, :3])  # 회전 행렬 추가
            
        #     if len(x_positions) > 0:
        #         # 평균 위치 계산
        #         avg_x = np.mean(x_positions) 
        #         avg_y = np.mean(y_positions) 
        #         avg_z = np.mean(z_positions)
                
        #         # ✅ 평균 회전 계산
        #         avg_rotation_matrix = self.average_rotations(rotation_matrices)
        #         avg_quaternion = self.rotation_matrix_to_quaternion(avg_rotation_matrix)
                
        #         # ✅ Pose 메시지 생성
        #         mobile_base_pose = Pose()
        #         mobile_base_pose.position.x = float(avg_x)
        #         mobile_base_pose.position.y = float(avg_y)
        #         mobile_base_pose.position.z = float(avg_z)

        #         # ✅ 평균 orientation 설정
        #         mobile_base_pose.orientation.x = float(avg_quaternion[0])  # x
        #         mobile_base_pose.orientation.y = float(avg_quaternion[1])  # y
        #         mobile_base_pose.orientation.z = float(avg_quaternion[2])  # z
        #         mobile_base_pose.orientation.w = float(avg_quaternion[3])  # w
                
        #         self.robot_pose_pub.publish(mobile_base_pose)

        #         # ✅ 위치 표준편차 계산
        #         std_x = np.std(x_positions) if len(x_positions) > 1 else 0.0
        #         std_y = np.std(y_positions) if len(y_positions) > 1 else 0.0
        #         std_z = np.std(z_positions) if len(z_positions) > 1 else 0.0
                
        #         # ✅ 회전 불확실성 계산 (각 회전 행렬과 평균의 차이)
        #         rotation_uncertainties = []
        #         if len(rotation_matrices) > 1:
        #             for R_mat in rotation_matrices:
        #                 # 두 회전 행렬 간의 각도 차이 계산
        #                 R_diff = avg_rotation_matrix.T @ R_mat
        #                 angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        #                 rotation_uncertainties.append(np.degrees(angle_diff))
                    
        #             rotation_std = np.std(rotation_uncertainties)
        #         else:
        #             rotation_std = 0.0
                
        #         # 위치 정확도 (표준편차의 크기)
        #         position_uncertainty = np.sqrt(std_x**2 + std_y**2 + std_z**2) * 1000  # mm
                
        #         # ✅ 오일러 각도로 변환 (표시용)
        #         euler_angles = R.from_quat(avg_quaternion).as_euler('xyz', degrees=True)
                
        #         # 로봇 위치 및 방향 표시
        #         cv2.putText(display_img_2_world, f"Robot Pose (from {len(x_positions)} cameras):", 
        #             (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        #         y_offset += 30
                
        #         cv2.putText(display_img_2_world, f"X: {avg_x:.3f}m (±{std_x*1000:.1f}mm)", 
        #             (10, y_offset), font, 0.5, (0, 255, 255), 2)
        #         y_offset += 25
                
        #         cv2.putText(display_img_2_world, f"Y: {avg_y:.3f}m (±{std_y*1000:.1f}mm)", 
        #             (10, y_offset), font, 0.5, (0, 255, 255), 2)
        #         y_offset += 25
                
        #         cv2.putText(display_img_2_world, f"Z: {avg_z:.3f}m (±{std_z*1000:.1f}mm)", 
        #             (10, y_offset), font, 0.5, (0, 255, 255), 2)
        #         y_offset += 25
                
        #         # ✅ 방향 정보 추가
        #         cv2.putText(display_img_2_world, f"Roll: {euler_angles[0]:.1f}° (±{rotation_std:.1f}°)", 
        #             (10, y_offset), font, 0.5, (255, 200, 100), 2)
        #         y_offset += 25
                
        #         cv2.putText(display_img_2_world, f"Pitch: {euler_angles[1]:.1f}° (±{rotation_std:.1f}°)", 
        #             (10, y_offset), font, 0.5, (255, 200, 100), 2)
        #         y_offset += 25
                
        #         cv2.putText(display_img_2_world, f"Yaw: {euler_angles[2]:.1f}° (±{rotation_std:.1f}°)", 
        #             (10, y_offset), font, 0.5, (255, 200, 100), 2)
        #         y_offset += 25
                
        #         # 전체 불확실성
        #         if len(x_positions) > 1:
        #             cv2.putText(display_img_2_world, f"Pos Uncertainty: ±{position_uncertainty:.1f}mm", 
        #                 (10, y_offset), font, 0.5, (255, 128, 0), 2)
        #             y_offset += 25
        #             cv2.putText(display_img_2_world, f"Rot Uncertainty: ±{rotation_std:.1f}°", 
        #                 (10, y_offset), font, 0.5, (255, 128, 0), 2)
        #             y_offset += 25
                
        #         # 품질 평가 (위치 + 회전 고려)
        #         overall_uncertainty = position_uncertainty + rotation_std * 5  # 회전은 5mm/degree로 가중
        #         if overall_uncertainty < 15:
        #             quality_text = "Quality: Excellent"
        #             quality_color = (0, 255, 0)  # 초록
        #         elif overall_uncertainty < 50:
        #             quality_text = "Quality: Good"  
        #             quality_color = (0, 255, 255)  # 노랑
        #         else:
        #             quality_text = "Quality: Poor"
        #             quality_color = (0, 0, 255)  # 빨강
                    
        #         cv2.putText(display_img_2_world, quality_text, 
        #             (10, y_offset), font, 0.5, quality_color, 2)
                
        #         # ✅ 로그에도 평균 위치 및 방향 출력
        #         if hasattr(self, '_log_counter'):
        #             self._log_counter += 1
        #         else:
        #             self._log_counter = 1
                    
        #         if self._log_counter % 50 == 0:  # 50회마다 로그 출력
        #             self.get_logger().info(
        #                 f"Robot Pose: Pos=[{avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f}]m ±{position_uncertainty:.1f}mm, "
        #                 f"Rot=[{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}]° ±{rotation_std:.1f}°"
        #             )

        # else:
        #     cv2.putText(display_img_2_world, "No robot detected", 
        #         (10, y_offset), font, font_scale, (0, 0, 255), thickness)
        
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
            rclpy.shutdown()
        elif key == ord('s'):
            # 스크린샷 저장
            timestamp = self.get_clock().now().to_msg()
            cv2.imwrite(f'cam1_{timestamp.sec}.jpg', display_img_1)
            cv2.imwrite(f'cam2_world_{timestamp.sec}.jpg', display_img_2_world)
            cv2.imwrite(f'cam2_robot_{timestamp.sec}.jpg', display_img_2_robot)
            cv2.imwrite(f'cam3_{timestamp.sec}.jpg', display_img_3)
            self.get_logger().info("스크린샷 저장됨")

    def main(args=None):
        rclpy.init(args=args)
        world_robot_node = World_Robot()
        rclpy.spin(world_robot_node)
        world_robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    World_Robot.main()