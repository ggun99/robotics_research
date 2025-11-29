import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import os
from multicam_aruco import ARUCOBoardPose, ARUCORobotPose
import cv2


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
        
        # 비교 결과 표시 (Camera 2 World 이미지에)
        y_offset = 90
        if len(H_world_to_robot_results) >= 2:
            cv2.putText(display_img_2_world, f"Comparing {len(H_world_to_robot_results)} cameras:", 
                    (10, y_offset), font, font_scale, (255, 255, 0), thickness)
            y_offset += 30
            
            cam_names = list(H_world_to_robot_results.keys())
            for i, cam1 in enumerate(cam_names):
                for cam2 in cam_names[i+1:]:
                    H1 = H_world_to_robot_results[cam1]
                    H2 = H_world_to_robot_results[cam2]
                    
                    pos_diff = np.linalg.norm(H1[:3, 3] - H2[:3, 3]) * 1000  # mm
                    
                    try:
                        R_diff = H1[:3, :3] @ H2[:3, :3].T
                        trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
                        rot_diff = np.degrees(np.arccos(trace_val))
                    except:
                        rot_diff = 0.0
                    
                    cv2.putText(display_img_2_world, f"{cam1} vs {cam2}: {pos_diff:.1f}mm, {rot_diff:.1f}deg", 
                            (10, y_offset), font, 0.5, (255, 255, 0), 1)
                    y_offset += 25
        
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