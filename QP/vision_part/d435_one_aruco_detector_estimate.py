import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import time
import tf2_ros
from scipy.spatial.transform import Rotation as R
# from tf_transformations import quaternion_from_matrix

class D435ArucoDetector(Node):
    def __init__(self):
        super().__init__('d435_aruco_detector')
        
        # 로봇 마커들 (0~3번) - 로컬 좌표계
        self.robot_markers_local = {
            0: np.array([-0.045, 0.0565, 0.0]),
            1: np.array([0.045, -0.0565, 0.0]),
            2: np.array([0.045, -0.0565, 0.0]),
            3: np.array([-0.045, 0.0565, 0.0])
        }
    
        self.marker_length = 0.075  # 마커 크기 5cm
        
        # 월드 좌표계 설정 상태
        self.world_established = False
        self.camera_to_world_R = None
        self.camera_to_world_t = None
        
        # 카메라 정보
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # 로봇 중심 추적
        self.latest_robot_center = None
        self.center_timestamp = None
        
        self.aruco_50 = [0, 1,2,3]
        self.aruco_75 = [10]

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 변환 행렬들
        self.H_world2cam = None    # 10번 마커 기준 월드→카메라
        self.H_cam2robot = None    # 카메라→로봇 중심
        self.H_world2robot = None  # 월드→로봇
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 구독자 설정
        self.create_subscription(
            Image, '/camera3/camera3/color/image_raw', 
            self.image_callback, 10
        )
        self.create_subscription(
            CameraInfo, '/camera3/camera3/color/camera_info',
            self.camera_info_callback, 10
        )
        
        # 발행자
        self.robot_center_pub = self.create_publisher(
            PoseStamped, '/robot_center', 10
        )
        
        self.get_logger().info("D435 ArUco Detector initialized")

    def camera_info_callback(self, msg):
        """카메라 내부 매개변수 수신"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info("Camera intrinsics received")

    def image_callback(self, msg):
        """이미지 처리 및 ArUco 검출"""
        if not self.camera_info_received:
            return
            
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ArUco 마커 검출
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return

        # 기본 포즈 추정 (카메라 좌표계)
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
        except Exception as e:
            self.get_logger().error(f"Pose estimation failed: {e}")
            return

        # 검출된 마커 정리
        detected_markers = {}
        for i, marker_id in enumerate(ids.flatten()):
            detected_markers[marker_id] = {
                'cam_tvec': tvecs[i].reshape(3),
                'cam_rvec': rvecs[i].reshape(3),
                'corners': corners[i]
            }

        # 🆕 단순한 월드 좌표계 설정: 10번 마커 = 월드 원점
        H_world2cam = None
        robot_markers_cam = {}

        for marker_id, data in detected_markers.items():
            # ✅ 10번 마커를 월드 좌표계로 설정 (단순!)
            if marker_id == 10:
                rvec = data['cam_rvec']
                tvec = data['cam_tvec']
                H_cam2world = np.eye(4)
                R, _ = cv2.Rodrigues(rvec)
                H_cam2world[0:3, 0:3] = R
                H_cam2world[0:3, 3] = tvec
                # 마커 10의 좌표계 = 월드 좌표계
                # H_cam2marker10 의 역변환 = H_marker10→cam = H_world→cam
                H_world2cam = np.linalg.inv(H_cam2world)
                self.H_world2cam = H_world2cam  # 클래스 변수에 저장
                self.world_established = True
                self.broadcast_world_transform(H_world2cam, msg.header.stamp)
                self.get_logger().info("✅ Marker 10 set as world origin")
                self.get_logger().info(f"Marker 10 position (cam): {tvec}")
            
            # 로봇 마커들 수집
            elif marker_id in self.robot_markers_local:
                rvec = data['cam_rvec']
                tvec = data['cam_tvec']
                R, _ = cv2.Rodrigues(rvec)
                H_cam2marker = np.eye(4)
                H_cam2marker[0:3, 0:3] = R
                H_cam2marker[0:3, 3] = tvec
                robot_markers_cam[marker_id] = {
                    'position': tvec,
                    'rotation': R,
                    'transform': H_cam2marker
                }

        # 10번 마커가 없으면 월드 좌표계 없음
        if not self.world_established:
            self.get_logger().warn("❌ Marker 10 not detected - no world coordinate system")
            self.visualize_results(cv_image, detected_markers)
            return

        # 🤖 로봇 중심 계산 (개선됨)
        if len(robot_markers_cam) > 0:
            # ✅ 강건한 로봇 중심 계산
            result = self.calculate_robot_center_robust(robot_markers_cam)
            if result[0] is not None:
                robot_center_cam, robot_rotation_cam = result
                
                # H_cam2robot
                H_cam2robot = np.eye(4)
                H_cam2robot[0:3, 0:3] = robot_rotation_cam
                H_cam2robot[0:3, 3] = robot_center_cam
                self.H_cam2robot = H_cam2robot
                
                # H_world2robot 계산
                H_world2robot = H_world2cam @ H_cam2robot
                robot_center_world = H_world2robot[:3, 3]
                
                self.broadcast_robot_transform(H_world2robot, msg.header.stamp)
                self.latest_robot_center = robot_center_world
                self.center_timestamp = time.time()
                
                # 검출된 마커 수 로그
                self.get_logger().info(f"🤖 Robot center from {len(robot_markers_cam)} markers: {robot_center_world}")
                
                # 발행
                self.publish_robot_center(robot_center_world, len(robot_markers_cam))
            else:
                self.get_logger().warn("❌ Cannot estimate robot center")

        # 시각화
        self.visualize_results(cv_image, detected_markers)

    def project_world_to_image_simple(self, world_point):
        """10번 마커 기준 월드 좌표를 이미지로 투영"""
        
        if not hasattr(self, 'H_world2cam') or self.H_world2cam is None:
            return None
        
        try:
            # 월드 → 카메라 변환
            world_point_homogeneous = np.append(world_point, 1)
            cam_point_homogeneous = self.H_world2cam @ world_point_homogeneous
            cam_point = cam_point_homogeneous[:3]
            
            if cam_point[2] <= 0:  # 카메라 뒤에 있음
                return None
            
            # 카메라 → 이미지 투영
            image_points, _ = cv2.projectPoints(
                cam_point.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                self.camera_matrix, self.dist_coeffs
            )
            
            return image_points[0][0]
            
        except Exception as e:
            self.get_logger().error(f"Projection failed: {e}")
            return None
        


    def publish_robot_center(self, center, num_markers):
        """로봇 중심 위치 발행"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = 1.0
        
        self.robot_center_pub.publish(msg)
        
        self.get_logger().info(
            f"Published robot center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
            f"from {num_markers} markers"
        )
        
    def broadcast_robot_transform(self, H_world2robot, timestamp):
        """로봇 좌표계 tf 브로드캐스트"""
        # 월드 → 로봇 변환을 로봇 → 월드로 변환
        H_robot2world = np.linalg.inv(H_world2robot)
        
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot_center'
        
        # 위치
        t.transform.translation.x = float(H_world2robot[0, 3])
        t.transform.translation.y = float(H_world2robot[1, 3])
        t.transform.translation.z = float(H_world2robot[2, 3])
        
        # 회전
        R_mat = H_robot2world[0:3, 0:3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def broadcast_world_transform(self, H_world2cam, timestamp):
        """월드 좌표계 tf 브로드캐스트"""
        # 카메라 → 월드 변환
        H_cam2world = np.linalg.inv(H_world2cam)
        
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'camera_link'
        t.child_frame_id = 'world'
        
        # 위치
        t.transform.translation.x = float(H_cam2world[0, 3])
        t.transform.translation.y = float(H_cam2world[1, 3])
        t.transform.translation.z = float(H_cam2world[2, 3])
        
        # 회전
        R_mat = H_cam2world[0:3, 0:3]
        rot = R.from_matrix(R_mat)
        quat = rot.as_quat()  # [x, y, z, w]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def calculate_robot_rotation_simple(self, robot_markers_cam):
        """간단한 아웃라이어 제거 + 회전 평균"""
        
        if len(robot_markers_cam) <= 1:
            return list(robot_markers_cam.values())[0]['rotation']
        
        from scipy.spatial.transform import Rotation as R_scipy
        
        # Quaternion 변환
        rotations_data = []
        for marker_id, data in robot_markers_cam.items():
            rot_matrix = data['rotation']
            r = R_scipy.from_matrix(rot_matrix)
            quat = r.as_quat()
            rotations_data.append((marker_id, quat, rot_matrix))
        
        # 간단한 아웃라이어 제거: 첫 번째와 너무 다른 것 제거
        reference_quat = rotations_data[0][1]
        valid_rotations = []
        
        for marker_id, quat, rot_matrix in rotations_data:
            # 각도 차이 계산
            dot_product = np.abs(np.dot(reference_quat, quat))
            angle_diff = 2 * np.arccos(np.clip(dot_product, 0, 1))
            
            if angle_diff <= 0.5:  # 약 30도 이내
                valid_rotations.append(quat)
            else:
                self.get_logger().warn(f"⚠️ Marker {marker_id} removed: angle diff {np.degrees(angle_diff):.1f}°")
        
        # 평균 계산
        if len(valid_rotations) > 0:
            mean_quat = np.mean(valid_rotations, axis=0)
            mean_quat = mean_quat / np.linalg.norm(mean_quat)
            return R_scipy.from_quat(mean_quat).as_matrix()
        else:
            return rotations_data[0][2]  # 첫 번째 마커 사용
    
    def visualize_results(self, image, detected_markers):
        """검출 결과와 좌표계 축 시각화"""
        
        # 검출된 마커들 표시
        for marker_id, data in detected_markers.items():
            corners = data['corners'][0]
            
            
            
            # 마커 ID 표시 - 색상 구분
            center = np.mean(corners, axis=0).astype(int)
            
            if marker_id == 10:
                # 10번 마커 - 월드 원점 (파란색)
                color = (255, 0, 0)
                label = f"World{marker_id}"
            # elif marker_id in self.robot_markers_local:
            #     # 로봇 마커들 (빨간색)
            #     color = (0, 0, 255)
            #     label = f"R{marker_id}"
                # 마커 경계 그리기
                cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(image, label, tuple(center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
                # # 🆕 마커별 좌표축 그리기
                self.draw_coordinate_axes(image, data['cam_rvec'], data['cam_tvec'], 0.05)

        # 🤖 로봇 중심과 로봇 좌표축 표시
        if (self.world_established and 
            self.latest_robot_center is not None and
            self.center_timestamp is not None and 
            time.time() - self.center_timestamp < 1.0):
            
            # 로봇 중심을 이미지로 투영
            center_image = self.project_world_to_image_simple(self.latest_robot_center)
            if center_image is not None:
                center_pt = tuple(center_image.astype(int))
                
                # 로봇 중심 표시
                cv2.circle(image, center_pt, 8, (0, 255, 255), -1)
                cv2.putText(image, "Robot", (center_pt[0]+15, center_pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 🆕 로봇 좌표축 그리기
                self.draw_robot_coordinate_axes(image)

        # 📊 간단한 상태 정보만
        world_status = "World: OK" if self.world_established else "World: No Marker 10"
        world_color = (0, 255, 0) if self.world_established else (0, 0, 255)
        cv2.putText(image, world_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, world_color, 2)
        
        # 로봇 좌표 (간단히)
        if self.latest_robot_center is not None:
            coord_text = f"Robot: ({self.latest_robot_center[0]:.2f}, {self.latest_robot_center[1]:.2f}, {self.latest_robot_center[2]:.2f})"
            cv2.putText(image, coord_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 이미지 표시
        cv2.imshow('ArUco Detection with Coordinate Axes', image)
        cv2.waitKey(1)

    def calculate_robot_center_robust(self, robot_markers_cam):
        """마커 일부가 보이지 않아도 안정적인 로봇 중심 계산"""
        
        if len(robot_markers_cam) == 0:
            return None, None
        
        # 검출된 마커들의 카메라 좌표
        detected_markers = {}
        for marker_id, data in robot_markers_cam.items():
            detected_markers[marker_id] = data['position']  # 카메라 좌표계에서의 위치
        
        # 방법 1: 단순 평균 (2개 이상)
        if len(detected_markers) >= 2:
            robot_center_cam = self.estimate_center_from_partial_markers(detected_markers)
            robot_rotation_cam = self.calculate_robot_rotation_simple(robot_markers_cam)
            return robot_center_cam, robot_rotation_cam
        
        # 방법 2: 1개 마커만 있을 때 - 해당 마커에서 중심까지의 벡터 계산
        elif len(detected_markers) == 1:
            marker_id = list(detected_markers.keys())[0]
            marker_pos_cam = detected_markers[marker_id]
            marker_rotation_cam = robot_markers_cam[marker_id]['rotation']
            
            # 마커 좌표계에서 로봇 중심까지의 벡터
            local_offset = -self.robot_markers_local[marker_id]  # 마커→중심 벡터
            
            # 마커의 회전을 적용하여 카메라 좌표계로 변환
            center_offset_cam = marker_rotation_cam @ local_offset
            robot_center_cam = marker_pos_cam + center_offset_cam
            
            return robot_center_cam, marker_rotation_cam
        
        return None, None

    def estimate_center_from_partial_markers(self, detected_markers):
        """부분적으로 검출된 마커들로부터 로봇 중심 추정"""
        
        marker_ids = list(detected_markers.keys())
        
        # 케이스별 최적화된 중심 계산
        if len(marker_ids) == 4:
            # 모든 마커 검출 - 단순 평균
            return np.mean(list(detected_markers.values()), axis=0)
        
        elif len(marker_ids) == 3:
            # 3개 마커 - 가중 평균으로 더 정확한 추정
            return self.estimate_center_from_three_markers(detected_markers)
        
        elif len(marker_ids) == 2:
            # 2개 마커 - 기하학적 관계 이용
            return self.estimate_center_from_two_markers(detected_markers)
        
        else:
            # 1개 마커는 위에서 처리됨
            return list(detected_markers.values())[0]

    def estimate_center_from_three_markers(self, detected_markers):
        """3개 마커로부터 중심 추정"""
        marker_ids = list(detected_markers.keys())
        
        # 없는 마커 찾기
        missing_marker = None
        for i in range(4):
            if i not in marker_ids:
                missing_marker = i
                break
        
        # 대각선 관계 이용해서 없는 마커 위치 추정
        if missing_marker is not None:
            estimated_missing = self.estimate_missing_marker_position(detected_markers, missing_marker)
            if estimated_missing is not None:
                # 4개 마커로 중심 계산
                all_positions = list(detected_markers.values()) + [estimated_missing]
                return np.mean(all_positions, axis=0)
        
        # 추정 실패시 단순 평균
        return np.mean(list(detected_markers.values()), axis=0)

    def estimate_center_from_two_markers(self, detected_markers):
        """2개 마커로부터 중심 추정"""
        marker_ids = list(detected_markers.keys())
        
        # 대각선 관계인지 확인
        if self.are_diagonal_markers(marker_ids):
            # 대각선 마커들의 중점이 로봇 중심
            return np.mean(list(detected_markers.values()), axis=0)
        
        else:
            # 인접한 마커들 - 기하학적 관계 이용
            return self.estimate_center_from_adjacent_markers(detected_markers, marker_ids)

    def are_diagonal_markers(self, marker_ids):
        """두 마커가 대각선 관계인지 확인"""
        diagonal_pairs = [(0, 2), (1, 3)]  # 대각선 쌍들
        marker_set = set(marker_ids)
        
        for pair in diagonal_pairs:
            if set(pair) == marker_set:
                return True
        return False

    def estimate_center_from_adjacent_markers(self, detected_markers, marker_ids):
        """인접한 2개 마커로부터 중심 추정"""
        
        # 알려진 로컬 좌표계에서의 관계 이용
        positions = list(detected_markers.values())
        pos1, pos2 = positions[0], positions[1]
        
        id1, id2 = marker_ids[0], marker_ids[1]
        
        # 로컬 좌표계에서의 관계
        local1 = self.robot_markers_local[id1]
        local2 = self.robot_markers_local[id2]
        local_center = np.array([0, 0, 0])  # 로봇 중심
        
        # 2D 평면에서의 기하학적 관계 이용 (Z=0 가정)
        # local1 + t1 * direction1 = center
        # local2 + t2 * direction2 = center
        # 이를 카메라 좌표계로 변환
        
        # 단순한 방법: 두 마커의 중점에서 기하학적 오프셋 적용
        midpoint = (pos1 + pos2) / 2
        
        # 로컬 좌표계에서의 중점과 실제 중심의 차이
        local_midpoint = (local1 + local2) / 2
        local_offset = local_center - local_midpoint
        
        # 마커들의 방향을 추정해서 오프셋 적용 (간단한 근사)
        direction = pos2 - pos1
        direction_norm = direction / np.linalg.norm(direction)
        
        # 로컬 오프셋을 카메라 좌표계로 근사 변환
        # (정확한 회전 없이 대략적인 스케일만 적용)
        camera_scale = np.linalg.norm(direction) / np.linalg.norm(local2 - local1)
        estimated_offset = local_offset * camera_scale
        
        return midpoint + estimated_offset

    def estimate_missing_marker_position(self, detected_markers, missing_id):
        """대각선 관계를 이용해서 없는 마커 위치 추정"""
        
        # 대각선 관계: 0↔2, 1↔3
        diagonal_map = {0: 2, 1: 3, 2: 0, 3: 1}
        diagonal_partner = diagonal_map.get(missing_id)
        
        if diagonal_partner in detected_markers:
            # 대각선 파트너가 있으면, 나머지 두 마커로부터 추정
            partner_pos = detected_markers[diagonal_partner]
            other_markers = {k: v for k, v in detected_markers.items() if k != diagonal_partner}
            
            if len(other_markers) >= 2:
                # 평행사변형 관계 이용
                other_positions = list(other_markers.values())
                # missing = partner + (other1 - other2) or similar geometric relation
                
                # 로컬 좌표계에서의 관계를 이용한 추정
                return self.estimate_by_parallelogram_rule(partner_pos, other_positions, missing_id, diagonal_partner)
        
        return None

    def estimate_by_parallelogram_rule(self, diagonal_pos, other_positions, missing_id, diagonal_id):
        """평행사변형 법칙으로 missing marker 위치 추정"""
        
        if len(other_positions) < 2:
            return None
        
        # 로컬 좌표계에서의 벡터 관계
        local_missing = self.robot_markers_local[missing_id]
        local_diagonal = self.robot_markers_local[diagonal_id]
        local_center = np.array([0, 0, 0])
        
        # 대각선 벡터
        local_diag_vector = local_diagonal - local_missing
        
        # 카메라 좌표계에서 대각선 벡터의 반대 방향으로 추정
        # (간단한 근사: 스케일과 방향만 고려)
        other_center = np.mean(other_positions, axis=0)
        camera_center_approx = (diagonal_pos + other_center) / 2
        
        # 대각선 관계로 missing marker 추정
        estimated_missing = 2 * camera_center_approx - diagonal_pos
        
        return estimated_missing
        
    def draw_robot_coordinate_axes(self, image):
        """로봇 좌표축을 draw_coordinate_axes 함수를 사용해서 그리기"""
        
        if self.H_cam2robot is None:
            return
        
        try:
            # H_cam2robot에서 rvec, tvec 추출
            robot_rotation = self.H_cam2robot[0:3, 0:3]  # 회전 행렬
            robot_translation = self.H_cam2robot[0:3, 3]  # 이동 벡터
            
            # 회전 행렬 → rodrigues 벡터 변환
            robot_rvec, _ = cv2.Rodrigues(robot_rotation)
            robot_tvec = robot_translation
            
            # ✅ 기존 draw_coordinate_axes 함수 사용
            self.draw_coordinate_axes(image, robot_rvec, robot_tvec, length=0.08)
            
            self.get_logger().info("✅ Robot coordinate axes drawn successfully")
            
        except Exception as e:
            self.get_logger().warn(f"❌ Failed to draw robot coordinate axes: {e}")
            
    def draw_coordinate_axes(self, image, rvec, tvec, length=0.03):
        """마커의 좌표축을 이미지에 그리기"""
        try:
            # 축 포인트들 정의 (마커 중심에서 각 축 방향으로)
            axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            
            # 3D 점들을 이미지로 투영
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                        self.camera_matrix, self.dist_coeffs)
            
            imgpts = np.int32(imgpts).reshape(-1,2)
            
            # 원점
            origin = tuple(imgpts[0])
            
            # X축 (빨간색)
            cv2.arrowedLine(image, origin, tuple(imgpts[1]), (0,0,255), 2)
            # Y축 (초록색)  
            cv2.arrowedLine(image, origin, tuple(imgpts[2]), (0,255,0), 2)
            # Z축 (파란색)
            cv2.arrowedLine(image, origin, tuple(imgpts[3]), (255,0,0), 2)
            
        except Exception as e:
            pass  # 조용히 무시

def main(args=None):
    rclpy.init(args=args)
    detector = D435ArucoDetector()
    
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