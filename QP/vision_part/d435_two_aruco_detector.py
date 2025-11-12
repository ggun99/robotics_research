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

class D435MultiCameraArucoDetector(Node):
    def __init__(self):
        super().__init__('d435_multi_camera_aruco_detector')
        
        # 로봇 마커들 (0~3번) - 로컬 좌표계
        self.robot_markers_local = {
            0: np.array([0.3, 0.135, 0.0]),
            1: np.array([0.3, -0.135, 0.0]),
            2: np.array([-0.3, -0.135, 0.0]),
            3: np.array([-0.3, 0.135, 0.0])
        }
    
        self.marker_length = 0.075  # 마커 크기
        
        # 월드 좌표계 설정 상태
        self.world_established = False
        
        # 🆕 다중 카메라 설정
        self.cameras = {
            'camera1': {
                'topics': {
                    'image': '/camera1/camera1/color/image_raw',
                    'info': '/camera1/camera1/color/camera_info'
                },
                'camera_matrix': None,
                'dist_coeffs': None,
                'info_received': False,
                'H_world2cam': None,
                'latest_detections': {},
                'detection_timestamp': None
            },
            'camera3': {
                'topics': {
                    'image': '/camera3/camera3/color/image_raw', 
                    'info': '/camera3/camera3/color/camera_info'
                },
                'camera_matrix': None,
                'dist_coeffs': None,
                'info_received': False,
                'H_world2cam': None,
                'latest_detections': {},
                'detection_timestamp': None
            }
        }
        
        # 로봇 중심 추적 (통합된 결과)
        self.latest_robot_center = None
        self.center_timestamp = None
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 변환 행렬들 (통합된 결과)
        self.H_world2cam_global = None  # 글로벌 월드→카메라 (첫 번째로 10번 마커를 본 카메라 기준)
        self.H_cam2robot = None         # 카메라→로봇 중심
        self.H_world2robot = None       # 월드→로봇
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 🆕 각 카메라별 구독자 설정
        self.setup_camera_subscriptions()
        
        # 발행자
        self.robot_center_pub = self.create_publisher(
            PoseStamped, '/robot_center', 10
        )
        
        # 🆕 주기적으로 다중 카메라 데이터 융합
        self.create_timer(0.1, self.fuse_multi_camera_data)  # 10Hz
        
        self.get_logger().info("D435 Multi-Camera ArUco Detector initialized")

    def setup_camera_subscriptions(self):
        """각 카메라별 구독자 설정 (간단한 방법)"""
        available_cameras = list(self.cameras.keys())
        
        for camera_name in available_cameras:
            # 기본 토픽 패턴
            image_topic = f'/{camera_name}/{camera_name}/color/image_raw'
            info_topic = f'/{camera_name}/{camera_name}/color/camera_info'

            # ✅ lambda에서 기본값 사용해서 바인딩 문제 해결
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

    def create_camera_subscription(self, camera_name, camera_config):
        """개별 카메라 구독자 생성"""
        
        # 이미지 구독자
        def image_callback_wrapper(msg):
            return self.image_callback(msg, camera_name)
        
        def camera_info_callback_wrapper(msg):
            return self.camera_info_callback(msg, camera_name)
        
        self.create_subscription(
            Image, 
            camera_config['topics']['image'], 
            image_callback_wrapper,
            10
        )
        
        self.create_subscription(
            CameraInfo, 
            camera_config['topics']['info'],
            camera_info_callback_wrapper,
            10
        )
        
        self.get_logger().info(f"📷 Created subscriptions for {camera_name}")

    def camera_info_callback(self, msg, camera_name):
        """카메라 내부 매개변수 수신"""
        camera_config = self.cameras[camera_name]
        
        if not camera_config['info_received']:
            camera_config['camera_matrix'] = np.array(msg.k).reshape(3, 3)
            camera_config['dist_coeffs'] = np.array(msg.d)
            camera_config['info_received'] = True
            self.get_logger().info(f"📷 {camera_name} intrinsics received")

    def image_callback(self, msg, camera_name):
        """각 카메라별 이미지 처리 및 ArUco 검출"""
        camera_config = self.cameras[camera_name]
        
        if not camera_config['info_received']:
            return
            
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ArUco 마커 검출
        detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            # 검출 실패시 타임스탬프만 업데이트
            camera_config['latest_detections'] = {}
            camera_config['detection_timestamp'] = time.time()
            return

        # 포즈 추정 (해당 카메라 좌표계)
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, 
                camera_config['camera_matrix'], 
                camera_config['dist_coeffs']
            )
        except Exception as e:
            self.get_logger().error(f"{camera_name} pose estimation failed: {e}")
            return

        # 검출된 마커 정리
        detected_markers = {}
        for i, marker_id in enumerate(ids.flatten()):
            detected_markers[marker_id] = {
                'cam_tvec': tvecs[i].reshape(3),
                'cam_rvec': rvecs[i].reshape(3),
                'corners': corners[i],
                'camera_name': camera_name  # 🆕 카메라 정보 추가
            }

        # 🆕 해당 카메라의 월드 좌표계 설정
        self.establish_world_coordinate_system(detected_markers, camera_config, camera_name, msg.header.stamp)

        # 🆕 검출 결과 저장
        camera_config['latest_detections'] = detected_markers
        camera_config['detection_timestamp'] = time.time()

        # 시각화 (각 카메라별)
        self.visualize_results(cv_image, detected_markers, camera_name)

    def establish_world_coordinate_system(self, detected_markers, camera_config, camera_name, timestamp):
        """통합 월드 좌표계 설정 (10번 마커 기준)"""
        
        for marker_id, data in detected_markers.items():
            if marker_id == 10:
                rvec = data['cam_rvec']
                tvec = data['cam_tvec']
                
                # 해당 카메라에서의 월드 좌표계 설정
                H_cam2world = np.eye(4)
                R_matrix, _ = cv2.Rodrigues(rvec)
                H_cam2world[0:3, 0:3] = R_matrix
                H_cam2world[0:3, 3] = tvec
                
                H_world2cam = np.linalg.inv(H_cam2world)
                camera_config['H_world2cam'] = H_world2cam
                
                # 🔧 통합된 월드 좌표계 설정 (첫 번째 검출 카메라 기준)
                if not self.world_established:
                    self.H_world2cam_global = H_world2cam
                    self.world_established = True
                    # 🌍 하나의 월드 프레임 브로드캐스트
                    self.broadcast_world_transform(H_world2cam, timestamp, camera_name)
                    self.get_logger().info(f"✅ Unified world coordinate system established by {camera_name}")
                
                break

    def fuse_multi_camera_data(self):
        """다중 카메라 데이터 융합"""
        
        if not self.world_established:
            return
        
        # 🆕 모든 카메라에서 검출된 로봇 마커들 수집
        all_robot_markers = {}
        current_time = time.time()
        
        for camera_name, camera_config in self.cameras.items():
            # 최근 검출 데이터만 사용 (1초 이내)
            if (camera_config['detection_timestamp'] is not None and 
                current_time - camera_config['detection_timestamp'] < 1.0 and
                camera_config['H_world2cam'] is not None):
                
                detections = camera_config['latest_detections']
                
                for marker_id, data in detections.items():
                    if marker_id in self.robot_markers_local:
                        # 해당 카메라 좌표계 → 월드 좌표계 변환
                        world_position = self.transform_to_world_coordinates(
                            data['cam_tvec'], 
                            camera_config['H_world2cam']
                        )
                        
                        if world_position is not None:
                            # 같은 마커가 여러 카메라에서 검출된 경우 평균 사용
                            if marker_id in all_robot_markers:
                                # 기존 위치와 평균
                                existing_pos = all_robot_markers[marker_id]['world_position']
                                count = all_robot_markers[marker_id]['count']
                                new_pos = (existing_pos * count + world_position) / (count + 1)
                                all_robot_markers[marker_id]['world_position'] = new_pos
                                all_robot_markers[marker_id]['count'] += 1
                                all_robot_markers[marker_id]['cameras'].append(camera_name)
                            else:
                                all_robot_markers[marker_id] = {
                                    'world_position': world_position,
                                    'cam_tvec': data['cam_tvec'], 
                                    'cam_rvec': data['cam_rvec'],
                                    'camera_name': camera_name,
                                    'cameras': [camera_name],
                                    'count': 1
                                }

        # 🤖 융합된 로봇 중심 계산
        if len(all_robot_markers) > 0:
            self.calculate_fused_robot_center(all_robot_markers)

    def transform_to_world_coordinates(self, cam_position, H_world2cam):
        """카메라 좌표 → 월드 좌표 변환"""
        try:
            H_cam2world = np.linalg.inv(H_world2cam)
            cam_homo = np.append(cam_position, 1)
            world_homo = H_cam2world @ cam_homo
            return world_homo[:3]
        except Exception:
            return None

    def calculate_fused_robot_center(self, all_robot_markers):
        """융합된 로봇 중심 계산"""
        
        # 월드 좌표계에서의 로봇 마커 위치들
        world_positions = [data['world_position'] for data in all_robot_markers.values()]
        robot_center_world = np.mean(world_positions, axis=0)
        
        # 로봇 회전 계산 (첫 번째 카메라의 데이터 사용 - 단순화)
        first_marker_data = list(all_robot_markers.values())[0]
        first_camera_name = first_marker_data['camera_name']
        
        # 해당 카메라에서의 회전 행렬들 사용
        robot_markers_cam = {}
        for marker_id, data in all_robot_markers.items():
            if data['camera_name'] == first_camera_name:
                R_matrix, _ = cv2.Rodrigues(data['cam_rvec'])
                robot_markers_cam[marker_id] = {
                    'position': data['cam_tvec'],
                    'rotation': R_matrix
                }
        
        if len(robot_markers_cam) > 0:
            robot_rotation_cam = self.calculate_robot_rotation_simple(robot_markers_cam)
            
            # H_world2robot 계산
            H_world2robot = np.eye(4)
            H_world2robot[0:3, 0:3] = robot_rotation_cam
            H_world2robot[0:3, 3] = robot_center_world
            self.H_world2robot = H_world2robot
            
            # TF 브로드캐스트
            timestamp = self.get_clock().now().to_msg()
            self.broadcast_robot_transform(H_world2robot, timestamp)
            
            # 상태 업데이트
            self.latest_robot_center = robot_center_world
            self.center_timestamp = time.time()
            
            # 로그
            camera_names = set()
            for data in all_robot_markers.values():
                camera_names.update(data['cameras'])
            
            self.get_logger().info(
                f"🤖 Fused robot center from {len(all_robot_markers)} markers "
                f"across cameras: {list(camera_names)}"
            )
            
            # 발행
            self.publish_robot_center(robot_center_world, len(all_robot_markers))

    # 🆕 기존 함수들을 그대로 유지 (약간 수정)
    def project_world_to_image_simple(self, world_point, camera_name=None):
        """월드 좌표를 특정 카메라의 이미지로 투영"""
        
        if camera_name is None:
            # 첫 번째 카메라 사용
            camera_name = list(self.cameras.keys())[0]
        
        camera_config = self.cameras[camera_name]
        H_world2cam = camera_config.get('H_world2cam')
        
        if H_world2cam is None:
            return None
        
        try:
            # 월드 → 카메라 변환
            world_point_homogeneous = np.append(world_point, 1)
            cam_point_homogeneous = H_world2cam @ world_point_homogeneous
            cam_point = cam_point_homogeneous[:3]
            
            if cam_point[2] <= 0:  # 카메라 뒤에 있음
                return None
            
            # 카메라 → 이미지 투영
            image_points, _ = cv2.projectPoints(
                cam_point.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                camera_config['camera_matrix'], camera_config['dist_coeffs']
            )
            
            return image_points[0][0]
            
        except Exception as e:
            return None

    def publish_robot_center(self, center, num_markers):
        """로봇 중심 위치 발행"""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'  # 🌍 통합된 월드 프레임 사용
        
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = 1.0
        
        self.robot_center_pub.publish(msg)

    def broadcast_robot_transform(self, H_world2robot, timestamp):
        """로봇 좌표계 tf 브로드캐스트 (통합 월드 프레임 기준)"""
        try:
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = 'world'  # 🌍 통합된 월드 프레임 사용
            t.child_frame_id = 'robot_center'
            
            # 위치
            t.transform.translation.x = float(H_world2robot[0, 3])
            t.transform.translation.y = float(H_world2robot[1, 3])
            t.transform.translation.z = float(H_world2robot[2, 3])
            
            # 회전
            R_mat = H_world2robot[0:3, 0:3]
            rot = R.from_matrix(R_mat)
            quat = rot.as_quat()  # [x, y, z, w]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().warn(f"Robot transform broadcast failed: {e}")

    def broadcast_world_transform(self, H_world2cam, timestamp, camera_name):
        """통합된 월드 좌표계 tf 브로드캐스트 (하나의 world 프레임)"""
        try:
            H_cam2world = np.linalg.inv(H_world2cam)
            
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = f'{camera_name}_link'  # 기준 카메라 프레임
            t.child_frame_id = 'world'  # 🌍 통합된 하나의 월드 프레임
            
            # 위치
            t.transform.translation.x = float(H_cam2world[0, 3])
            t.transform.translation.y = float(H_cam2world[1, 3])
            t.transform.translation.z = float(H_cam2world[2, 3])
            
            # 회전
            R_mat = H_cam2world[0:3, 0:3]
            rot = R.from_matrix(R_mat)
            quat = rot.as_quat()
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            
            self.get_logger().info(f"🌍 Unified world frame established from {camera_name}_link")
            
        except Exception as e:
            self.get_logger().warn(f"World transform broadcast failed for {camera_name}: {e}")

    def calculate_robot_rotation_simple(self, robot_markers_cam):
        """간단한 아웃라이어 제거 + 회전 평균 (기존과 동일)"""
        
        if len(robot_markers_cam) <= 1:
            return list(robot_markers_cam.values())[0]['rotation']
        
        from scipy.spatial.transform import Rotation as R_scipy
        
        rotations_data = []
        for marker_id, data in robot_markers_cam.items():
            rot_matrix = data['rotation']
            r = R_scipy.from_matrix(rot_matrix)
            quat = r.as_quat()
            rotations_data.append((marker_id, quat, rot_matrix))
        
        reference_quat = rotations_data[0][1]
        valid_rotations = []
        
        for marker_id, quat, rot_matrix in rotations_data:
            dot_product = np.abs(np.dot(reference_quat, quat))
            angle_diff = 2 * np.arccos(np.clip(dot_product, 0, 1))
            
            if angle_diff <= 0.5:  # 약 30도 이내
                valid_rotations.append(quat)
            else:
                self.get_logger().warn(f"⚠️ Marker {marker_id} removed: angle diff {np.degrees(angle_diff):.1f}°")
        
        if len(valid_rotations) > 0:
            mean_quat = np.mean(valid_rotations, axis=0)
            mean_quat = mean_quat / np.linalg.norm(mean_quat)
            return R_scipy.from_quat(mean_quat).as_matrix()
        else:
            return rotations_data[0][2]
    
    def visualize_results(self, image, detected_markers, camera_name):
        """검출 결과 시각화 (카메라별)"""
        
        # 검출된 마커들 표시
        for marker_id, data in detected_markers.items():
            corners = data['corners'][0]
            center = np.mean(corners, axis=0).astype(int)
            
            if marker_id == 10:
                # 10번 마커 - 월드 원점
                color = (255, 0, 0)
                label = f"World{marker_id}"
                cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(image, label, tuple(center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                self.draw_coordinate_axes(image, data['cam_rvec'], data['cam_tvec'], 0.05, camera_name)

        # 상태 정보 표시
        world_status = "World: OK" if self.world_established else "World: No Marker 10"
        world_color = (0, 255, 0) if self.world_established else (0, 0, 255)
        cv2.putText(image, f"{camera_name} - {world_status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, world_color, 2)
        
        # 로봇 좌표 표시
        if self.latest_robot_center is not None:
            coord_text = f"Robot: ({self.latest_robot_center[0]:.2f}, {self.latest_robot_center[1]:.2f}, {self.latest_robot_center[2]:.2f})"
            cv2.putText(image, coord_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 🆕 카메라별 창 표시
        cv2.imshow(f'{camera_name} - ArUco Detection', image)
        cv2.waitKey(1)

    def draw_coordinate_axes(self, image, rvec, tvec, length, camera_name):
        """좌표축 그리기 (카메라별)"""
        camera_config = self.cameras[camera_name]
        
        try:
            axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                        camera_config['camera_matrix'], 
                                        camera_config['dist_coeffs'])
            
            imgpts = np.int32(imgpts).reshape(-1,2)
            origin = tuple(imgpts[0])
            
            # X축 (빨간색), Y축 (초록색), Z축 (파란색)
            cv2.arrowedLine(image, origin, tuple(imgpts[1]), (0,0,255), 2)
            cv2.arrowedLine(image, origin, tuple(imgpts[2]), (0,255,0), 2)
            cv2.arrowedLine(image, origin, tuple(imgpts[3]), (255,0,0), 2)
            
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    detector = D435MultiCameraArucoDetector()
    
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