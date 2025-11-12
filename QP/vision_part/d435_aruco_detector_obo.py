#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from threading import Lock
import time
import yaml
import os
from scipy.spatial.transform import Rotation as R

class D435ArUcoDetector(Node):
    def __init__(self):
        super().__init__('d435_aruco_detector')

        # ROS2
        self.bridge = CvBridge()
        self.rectangle_center_pub = self.create_publisher(PoseStamped, '/aruco_rectangle_center', 10)

        # ArUco 세팅
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.05  # [m]

        # 마커의 로컬 좌표 (로봇 기준)
        self.marker_positions_local = {
            0: np.array([0.3, 0.135, 0.0]),
            1: np.array([0.3, -0.135, 0.0]),
            2: np.array([-0.3, -0.135, 0.0]),
            3: np.array([-0.3, 0.135, 0.0])
        }

        # 상태 저장용
        self.detection_lock = Lock()
        self.latest_detections = {}
        self.camera_info = {}
        self.camera_extrinsics = {}

        # 🟩 최신 마커 정보 저장용 변수
        self.last_marker_positions = {}
        self.last_marker_rotations = {}
        self.last_detected_ids = []

        # Kalman filter
        self.kf = self.init_kalman_filter(dt=0.1)
        self.kalman_initialized = False

        # Camera extrinsics 불러오기
        self.load_camera_calibrations()
        self.setup_camera_subscribers()

        # 🟩 타이머 콜백 -> 내부에서 latest 정보를 사용
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.latest_rectangle_center = None
        self.center_timestamp = 0.0

        self.get_logger().info("D435 ArUco Detector (with Kalman) initialized")

    # -------------------------------
    # Kalman filter helpers
    # -------------------------------
    def init_kalman_filter(self, dt=0.1):
        """
        6-state Kalman: [x,y,z, vx,vy,vz], 3-measurement: [x,y,z]
        dt: time step in seconds
        """
        kf = cv2.KalmanFilter(6, 3, 0)  # stateDim=6, measDim=3
        # Transition matrix
        # [ I3  dt*I3 ]
        # [ 0    I3   ]
        A = np.eye(6, dtype=np.float32)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        kf.transitionMatrix = A

        # Measurement matrix: meas = H * state, H picks position
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        kf.measurementMatrix = H

        # Covariances
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

        # initial state (zero)
        kf.statePost = np.zeros((6,1), dtype=np.float32)
        return kf

    def kalman_predict(self):
        """Predict step. Returns predicted position as np.array([x,y,z])"""
        try:
            state = self.kf.predict()  # returns 6x1 float32
            pos = state[0:3].reshape(3)
            return pos.astype(float)
        except Exception as e:
            self.get_logger().error(f"Kalman predict error: {e}")
            return None

    def kalman_correct(self, measurement):
        """Correct step with 3-vector measurement (x,y,z)"""
        if measurement is None:
            return
        meas = np.asarray(measurement, dtype=np.float32).reshape(3,1)
        try:
            if not self.kalman_initialized:
                # initialize position, zero velocity
                self.kf.statePost = np.zeros((6,1), dtype=np.float32)
                self.kf.statePost[0:3,0] = meas.reshape(3)
                self.kf.statePost[3:6,0] = 0.0
                self.kalman_initialized = True
                self.get_logger().debug("Kalman initialized with first measurement")
                return
            self.kf.correct(meas)
        except Exception as e:
            self.get_logger().error(f"Kalman correct error: {e}")

    # -------------------------------
    # Calibration / subscribers
    # -------------------------------
    def load_camera_calibrations(self):
        """multi_camera_calibration.yaml 또는 개별 extrinsic 파일에서 읽음"""
        self.camera_extrinsics = {}

        # 우선 통합 파일 시도
        multi_path = 'multi_camera_calibration.yaml'
        if os.path.exists(multi_path):
            try:
                with open(multi_path, 'r') as f:
                    data = yaml.safe_load(f)
                for name, cam in data.get('cameras', {}).items():
                    pos = np.array(cam['position'], dtype=float)
                    rot = np.array(cam['rotation_matrix'], dtype=float)
                    self.camera_extrinsics[name] = {'position': pos, 'rotation': rot}
                self.get_logger().info(f"Loaded multi-camera calibration from {multi_path}")
                return
            except Exception as e:
                self.get_logger().warn(f"Failed to read {multi_path}: {e}")

        # 개별 파일 시도
        cam_list = ['camera1', 'camera2', 'camera3']
        loaded = 0
        for name in cam_list:
            fn = f'{name}_extrinsic.yaml'
            if os.path.exists(fn):
                try:
                    with open(fn, 'r') as f:
                        cam = yaml.safe_load(f)
                    pos = np.array(cam['position'], dtype=float)
                    rot = np.array(cam['rotation_matrix'], dtype=float)
                    self.camera_extrinsics[name] = {'position': pos, 'rotation': rot}
                    loaded += 1
                except Exception as e:
                    self.get_logger().warn(f"Failed to load {fn}: {e}")

        if loaded == 0:
            self.get_logger().warn("No extrinsic files found: using default example extrinsics (please calibrate)")
            self.setup_default_extrinsics()
        else:
            self.get_logger().info(f"Loaded extrinsics for {loaded} cameras")

    def setup_default_extrinsics(self):
        self.camera_extrinsics = {
            'camera1': {'position': np.array([0.0, -1.5, 1.2]), 'rotation': np.eye(3)},
            'camera2': {'position': np.array([1.3, -0.75, 1.2]), 'rotation': self.rotation_from_euler(0,0,-np.pi/6)},
            'camera3': {'position': np.array([-1.3, -0.75, 1.2]), 'rotation': self.rotation_from_euler(0,0,np.pi/6)}
        }

    def rotation_from_euler(self, roll, pitch, yaw):
        return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    def setup_camera_subscribers(self):
        available_cameras = list(self.camera_extrinsics.keys())
        for camera_name in available_cameras:
            # 기본 토픽 패턴 (필요하면 사용자 환경에 맞게 수정)
            image_topic = f'/{camera_name}/{camera_name}/color/image_raw'
            info_topic = f'/{camera_name}/{camera_name}/color/camera_info'

            self.create_subscription(Image, image_topic, lambda msg, name=camera_name: self.image_callback(msg, name), 10)
            self.create_subscription(CameraInfo, info_topic, lambda msg, name=camera_name: self.camera_info_callback(msg, name), 10)
            self.get_logger().info(f"Subscribed to {image_topic} and {info_topic}")

    # -------------------------------
    # Callbacks: CameraInfo, Image
    # -------------------------------
    def camera_info_callback(self, msg: CameraInfo, camera_name):
        try:
            K = np.array(msg.k, dtype=float).reshape(3,3)
            d = np.array(msg.d, dtype=float)
            # fallback width/height if not present
            w = int(msg.width) if hasattr(msg, 'width') else 640
            h = int(msg.height) if hasattr(msg, 'height') else 480
            self.camera_info[camera_name] = {'camera_matrix': K, 'dist_coeffs': d, 'width': w, 'height': h}
            # self.get_logger().info(f"Camera info: {self.camera_info}")
        except Exception as e:
            self.get_logger().error(f"camera_info_callback error for {camera_name}: {e}")

    def image_callback(self, msg: Image, camera_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = self.detect_aruco_markers(cv_image, camera_name)
            vis_image = self.draw_detections_with_center(cv_image.copy(), detections, camera_name)

            # --- 새 프레임 기준으로 완전 초기화 ---
            marker_positions_repr = {}
            marker_rotations_repr = {}
            detected_ids = []

            for d in detections:
                mid = d['id']
                marker_positions_repr[mid] = d['world_tvec']
                marker_rotations_repr[mid] = d['world_R']
                detected_ids.append(mid)

            with self.detection_lock:
                # ✅ 덮어쓰기: 이전 프레임 내용 제거
                self.last_marker_positions = marker_positions_repr
                self.last_marker_rotations = marker_rotations_repr
                self.last_detected_ids = detected_ids

            cv2.imshow(f'{camera_name}_aruco_detection', vis_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"image_callback error for {camera_name}: {e}")


    # -------------------------------
    # ArUco detection and transform
    # -------------------------------
    def detect_aruco_markers(self, image, camera_name):
        """각 이미지에서 마커 감지 후 카메라 좌표->월드 좌표(회전 포함) 변환"""
        detections = []
        if camera_name not in self.camera_info:
            return detections

        cam_info = self.camera_info[camera_name]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return detections

        # pose estimate
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, cam_info['camera_matrix'], cam_info['dist_coeffs'])
        except Exception as e:
            self.get_logger().error(f"estimatePoseSingleMarkers error: {e}")
            return detections

        # extrinsics for camera -> world
        extrinsic = self.camera_extrinsics.get(camera_name, None)
        if extrinsic is None:
            self.get_logger().warn(f"No extrinsics for {camera_name}, skipping world transform")
            return detections

        R_wc = extrinsic['rotation']   # camera -> world rotation
        t_wc = extrinsic['position']   # camera position in world
        if np.linalg.norm(t_wc) < 1e-6:
            self.get_logger().warn(f"Extrinsic position for {camera_name} is near zero. Verify extrinsic file.")

        for i, mid in enumerate(ids.flatten()):
            if int(mid) not in self.marker_positions_local:
                continue
            cam_t = tvecs[i].reshape(3)   # marker position in camera coordinates
            if np.linalg.norm(cam_t) < 1e-6:
                self.get_logger().warn(f"Marker {mid} produced near-zero tvec in {camera_name}; skip")
                continue
            cam_r = rvecs[i].reshape(3)   # rvec (Rodrigues) in camera coords

            # compute marker rotation matrix in camera frame
            try:
                R_cam_marker, _ = cv2.Rodrigues(cam_r)  # rotation marker <- camera (i.e., rotation of marker in camera frame)
            except Exception:
                R_cam_marker = np.eye(3)

            # Convert marker pose to world frame:
            # world_pos = R_wc @ cam_t + t_wc
            world_pos = (R_wc @ cam_t) + t_wc
            # marker rotation in world: R_world_marker = R_wc @ R_cam_marker
            R_world_marker = R_wc @ R_cam_marker
            print(f"Detected marker {mid} in {camera_name}: cam_t={cam_t}, world_t={world_pos}, R_world_marker=\n{R_world_marker}")
            detections.append({
                'id': int(mid),
                'cam_tvec': cam_t,
                'cam_rvec': cam_r,
                'world_tvec': world_pos,
                'world_R': R_world_marker,
                'corners': corners[i],
                'camera_name': camera_name
            })

        return detections

    # -------------------------------
    # Visualization helpers
    # -------------------------------
    def draw_detections_with_center(self, image, detections, camera_name):
        # draw markers
        for d in detections:
            corners = d['corners']
            mid = d['id']
            try:
                cv2.aruco.drawDetectedMarkers(image, [corners], np.array([mid]))
            except Exception:
                pass
            corner = corners[0][0].astype(int)
            cv2.putText(image, f"ID:{mid}", (corner[0], corner[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # project center if available and recent
        if self.latest_rectangle_center is not None and (time.time() - self.center_timestamp) < 0.5:
            img_pt = self.project_world_to_image(self.latest_rectangle_center, camera_name)
            if img_pt is not None:
                self.draw_rectangle_center_on_image(image, img_pt, self.latest_rectangle_center)

        # draw lines between detected markers' image centers
        if len(detections) >= 2:
            self.draw_rectangle_outline(image, detections)

        return image

    def project_world_to_image(self, world_point, camera_name):
        if camera_name not in self.camera_extrinsics or camera_name not in self.camera_info:
            return None
        extrinsic = self.camera_extrinsics[camera_name]
        R_wc = extrinsic['rotation']
        t_wc = extrinsic['position']

        # world -> cam: R_cw = R_wc.T, t_cw = -R_cw @ t_wc
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        wp = np.array(world_point).reshape(3,1)
        cam_pt = R_cw @ wp + t_cw.reshape(3,1)
        cam_pt = cam_pt.flatten()
        if cam_pt[2] <= 0:
            return None

        cam_info = self.camera_info[camera_name]
        img_pts, _ = cv2.projectPoints(cam_pt.reshape(1,1,3), np.zeros(3), np.zeros(3), cam_info['camera_matrix'], cam_info['dist_coeffs'])
        x, y = img_pts[0][0]
        if 0 <= x < cam_info['width'] and 0 <= y < cam_info['height']:
            return (int(x), int(y))
        return None

    def draw_rectangle_center_on_image(self, image, center_point, world_center):
        cx, cy = int(center_point[0]), int(center_point[1])
        cv2.circle(image, (cx, cy), 8, (0,0,255), -1)
        cv2.line(image, (cx-15, cy), (cx+15, cy), (0,0,255), 2)
        cv2.line(image, (cx, cy-15), (cx, cy+15), (0,0,255), 2)
        coord_text = f"({world_center[0]:.2f},{world_center[1]:.2f},{world_center[2]:.2f})"
        cv2.putText(image, coord_text, (cx+10, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    def draw_rectangle_outline(self, image, detections):
        # compute image centers per id
        marker_centers = {}
        for d in detections:
            corners = d['corners'][0]
            center = np.mean(corners, axis=0).astype(int)
            marker_centers[d['id']] = tuple(center)
        order = [0,1,2,3,0]
        for i in range(len(order)-1):
            a,b = order[i], order[i+1]
            if a in marker_centers and b in marker_centers:
                cv2.line(image, marker_centers[a], marker_centers[b], (255,0,255), 2)
    
    def timer_callback(self):
        """주기적으로 칼만 필터 업데이트 및 중심 계산"""
        with self.detection_lock:
            marker_positions = dict(self.last_marker_positions)
            marker_rotations = dict(self.last_marker_rotations)
            detected_ids = list(self.last_detected_ids)

        self.calculate_rectangle_center(marker_positions, marker_rotations, detected_ids)
    # -------------------------------
    # Fusion & center computation
    # -------------------------------

    def calculate_rectangle_center(self, marker_positions_repr, marker_rotations_repr, detected_ids):
        if not detected_ids:
            # --- 아무 마커도 안 보일 때 Kalman 예측 ---
            if self.kalman_initialized:
                self.kf.predict()
                predicted_pos = self.kf.statePre[:3].reshape(3)
                self.latest_rectangle_center = predicted_pos
                self.center_timestamp = time.time()
                self.publish_rectangle_center(predicted_pos, 0)
                print("[Kalman] Predict-only (no markers):", predicted_pos)
            return

        fused_positions = []
        fused_rotations = []

        for mid in detected_ids:
            if mid not in marker_positions_repr or mid not in marker_rotations_repr:
                continue

            world_marker_pos = marker_positions_repr[mid]
            world_R_marker = marker_rotations_repr[mid]
            local_marker_pos = self.marker_positions_local[mid]

            # 중심 계산
            local_offset = -local_marker_pos
            world_offset = world_R_marker @ local_offset
            center_world = world_marker_pos + world_offset

            fused_positions.append(center_world)
            fused_rotations.append(world_R_marker)

        if len(fused_positions) == 0:
            if self.kalman_initialized:
                self.kf.predict()
                predicted_pos = self.kf.statePre[:3].reshape(3)
                self.latest_rectangle_center = predicted_pos
                self.center_timestamp = time.time()
                self.publish_rectangle_center(predicted_pos, 0)
                print("[Kalman] Predict-only (no valid markers):", predicted_pos)
            return

        # 회전 평균
        quats = np.array([R.from_matrix(Rm).as_quat() for Rm in fused_rotations])
        quat_mean = np.mean(quats, axis=0)
        quat_mean /= np.linalg.norm(quat_mean)
        R_fused = R.from_quat(quat_mean).as_matrix()

        # 회전 이상치 제거
        angles = [np.linalg.norm(R.from_matrix(Rm.T @ R_fused).as_rotvec()) for Rm in fused_rotations]
        valid_mask = np.array(angles) < np.deg2rad(15)
        valid_positions = np.array(fused_positions)[valid_mask] if np.any(valid_mask) else np.array(fused_positions)

        # 위치 평균
        center_world = np.mean(valid_positions, axis=0)

        # 칼만 필터 업데이트
        if not self.kalman_initialized:
            self.kf.statePost[:3] = center_world.reshape(3, 1)
            self.kf.statePost[3:] = 0
            self.kalman_initialized = True
            print("[Kalman] initialized with:", center_world)

        self.kalman_correct(center_world)
        filtered_pos = self.kf.statePost[:3].reshape(3)

        self.latest_rectangle_center = filtered_pos
        self.center_timestamp = time.time()
        self.publish_rectangle_center(filtered_pos, len(valid_positions))

        print(f"[INFO] Rectangle center: ({filtered_pos[0]:.3f}, {filtered_pos[1]:.3f}, {filtered_pos[2]:.3f}) from {len(valid_positions)} markers")


    # -------------------------------
    # Publishing
    # -------------------------------
    def publish_rectangle_center(self, center, num_markers):
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = center.tolist()
        msg.pose.orientation.w = 1.0
        self.rectangle_center_pub.publish(msg)
        self.get_logger().info(f'Rectangle center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) from {num_markers} markers')

# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = D435ArUcoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
