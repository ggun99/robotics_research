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

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_length = 0.05  # [m]

        # 로컬 마커 좌표 (id: [x,y,z]) - 사용자 제공 값 사용
        self.marker_positions_local = {
            0: np.array([0.3, 0.135, 0.0]),
            1: np.array([0.3, -0.135, 0.0]),
            2: np.array([-0.3, -0.135, 0.0]),
            3: np.array([-0.3, 0.135, 0.0])
        }

        # 상태 저장
        self.detection_lock = Lock()
        # latest_detections: { camera_name: { 'timestamp': float, 'detections': [ {id, cam_tvec, cam_rvec, world_tvec, world_R (3x3)} ] , 'image', 'vis_image' } }
        self.latest_detections = {}
        self.camera_info = {}            # camera_name -> { camera_matrix, dist_coeffs, width, height }
        self.camera_extrinsics = {}      # camera_name -> { 'position': np.array(3), 'rotation': np.array(3x3) }

        # Kalman filter (3D position + velocity)
        self.kf = self.init_kalman_filter(dt=0.1)  # timer 주기(초)
        self.kalman_initialized = False

        # 캘리브레이션 로드
        self.load_camera_calibrations()

        # 구독자 생성
        self.setup_camera_subscribers()

        # 주기적으로 중심 계산 (10Hz)
        self.timer = self.create_timer(0.1, self.calculate_rectangle_center)
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

            with self.detection_lock:
                self.latest_detections[camera_name] = {
                    'timestamp': time.time(),
                    'detections': detections,
                    'image': cv_image,
                    'vis_image': vis_image
                }

            # 가시화 (선택)
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

    # -------------------------------
    # Fusion & center computation
    # -------------------------------
    def calculate_rectangle_center(self):
        # 1. 수집된 최신 검출 복사
        with self.detection_lock:
            detections_copy = {k: v.copy() for k,v in self.latest_detections.items()}

        # 2. 최근 time window 내 감지들만 사용
        now = time.time()
        time_window = 0.5  # seconds
        per_marker_worlds = {}  # id -> list of (world_pos, world_R, camera_name, timestamp)

        for cam_name, data in detections_copy.items():
            ts = data.get('timestamp', 0)
            if now - ts > time_window:
                continue
            for d in data.get('detections', []):
                mid = d['id']
                wp = np.array(d['world_tvec'], dtype=float).reshape(3)
                WR = np.array(d.get('world_R', np.eye(3)), dtype=float)
                per_marker_worlds.setdefault(mid, []).append({'pos': wp, 'R': WR, 'camera': d['camera_name'], 't': ts})

        # If no markers observed in this time window -> use Kalman predict
        if len(per_marker_worlds) == 0:
            if not self.kalman_initialized:
                self.get_logger().debug("No markers and kalman not initialized — skip publish")
                return
            pred = self.kalman_predict()
            if pred is not None:
                self.latest_rectangle_center = pred
                self.center_timestamp = time.time()
                # publish predicted position (num_markers=0)
                self.publish_rectangle_center(pred, 0)
                self.get_logger().debug("Published predicted center (no markers)")
            return

        # 3. 각 마커별로 중간값(median) 또는 중앙값 기반 대표 위치 선택 => outlier에 강함
        marker_positions_repr = {}
        marker_rotations_repr = {}
        for mid, obs in per_marker_worlds.items():
            pts = np.array([o['pos'] for o in obs])
            if pts.shape[0] == 1:
                med = pts[0]
            else:
                med = np.median(pts, axis=0)
            marker_positions_repr[mid] = med

            # rotation: 평균 rotation via simple SVD average
            Rs = np.array([o['R'] for o in obs])
            M = np.sum(Rs, axis=0)
            try:
                U, S, Vt = np.linalg.svd(M)
                R_avg = U @ Vt
                if np.linalg.det(R_avg) < 0:
                    Vt[-1,:] *= -1
                    R_avg = U @ Vt
            except Exception:
                R_avg = Rs[0]
            marker_rotations_repr[mid] = R_avg

        # 4. If >=2 markers: run Kabsch with simple outlier rejection (RANSAC-like)
        detected_ids = list(marker_positions_repr.keys())
        center_world = None

        if len(detected_ids) >= 2:
            # print("두개 이상")
            local_positions = np.array([self.marker_positions_local[mid] for mid in detected_ids])
            world_positions = np.array([marker_positions_repr[mid] for mid in detected_ids])

            R_opt, t_opt = self.robust_kabsch(local_positions, world_positions)
            center_world = t_opt

            # # --- 보정 & 퍼블리시 ---
            # self.kalman_correct(center_world)
            # filtered_pos = self.kf.statePost[0:3].reshape(3).astype(float)
            # self.latest_rectangle_center = filtered_pos
            # self.center_timestamp = time.time()
            # self.publish_rectangle_center(filtered_pos, len(detected_ids))

        elif len(detected_ids) == 1:
            # single marker fallback: use marker rotation if available to rotate offset
            mid = detected_ids[0]
            world_marker_pos = marker_positions_repr[mid]
            world_R_marker = marker_rotations_repr.get(mid, np.eye(3))
            local_marker_pos = self.marker_positions_local[mid]
            # local_center - local_marker_pos = -local_marker_pos
            local_offset = -local_marker_pos
            # world_offset = R_world_marker @ local_offset
            world_offset = world_R_marker @ local_offset
            center_world = world_marker_pos + world_offset

            # Kalman correct with this (noisy) measurement
            # self.kalman_correct(center_world)
            # filtered_pos = self.kf.statePost[0:3].reshape(3).astype(float) if self.kalman_initialized else center_world
            # self.latest_rectangle_center = filtered_pos
            # self.center_timestamp = time.time()
            # self.publish_rectangle_center(filtered_pos, 1)

        else:
            # should not reach here due to earlier empty check, but safe-guard
            return
        
        if center_world is not None:
            # --- 칼만 초기화 (처음 한 번만) ---
            if not self.kalman_initialized:
                self.kf.statePost[:3, 0] = center_world  # ✅ 이렇게만 해도 충분
                self.kf.statePost[3:, 0] = 0
                self.kalman_initialized = True
                print("[Kalman] initialized with first center:", center_world)

            # --- 이후 보정 및 퍼블리시 ---
            print(f"[DEBUG] center_world before kalman: {center_world}, from {len(detected_ids)} markers")

            self.kalman_correct(center_world)
            filtered_pos = self.kf.statePost[0:3].reshape(3).astype(float)
            self.latest_rectangle_center = filtered_pos
            self.center_timestamp = time.time()
            self.publish_rectangle_center(filtered_pos, len(detected_ids))


    def robust_kabsch(self, local_positions, world_positions, max_iters=5, error_thresh=0.08):
        """
        Kabsch + simple iterative outlier rejection:
         - compute R,t with all correspondences
         - compute residuals, drop worst if > error_thresh, repeat up to max_iters
        Returns R,t
        """
        L = local_positions.copy()
        W = world_positions.copy()
        if L.shape[0] == 0:
            return None, None

        best_R, best_t = None, None
        for it in range(max_iters):
            try:
                R_est, t_est = self.compute_optimal_transformation(L, W)
            except Exception as e:
                self.get_logger().error(f"Kabsch compute failed: {e}")
                return None, None

            # residuals
            transformed = (R_est @ L.T).T + t_est.reshape(1,3)
            residuals = np.linalg.norm(transformed - W, axis=1)
            max_err = np.max(residuals)
            mean_err = np.mean(residuals)
            self.get_logger().debug(f"Kabsch iter {it}: mean_err={mean_err:.4f}, max_err={max_err:.4f}")

            best_R, best_t = R_est, t_est

            # if all good, stop
            if max_err <= error_thresh or L.shape[0] <= 2:
                break

            # else drop the worst one and repeat
            worst_idx = int(np.argmax(residuals))
            L = np.delete(L, worst_idx, axis=0)
            W = np.delete(W, worst_idx, axis=0)

            if L.shape[0] < 2:
                break

        # final validation: if max_err still large, log warning
        transformed_final = (best_R @ local_positions.T).T + best_t.reshape(1,3)
        final_res = np.linalg.norm(transformed_final - world_positions, axis=1)
        final_max = np.max(final_res)
        if final_max > 0.2:
            self.get_logger().warn(f"Final transform has large residual {final_max:.3f} m")

        return best_R, best_t

    def compute_optimal_transformation(self, A, B):
        """
        Compute R, t s.t.  R @ A_i + t ~= B_i using Kabsch
        A, B are Nx3 arrays
        """
        assert A.shape == B.shape
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        Am = A - centroid_A
        Bm = B - centroid_B
        H = Am.T @ Bm
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0:
            Vt[-1,:] *= -1
            R_mat = Vt.T @ U.T
        t = centroid_B - R_mat @ centroid_A
        print("t: ", t)
        if np.linalg.norm(t) < 1e-6:
            self.get_logger().warn("Computed translation is near zero; check input world positions")
        return R_mat, t

    # -------------------------------
    # Publishing
    # -------------------------------
    def publish_rectangle_center(self, center, num_markers):
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = 1.0
        self.rectangle_center_pub.publish(msg)
        self.get_logger().info(f'Rectangle center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) from {num_markers} markers')

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
