import numpy as np
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
import roboticstoolbox as rtb
import rtde_control
import rtde_receive
import sys

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist, Pose, PoseArray
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray

import time

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from butter import RealtimeButterworthFilter
from trajectory_logger import TrajectoryLogger
from qp_solver import MobileManipulatorQPSolver

class QP_mbcontorller(Node):
    def __init__(self):
        super().__init__('mbcontroller')
        self.ROBOT_IP = '192.160.0.4'
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            print(f"✅ UR 로봇 연결 성공 (IP: {self.ROBOT_IP})")
        except Exception as e:
            print(f"❌ UR 로봇 연결 실패: {e}")
            sys.exit(1)
        
        self.ur5e_robot = rtb.models.UR5()
        self.n_dof = 8 # base(2) + arm(6)
        self.base_position = self.create_subscription(Pose, '/mobile_base/pose', self.set_base_position, 10)
        self.cable_position = self.create_subscription(PoseArray, '/cable_points', self.set_cable_positions, 10)
        self.human_position = self.create_subscription(Pose, '/hand_pose', self.set_human_position, 10)
        self.qdlim = np.array([0.15]*8)
        self.qdlim[:1] = 0.05
        self.qdlim[1] = 0.05
        self.d_safe = 0.1
        self.d_influence = 1.5
        self.q = None
        self.obstacle_radius = 0.25
        self.dt = 0.05
        self.create_timer(0.05, self.QP_real)
        self.scout_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.butter_x = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_y = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_z = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_hand_x = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=15.0)
        self.butter_hand_y = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=15.0)
        self.butter_hand_z = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=15.0)
        self.kf_initialized = False
        self.kf_x = np.zeros(6)
        self.kf_P = np.eye(6) * 0.05
        self.kf_R = np.eye(3) * 0.01
        self.kf_q_pos = 1e-4
        self.kf_q_vel = 1e-3
        self.human_last_time = None
        self.kf_gate_dist = 0.6
        self.cable_filter = [RealtimeButterworthFilter(order=1, cutoff=3.0, fs=15.0), RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0), RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)]

        self.marker_pub = self.create_publisher(MarkerArray, '/robot_collision_markers', 10)
        self.create_timer(0.1, self.publish_collision_markers)

        self.human_position = None
        self.obstacles_positions= np.array([[0.7,0.0,0.5]])
        self.base_quaternion= None
        self.lambda_h_a_param = 0.3
        self.w1 = 0.5
        self.w2 = 0.5
        self.w3 = 0.0
        self.w4 = 0.2
        self.define_robot_collision_points()
        self.ur5e_frame_correction = np.eye(4)
        self.ur5e_frame_correction[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        self.cable_points = []
        self.target_reached_threshold = 0.2
        self.target_reached = False
        self.current_waypoint_id = 0
        self.last_target_position = None
        self.target_reached_debounce_time = 0.5
        self.target_reached_time = None
        
        self.ee_rotation = False

        self.history = TrajectoryLogger(self.w1, self.w2, self.w4, self.obstacle_radius, self.obstacles_positions)
        self.qp_solver = MobileManipulatorQPSolver(
            n_dof=self.n_dof,
            qdlim=self.qdlim,
            d_safe=self.d_safe,
            d_influence=self.d_influence,
            obstacle_radius=self.obstacle_radius,
            lambda_h_a_param=self.lambda_h_a_param,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
            ur5e_link_points=self.ur5e_link_points,
        )
        print("🎯 사람 추종 시스템 초기화 완료")

    def define_robot_collision_points(self):
        """각 링크별 충돌 검사용 점들 정의 (링크 로컬 좌표계 기준)"""
        
        # UR5e 각 링크의 충돌 검사 점들 (링크 프레임 기준)
        self.ur5e_link_points = {
            'base': np.array([
                [0.0, 0.0, 0.0],      # 베이스 중심
                [0.08, 0.08, 0.05],   # 베이스 모서리들
                [-0.08, 0.08, 0.05],
                [0.08, -0.08, 0.05],
                [-0.08, -0.08, 0.05],
                [0.06, 0.0, 0.1],     # 베이스 상단
                [-0.06, 0.0, 0.1],
                [0.0, 0.06, 0.1],
                [0.0, -0.06, 0.1]
            ]),
            'shoulder': np.array([
                [0.0, 0.0, 0.0],
                [0.06, 0.06, 0.08],
                [-0.06, 0.06, 0.08],
                [0.06, -0.06, 0.08],
                [-0.06, -0.06, 0.08],
                [0.0, 0.0, 0.12]
            ]),
            'upper_arm': np.array([
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.1],     # 상완 중간 지점들
                [-0.05, 0.0, 0.1],
                [0.0, 0.05, 0.1],
                [0.0, -0.05, 0.1],
                [0.04, 0.0, 0.2],     # 상완 끝 부분
                [-0.04, 0.0, 0.2],
                [0.0, 0.04, 0.25],
                [0.0, -0.04, 0.25]
            ]),
            'forearm': np.array([
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.08],    # 전완 중간
                [-0.04, 0.0, 0.08],
                [0.0, 0.04, 0.08],
                [0.0, -0.04, 0.08],
                [0.03, 0.0, 0.15],    # 전완 끝
                [-0.03, 0.0, 0.15],
                [0.0, 0.03, 0.17],
                [0.0, -0.03, 0.17]
            ]),
            'wrist_1': np.array([
                [0.0, 0.0, 0.0],
                [0.03, 0.03, 0.0],
                [-0.03, 0.03, 0.0],
                [0.03, -0.03, 0.0],
                [-0.03, -0.03, 0.0]
            ]),
            'wrist_2': np.array([
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.03],
                [-0.03, 0.0, 0.03],
                [0.0, 0.03, 0.03],
                [0.0, -0.03, 0.03]
            ]),
            'wrist_3': np.array([
                [0.0, 0.0, 0.0],
                [0.025, 0.025, 0.02],
                [-0.025, 0.025, 0.02],
                [0.025, -0.025, 0.02],
                [-0.025, -0.025, 0.02]
            ])
        }
        
        # 모바일 베이스 충돌 점들 (베이스 프레임 기준)
        self.mobile_base_points = np.array([
            [0.35, 0.25, 0.1],    # 베이스 모서리들 (Scout 크기 고려)
            [-0.35, 0.25, 0.1],
            [0.35, -0.25, 0.1],
            [-0.35, -0.25, 0.1],
            [0.3, 0.2, 0.3],      # 베이스 상단
            [-0.3, 0.2, 0.3],
            [0.3, -0.2, 0.3],
            [-0.3, -0.2, 0.3]
        ])

    def publish_collision_markers(self):
        """Publish mobile_base_points and ur5e_link_points as MarkerArray in 'world' frame."""
        try:
            pts_world, num_mobile, num_mani = self.compute_dynamic_robot_collision_points()
            if pts_world is None or len(pts_world) == 0:
                return
            ma = MarkerArray()
            now = self.get_clock().now().to_msg()
            lifetime = Duration(sec=0, nanosec=200_000_000)  # 0.2s
            scale = 0.05

            for i in range(num_mobile):
                p = pts_world[i]
                m = Marker()
                m.header.stamp = now
                m.header.frame_id = 'world'
                m.ns = 'mobile_base_points'
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(p[0])
                m.pose.position.y = float(p[1])
                m.pose.position.z = float(p[2])
                m.pose.orientation.w = 1.0
                m.scale.x = scale
                m.scale.y = scale
                m.scale.z = scale
                m.color.r = 0.0
                m.color.g = 0.2
                m.color.b = 1.0
                m.color.a = 0.9
                m.lifetime = lifetime
                ma.markers.append(m)

            for j, idx in enumerate(range(num_mobile, len(pts_world))):
                p = pts_world[idx]
                m = Marker()
                m.header.stamp = now
                m.header.frame_id = 'world'
                m.ns = 'manipulator_points'
                m.id = j
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(p[0])
                m.pose.position.y = float(p[1])
                m.pose.position.z = float(p[2])
                m.pose.orientation.w = 1.0
                m.scale.x = scale * 0.9
                m.scale.y = scale * 0.9
                m.scale.z = scale * 0.9
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.2
                m.color.a = 0.9
                m.lifetime = lifetime
                ma.markers.append(m)

            self.marker_pub.publish(ma)
        except Exception as e:
            self.get_logger().warn(f"publish_collision_markers failed: {e}")

    def compute_dynamic_robot_collision_points(self):
        """현재 로봇 상태에서 모든 충돌 검사 점들의 월드 좌표 계산"""
        
        if self.base_position is None or self.base_quaternion is None:
            return None, 0, 0
        
        collision_points_world = []
        
        # 1. 모바일 베이스 변환 행렬
        T_sb = np.eye(4)
        T_sb[0,3] = self.base_position[0]
        T_sb[1,3] = self.base_position[1] 
        T_sb[2,3] = self.base_position[2] 
        T_sb[:3, :3] = R.from_quat(self.base_quaternion).as_matrix()
        
        # 2. 베이스에서 UR5e 베이스로의 변환
        T_b0 = np.eye(4)
        T_b0[0,3] = 0.1315
        T_b0[2,3] = 0.51921
        
        # 3. 모바일 베이스 점들을 월드 좌표계로 변환
        for point in self.mobile_base_points:
            point_homogeneous = np.append(point, 1)
            point_world = T_sb @ point_homogeneous
            collision_points_world.append(point_world[:3])
        num_mobile = len(collision_points_world)
        # 4. UR5e 각 링크의 점들 변환
        link_names = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist_1', 'wrist_2', 'wrist_3']
        num_mani = 0
        # build manual fk chain from arm joint values (self.q[2:] is arm q)
        fk_chain = self.ur5e_fk_chain_manual(self.q[2:])
        for i, link_name in enumerate(link_names):
            # use precomputed chain (T_0 = identity, T_1.. = after joint1, ...)
            if i < len(fk_chain):
                T_0i = fk_chain[i]
            else:
                T_0i = fk_chain[-1] if len(fk_chain) > 0 else np.eye(4)

            # 월드에서 i번째 링크로의 변환
            T_si = T_sb @ T_b0 @ T_0i
            corr = self.ur5e_frame_correction
            T_si = T_sb @ T_b0 @ corr @ T_0i
            p_si = T_si[:3, 3]
            q_si = R.from_matrix(T_si[0:3,0:3]).as_quat()
            self.make_tf_msg(p_si, q_si, "world", f"link{i}")
            # 해당 링크의 충돌 점들을 월드 좌표계로 변환
            if link_name in self.ur5e_link_points:
                for point in self.ur5e_link_points[link_name]:
                    point_homogeneous = np.append(point, 1)
                    point_world = T_si @ point_homogeneous
                    collision_points_world.append(point_world[:3])
                    num_mani += 1

        return collision_points_world, num_mobile, num_mani


    def _dh_transform(self, a, alpha, d, theta):
        """Classic DH homogeneous transform"""
        ca = np.cos(alpha); sa = np.sin(alpha)
        ct = np.cos(theta); st = np.sin(theta)
        T = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0.,     sa,     ca,    d],
            [0.,    0.,    0.,   1.0]
        ], dtype=float)
        return T

    def ur5e_fk_chain_manual(self, q_arm):
        """Compute FK chain for UR5e using standard DH params.
           q_arm: array-like of length 1..6 (radians).
           Returns list of 4x4 transforms [T_0 (identity), T_1, ..., T_n]
           where T_k is transform from arm-base to link_k frame (after joint k).
        """
        # UR5 DH parameters (standard)
        a = [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0]
        alpha = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
        d = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]
        # ensure vector
        q = np.asarray(q_arm, dtype=float).reshape(-1)
        # pad q to 6 with zeros if shorter
        q_pad = np.zeros(6, dtype=float)
        q_pad[:q.size] = q

        transforms = [np.eye(4)]
        T_acc = np.eye(4)
        for i in range(6):
            T_link = self._dh_transform(a[i], alpha[i], d[i], q_pad[i])
            T_acc = T_acc @ T_link
            transforms.append(T_acc.copy())
            if i + 1 >= q.size:
                break
        return transforms


    def set_base_position(self, msg):
        buttered_x = self.butter_x.update(msg.position.x)
        buttered_y = self.butter_y.update(msg.position.y)
        buttered_z = self.butter_z.update(msg.position.z)
        
        H_world_aruco = np.eye(4)
        H_world_aruco[0,3] = buttered_x #msg.position.x
        H_world_aruco[1,3] = buttered_y #msg.position.y
        H_world_aruco[2,3] = buttered_z #msg.position.z
        H_world_aruco[:3, :3] = R.from_quat([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w]).as_matrix()
        H_aruco_base = np.eye(4)
        H_aruco_base[0,3] = 0.015
        H_aruco_base[1,3] = -0.16
        H_aruco_base[2,3] = -0.51921
        H_aruco_base[:3, :3] = np.eye(3)
        H_world_base = H_world_aruco @ H_aruco_base

        self.base_position = [H_world_base[0,3],
                            H_world_base[1,3],
                            H_world_base[2,3]]
        
        self.base_quaternion = [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ]
        
    def set_human_position(self, msg):
        if self.base_position is None:
            return
        buttered_hand_x = self.butter_hand_x.update(msg.position.x)
        buttered_hand_y = self.butter_hand_y.update(msg.position.y)
        buttered_hand_z = self.butter_hand_z.update(msg.position.z)
        self.human_position = [buttered_hand_x, 
                               buttered_hand_y, 
                               buttered_hand_z]
        
    def set_cable_positions(self, msg):
        self.cable_points = []

        for pose in msg.poses:
            filtered = [
                f.update(v) for f, v in zip(self.cable_filter,
                                        [pose.position.x,
                                            pose.position.y,
                                            pose.position.z])
            ]
            filtered = np.array(filtered, dtype=float)
            self.cable_points.append(filtered)
    
    def make_tf_msg(self, pos, quat, parent_name, child_frame_name):
        tfmsg = TransformStamped()
        tfmsg.header.stamp = self.get_clock().now().to_msg()
        tfmsg.header.frame_id = parent_name
        tfmsg.child_frame_id = child_frame_name
        tfmsg.transform.translation.x = pos[0]
        tfmsg.transform.translation.y = pos[1]
        tfmsg.transform.translation.z = pos[2]
        tfmsg.transform.rotation.x = quat[0]
        tfmsg.transform.rotation.y = quat[1]
        tfmsg.transform.rotation.z = quat[2]
        tfmsg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(tfmsg)

    def QP_real(self):
        entire_start_t = time.time()
        self.rtde_c.initPeriod()
        current_joint_positions = self.rtde_r.getActualQ() # 실제 현재 joint 위치

        self.q = np.zeros(8)
        self.q[0] = 0.0
        self.q[1] = 0.0
        self.q[2:] = current_joint_positions

        if self.base_position is None or self.base_quaternion is None:
            return

        robot_collision_check, num_mobile, num_mani = self.compute_dynamic_robot_collision_points()
        xform_pose = [np.array(p) for p in robot_collision_check]
        xform_pose.extend(np.array(p) for p in self.cable_points)

        T_sb = np.eye(4)
        T_sb[0,3] = self.base_position[0]
        T_sb[1,3] = self.base_position[1]
        T_sb[2,3] = self.base_position[2]
        T_sb[:3, :3] = R.from_quat(self.base_quaternion).as_matrix()
        T_b0 = np.eye(4)
        T_b0[0,3] = 0.1315
        T_b0[2,3] = 0.51921

        pppp = list(T_sb[0:3,3])
        qqqq = R.from_matrix(T_sb[0:3,0:3]).as_quat()
        self.make_tf_msg(pppp, qqqq, "world", "base_world")

        ppose = list(T_b0[0:3,3])
        qqua = R.from_matrix(T_b0[0:3,0:3]).as_quat()
        self.make_tf_msg(ppose, qqua, "base_world", "base_0")

        T_0e = self.ur5e_robot.fkine(self.q[2:]).A
        T_be = T_b0 @ T_0e

        pppose = list(T_be[0:3,3])
        qqqua = R.from_matrix(T_be[0:3,0:3]).as_quat()
        self.make_tf_msg(pppose, qqqua, "base_world", "ee_base")

        H_current = SE3(T_be)
        xform_pose = np.array(xform_pose) if len(xform_pose) > 0 else np.array([]).reshape(0, 3)

        T_cur = T_sb @ T_be
        self.T_cur = T_cur
        current_ee_position = T_cur[:3, 3]
        current_base_position = T_sb[:3, 3]

        current_time = time.time() - self.history.start_time

        ppp = list(T_cur[0:3,3])
        qqq = R.from_matrix(T_cur[0:3,0:3]).as_quat()
        self.make_tf_msg(ppp, qqq, "world", "ee")
        p_obs = [self.obstacles_positions[:, :3][0][0], self.obstacles_positions[:, :3][0][1],self.obstacles_positions[:, :3][0][2]]
        q_obs = R.from_matrix(T_cur[0:3,0:3]).as_quat()
        self.make_tf_msg(p_obs, q_obs, "world", "obstacle")

        if self.human_position is None:
            print('No Desired Position')
            return

        robot_target_position_homogeneous = np.append(self.human_position, 1) 
        robot_target_position_local = T_cur @ robot_target_position_homogeneous
        robot_target_position_ee = robot_target_position_local[:3]

        if not self.ee_rotation:
            self.rotation_matrix = T_cur[:3, :3]
            self.ee_rotation = True

        T_sd = np.eye(4)
        T_sd[:3, :3] = self.rotation_matrix
        det = np.linalg.det(self.rotation_matrix)
        orthogonality_check = np.allclose(self.rotation_matrix.T @ self.rotation_matrix, np.eye(3))

        if not np.isclose(det, 1.0) or not orthogonality_check:
            print("Invalid rotation matrix detected. Normalizing...")
            U, _, Vt = np.linalg.svd(self.rotation_matrix)
            self.rotation_matrix = U @ Vt
            T_sd[:3, :3] = self.rotation_matrix

        T_sd[0, 3] = (robot_target_position_ee[0] + T_cur[0, 3]) / 2
        T_sd[1, 3] = (robot_target_position_ee[1] + T_cur[1, 3]) / 2
        T_sd[2, 3] = (robot_target_position_ee[2] + T_cur[2, 3]) / 2
        
        self.history.record_trajectory(
            current_time,
            current_base_position,
            current_ee_position,
            T_sd[:3, 3],
        )

        ppppp = list(T_sd[0:3,3])
        qqqqq = R.from_matrix(T_sd[0:3,0:3]).as_quat()
        self.make_tf_msg(ppppp, qqqqq, "world", "desired")

        ppppp_ = list(robot_target_position_ee)
        qqqqq_ = R.from_matrix(T_sd[0:3,0:3]).as_quat()
        self.make_tf_msg(ppppp_, qqqqq_, "world", "human")

        sight_vec = T_cur[:3,0]
        sight_vec /= np.linalg.norm(sight_vec)
        direction_unit_vector = self.human_position - T_cur[:3, 3]
        direction_unit_vector = direction_unit_vector / np.linalg.norm(direction_unit_vector)
        cos_theta = np.dot(direction_unit_vector, sight_vec)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        T_bd = np.linalg.inv(T_sb) @ T_sd
        H_desired = SE3(T_bd)

        solver_result = self.qp_solver.solve(
            q=self.q,
            ur5e_robot=self.ur5e_robot,
            T_be=T_be,
            H_current=H_current,
            H_desired=H_desired,
            xform_pose=xform_pose,
            num_mobile=num_mobile,
            obstacles_positions=self.obstacles_positions,
            direction_unit_vector=direction_unit_vector,
            theta=theta,
        )

        qd = solver_result["qd"]
        et = solver_result["et"]
        self.history.record_timing(solver_result["problem_solve_time"], 0.0)
        self.history.record_objective(
            qd,
            solver_result["Q"],
            C1=solver_result["C1"],
            C2=solver_result["C2"],
            C3=solver_result["C3"],
            C4=solver_result["C4"],
        )

        if et < self.target_reached_threshold:
            qd = qd[: self.n_dof]
            qd = 0 * qd

        print('qd: ', qd[:8])

        wc, vc = qd[0], qd[1]
        qdc = qd[2:8]
        twist = Twist()
        twist.linear.x = vc
        twist.angular.z = wc

        entire_end_t = time.time()
        entire_elap_t = entire_end_t - entire_start_t
        self.history.entire_t[-1] = entire_elap_t


    def plot_objective_time_series(self, save_path=None, show_plot=False):
        self.history.plot_objective_time_series(save_path=save_path, show_plot=show_plot)

    def save_ee_trajectory(self):
        self.history.save_ee_trajectory()

    def cleanup_and_plot(self):
        """정리 작업 및 궤적 저장"""
        print("\n🛑 프로그램 종료 중...")
        self.save_ee_trajectory()
        try:
            self.plot_objective_time_series()
        except Exception as e:
            print(f"Failed to save objective plot: {e}")

if __name__ == '__main__':
    rclpy.init()
    node = QP_mbcontorller()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 감지됨")
    finally:
        if rclpy.ok():
            print("✨ ROS 2 종료 정리 시작...")
            node.cleanup_and_plot()
            node.destroy_node()
            rclpy.shutdown()
        else:
            print("✨ ROS 2 이미 종료됨. 추가 정리 생략.")