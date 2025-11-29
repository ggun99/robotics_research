import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.interpolate
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import random
from cv2 import waitKey
import numpy as np
import rtde_control
import rtde_receive
import cvxpy as cp
import signal
import sys
import os

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import Twist, Pose, PoseArray
from sensor_msgs.msg import JointState

import time
from std_msgs.msg import Bool, Int32

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from butter import RealtimeButterworthFilter

class QP_mbcontorller(Node):
    def __init__(self):
        super().__init__('mbcontroller')
        self.ROBOT_IP = '192.160.0.4'
        # RTDE 수신 객체 생성
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
        # RTDE Control Interface 초기화
        self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
        
        self.ur5e_robot = rtb.models.UR5()
        self.n_dof = 8 # base(2) + arm(6)
        self.base_position = self.create_subscription(Pose, '/mobile_base/pose', self.set_base_position, 10)
        self.cable_position = self.create_subscription(PoseArray, '/cable_points', self.set_cable_positions, 10)
        self.human_position = self.create_subscription(Pose, '/hand_pose', self.set_human_position, 10)
        self.eta = 1
        self.qdlim = np.array([0.15]*8)
        self.qdlim[:1] = 0.05  # 베이스 조인트 속도 제한
        self.qdlim[1] = 0.05
        self.qlim = np.array([[-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ np.inf, np.inf, 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])
        self.H_desired = None
        # collision avoidance parameters
        self.d_safe = 0.2
        self.d_influence = 2.0
        self.current_joint_positions = None
        self.q = None
        # self.num_points = 10
        self.obstacle_radius = 0.25
        self.lambda_max = 0.32
        self.dt = 0.05
        self.create_timer(0.05, self.QP_real)  # 20Hz
        self.scout_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.ur5e_publisher = self.create_publisher(JointState, 'ur5e_vel', 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        # 🚨 수정: X, Y, Z용 필터 인스턴스 3개 생성
        self.butter_x = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.butter_y = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.butter_z = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.butter_hand_x = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.butter_hand_y = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.butter_hand_z = RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)
        self.cable_filter = [RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0), RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0), RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0)]
        
        self.human_position = None
        self.obstacles_positions= np.array([5.0,5.0,1.0])
        self.points_between= None
        self.base_quaternion= None
        self.robot_collision_check= []
        self.lambda_h_a_param = 0.3
        self.w1 = 0.2
        self.w2 = 0.5
        self.w3 = 0.0
        self.w4 = 0.0
        # 로봇 링크별 충돌 검사 점들 정의
        self.define_robot_collision_points()

        self.cable_points = []
        self.obstacles_positions = np.array([[1.25,4.0, 1.]])

        # 궤적 추종 관련 Publisher/Subscriber 추가
        self.next_waypoint_publisher = self.create_publisher(Bool, '/next_waypoint_trigger', 10)
        self.trajectory_status_sub = self.create_subscription(
            Bool, '/trajectory_completed', self.trajectory_completed_callback, 10)
        self.current_waypoint_sub = self.create_subscription(
            Int32, '/current_waypoint', self.current_waypoint_callback, 10)
        
        # 궤적 추종 상태 변수 (수정됨)
        self.target_reached_threshold = 0.08  # 10cm 이내면 도달로 판단
        self.target_reached = False
        self.current_waypoint_id = 0
        self.last_target_position = None  # 이전 목표 위치 저장
        self.target_reached_debounce_time = 0.5  # 도달 판정 후 1초 디바운스
        self.target_reached_time = None
        
        self.ee_rotation = False

        # End effector 궤적 기록용
        self.ee_trajectory = {'x': [], 'y': [], 'timestamps': []}
        self.desired_trajectory = {'x': [], 'y': [], 'timestamps': []}
        self.last_desired_position = None
        self.start_time = time.time()
        self.obj_timestamps = []
        self.obj_total = []
        self.obj_Qterm = []
        self.obj_C1 = []
        self.obj_C2 = []
        self.obj_C3 = []
        self.obj_C4 = []
        
        print("🎯 궤적 추종 시스템 초기화 완료")

    def define_robot_collision_points(self):
        """각 링크별 충돌 검사용 점들 정의 (링크 로컬 좌표계 기준)"""
        
        # UR5e 각 링크의 충돌 검사 점들 (링크 프레임 기준)
        self.ur5e_link_points = {
            # 'base': np.array([
            #     [0.0, 0.0, 0.0],      # 베이스 중심
            #     [0.08, 0.08, 0.05],   # 베이스 모서리들
            #     [-0.08, 0.08, 0.05],
            #     [0.08, -0.08, 0.05],
            #     [-0.08, -0.08, 0.05],
            #     [0.06, 0.0, 0.1],     # 베이스 상단
            #     [-0.06, 0.0, 0.1],
            #     [0.0, 0.06, 0.1],
            #     [0.0, -0.06, 0.1]
            # ]),
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
            # [0.25, 0.0, 0.4],     # 베이스 중앙 상단
            # [-0.25, 0.0, 0.4],
            # [0.0, 0.2, 0.4],
            # [0.0, -0.2, 0.4]
        ])

    def trajectory_completed_callback(self, msg):
        """궤적 완료 콜백"""
        pass

    def current_waypoint_callback(self, msg):
        """현재 웨이포인트 ID 업데이트"""
        prev_waypoint_id = self.current_waypoint_id
        self.current_waypoint_id = msg.data
         # 새로운 웨이포인트로 변경되면 상태 리셋
        if prev_waypoint_id != self.current_waypoint_id:
            self.target_reached = False
            self.target_reached_time = None
            print(f"📍 새 웨이포인트 ID: {self.current_waypoint_id} (상태 리셋)")

    def check_target_reached(self, current_pos, target_pos):
        """목표 지점 도달 여부 확인"""
        if target_pos is None or current_pos is None:
            return False
        
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        self.get_logger().info(f"Distance to target: {distance:.3f} m")
        # self.get_logger().info(f"Current Pos: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
        return distance < self.target_reached_threshold

    
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
        
        for i, link_name in enumerate(link_names):
            # 각 링크까지의 변환 행렬 계산
            if i == 0:  # base_link
                T_0i = np.eye(4)
            else:  # 다른 링크들
                T_0i = self.ur5e_robot.fkine(self.q[2:2+i]).A
            
            # 월드에서 i번째 링크로의 변환
            T_si = T_sb @ T_b0 @ T_0i
            
            # 해당 링크의 충돌 점들을 월드 좌표계로 변환

            if link_name in self.ur5e_link_points:
                num_mani = 0
                for point in self.ur5e_link_points[link_name]:
                    point_homogeneous = np.append(point, 1)
                    point_world = T_si @ point_homogeneous
                    collision_points_world.append(point_world[:3])
                    num_mani += 1

        return np.array(collision_points_world), num_mobile, num_mani

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
            self.cable_points.append(filtered)
        # self.cable_points = [
        #     (pose.position.x, pose.position.y, pose.position.z)
        #     for pose in msg.poses
        # ]

    def get_nearest_obstacle_distance(self, position, obstacles, obstacle_radius):
        """
        Calculate the distance to the nearest obstacle from a given position in the end-effector frame.
        
        Args:
            position (np.ndarray): The position in world coordinates.
            obstacles (list): A list of obstacle positions in world coordinates.
            obstacle_radius (float): The radius of the obstacles.
            T_cur (np.ndarray): The transformation matrix from world to the robot base.
            T (np.ndarray): The transformation matrix from the robot base to the end-effector.

        Returns:
            float: The distance to the nearest obstacle.
            int: The index of the nearest obstacle.
            np.ndarray: The directional vector to the nearest obstacle in the end-effector frame.
        """
        # 엔드 이펙터의 변환 행렬
        # T_e = T_cur @ T  # 월드 좌표계에서 엔드 이펙터 좌표계로의 변환
        g_vec = np.zeros(3)
        obstacles_local = []
        obs_real = []
        
        for obs in obstacles:
            obs[2] = position[2]
            obs_real.append(obs)
            # obs_ee = np.append(obs, 1)  # 동차 좌표로 확장
            # obs_local = np.linalg.inv(T_e) @ obs_homogeneous
            obstacles_local.append(obs)  # 3차원으로 변환
            # position_homogeneous = np.append(position, 1)  # 동차 좌표로 확장
            # position_local = np.linalg.inv(T_e) @ position_homogeneous
            # position_local = position_local[:3]  # 3차원으로 변환
            
        position_ee = position  # 동차 좌표로 확장
        # position_local = np.linalg.inv(T_e) @ position_homogeneous
        # position_local = position_ee[:3]  # 3차원으로 변환
        # position_local = position  # 3차원으로 변환
        
        distances = [((np.linalg.norm(position_ee - obse)) - obstacle_radius) for obse in obstacles_local]
        index = np.argmin(distances)

        g_vec = (position - obs_real[index])
        g_vec /= np.linalg.norm(g_vec) 
        # print('g_vec: ', g_vec)
        return distances, index, g_vec
    
    def joint_velocity_damper(self, 
            ps: float = 0.05,
            pi: float = 0.1,
            n: int = 8,
            gain: float = 1.0,
        ):
            """
            Compute the joint velocity damper for QP motion control

            Formulates an inequality contraint which, when optimised for will
            make it impossible for the robot to run into joint limits. Requires
            the joint limits of the robot to be specified. See examples/mmc.py
            for use case

            Attributes
            ----------
            ps
                The minimum angle (in radians) in which the joint is
                allowed to approach to its limit
            pi
                The influence angle (in radians) in which the velocity
                damper becomes active
            n
                The number of joints to consider. Defaults to all joints
            gain
                The gain for the velocity damper

            Returns
            -------
            Ain
                A (6,) vector inequality contraint for an optisator
            Bin
                b (6,) vector inequality contraint for an optisator

            """

            Ain = np.zeros((n, n))
            Bin = np.zeros(n)

            for i in range(n):
                if self.q[i] - self.qlim[0, i] <= pi:
                    Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                    Ain[i, i] = -1
                if self.qlim[1, i] - self.q[i] <= pi:
                    Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                    Ain[i, i] = 1

            return Ain, Bin


    # 비콘을 이용한 3차원 위치

    # obstacles_positions = np.array([
    #     [1.2,1.8, 0.97],
    #     [2.8, 0.5, 0.97],
    #     [2.5 , 2.3, 0.97]])


    # # 원기둥 생성
    # obstacle_radius = 0.2
    # obstacle_height = 2.3

    # def joint_sub(self):
    #     # sub the joints values
    #     current_joint_positions = cur_j # 실제 현재 joint 위치
    #     self.current_joint_positions = current_joint_positions
    #     self.x = mobile_base_pose[0][0]
    #     self.y = mobile_base_pose[0][1] 
    #     self.z = mobile_base_pose[0][2] 

    #     quat = mobile_base_quat[0]
    #     self.r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    #     self.euler = self.r.as_euler('zyx', degrees=False)  # 'zyx' 순서로 euler angles 추출

    #     self.q = np.zeros(8)
    #     self.q[0] = 0.0
    #     self.q[1] = 0.0 
    #     self.q[2:] = current_joint_positions[4:10]  # UR5e 조인트 위치

    def calculate_natural_rotation(self, T_cur, target_position):

        cur_z_axis = T_cur[:3, 2]
        current_position = T_cur[:3, 3]
        direction_vector = target_position - current_position
        direction_vector /= np.linalg.norm(direction_vector)
        
        new_z_axis = direction_vector
        new_y_axis = np.cross(cur_z_axis, new_z_axis)
        if np.linalg.norm(new_y_axis) < 1e-6:
            new_y_axis = np.array([0, 1, 0])
        new_y_axis /= np.linalg.norm(new_y_axis)
        
        new_x_axis = np.cross(new_y_axis, new_z_axis)
        new_x_axis /= np.linalg.norm(new_x_axis)
        
        rotation_matrix = np.vstack([new_x_axis, new_y_axis, new_z_axis]).T
        return rotation_matrix

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

    # 여기서 ros를 사용한 것으로 변경
    def QP_real(self):
        t_start = self.rtde_c.initPeriod()
        # sub the joints values
        current_joint_positions = self.rtde_r.getActualQ() # 실제 현재 joint 위치
        # 현재 로봇 베이스의 쿼터니언 회전값
        
        self.q = np.zeros(8)
        self.q[0] = 0.0
        self.q[1] = 0.0 
        self.q[2:] = current_joint_positions  # UR5e 조인트 위치

        if self.base_position is None or self.base_quaternion is None:
            return 
        # print('base_position', self.base_position)
        # print('base_quaternion', self.base_quaternion)
        # 동적으로 로봇 충돌 점들 계산
        self.robot_collision_check, num_mobile, num_mani = self.compute_dynamic_robot_collision_points()
        xform_pose = list(self.robot_collision_check)
        num_cable = len(self.cable_points)
        num_points = num_mobile + num_mani + num_cable
        if num_cable == 0:
            pass
        else: 
            xform_pose.append(self.cable_points)

        
        
         # 베이스 프레임 변환 행렬
        T_sb = np.eye(4)
        T_sb[0,3] = self.base_position[0]
        T_sb[1,3] = self.base_position[1] 
        T_sb[2,3] = self.base_position[2] 
        T_sb[:3, :3] = R.from_quat(self.base_quaternion).as_matrix() 
        T_b0 = np.eye(4)
        T_b0[0,3] = 0.1315 # 0.1015
        T_b0[2,3] = 0.51921  # 0.47921

        # self.make_tf_msg(fakep, fakeq, "base", "ee_base_all0")
        # print(T_be)
        pppp = list(T_sb[0:3,3])
        # print(pppp)
        qqqq = R.from_matrix(T_sb[0:3,0:3]).as_quat()
        # print(qqqq)
        self.make_tf_msg(pppp, qqqq, "world", "base_world")

        ppose = list(T_b0[0:3,3])
        # print(pppp)
        qqua = R.from_matrix(T_b0[0:3,0:3]).as_quat()
        # print(qqqq)
        self.make_tf_msg(ppose, qqua, "base_world", "base_0")

        # rot_mat = np.eye(4)
        # rot_mat[0,0] = -1.
        # rot_mat[1,1] = -1.
        # rot_mat[2,2] = 1.
        # print(self.q[2:])
        T_0e = self.ur5e_robot.fkine(self.q[2:]).A 
        # print(T_0e)
        # ppppose = list(T_0e[0:3,3])
        # # print(pppp)
        # qqqqua = R.from_matrix(T_0e[0:3,0:3]).as_quat()
        # # print(qqqq)
        # self.make_tf_msg(ppppose, qqqqua, "base_0", "ee_0")

        T = T_b0 @ T_0e  # 베이스 프레임 기준 end-effector 위치

        # correction = np.array([[ 0,  0,  1,  0],
        #                     [-1,  0,  0,  0],
        #                     [ 0, -1,  0,  0],
        #                     [ 0,  0,  0,  1]])

        T_be = T #@ correction
        # print(T_be)
        pppose = list(T_be[0:3,3])
        # print(pppp)
        qqqua = R.from_matrix(T_be[0:3,0:3]).as_quat()
        # print(qqqq)
        self.make_tf_msg(pppose, qqqua, "base_world", "ee_base")

        H_current = SE3(T_be)  # 현재 end-effector 위치
        

        # 각 조인트의 변환 행렬 계산
        # for i in range(1, 7):  # UR5e의 6개의 조인트
        #     T_bi = self.ur5e_robot.fkine(self.q[2:i+2]).A  # 베이스 좌표계에서 i번째 조인트까지의 변환 행렬
        #     T_wi = T_sb @ T_bi  # 월드 좌표계에서 i번째 조인트까지의 변환 행렬
        #     joint_position = T_wi[:3, 3]  # 동차 좌표에서 [x, y, z] 추출
        #     xform_pose.append(joint_position)
        xform_pose = np.array(xform_pose) if len(xform_pose) > 0 else np.array([]).reshape(0, 3)

        # 로봇이 사람을 따라가기
        T_cur = T_sb @ T_be  # 현재 로봇 위치 (월드 좌표계 기준)
        self.T_cur = T_cur
        # 현재 엔드이펙터 위치
        current_ee_position = T_cur[:3, 3]
        
        # End effector 궤적 기록
        current_time = time.time() - self.start_time
        self.ee_trajectory['x'].append(current_ee_position[0])
        self.ee_trajectory['y'].append(current_ee_position[1])
        self.ee_trajectory['timestamps'].append(current_time)

        # print(T_cur)
        ppp = list(T_cur[0:3,3])
        # print(ppp)
        qqq = R.from_matrix(T_cur[0:3,0:3]).as_quat()
        # print(qqq)
        self.make_tf_msg(ppp, qqq, "world", "ee")

        # 엔드 이펙터의 변환 행렬
        # T_e = T_cur  # 월드 좌표계에서 엔드 이펙터 좌표계로의 변환

        # # robot_target_position을 엔드 이펙터 좌표계로 변환
        # robot_target_position_homogeneous = np.append(self.human_position, 1)  # 동차 좌표로 확장
        # robot_target_position_local = np.linalg.inv(T_e) @ robot_target_position_homogeneous
        # robot_target_position_local = self.human_position  # 3차원으로 변환

        # # 현재 엔드 이펙터 위치를 엔드 이펙터 좌표계로 변환 (항상 원점)

        # # 목표 방향 계산 (엔드 이펙터 좌표계 기준)
        # direction_vector = robot_target_position_local # - cur_p_local
        # direction_vector /= np.linalg.norm(direction_vector)  # 방향 벡터 정규화

        # # 로봇의 현재 x축 방향 (엔드 이펙터의 x축)
        # current_x_axis = T_e[:3, 0]  # 엔드 이펙터 변환 행렬의 첫 번째 열

        # # 엔드 이펙터 기준의 방향 벡터 (direction_vector)를 월드 좌표계로 변환
        # direction_vector_homogeneous = np.append(direction_vector, 0)  # 방향 벡터는 동차 좌표로 확장 (위치가 아니므로 마지막 값은 0)
        # direction_vector_world = T_e[:3, :3] @ direction_vector_homogeneous[:3]  # 회전 행렬만 적용하여 월드 좌표계로 변환

        if self.human_position is None:
            print('No Desired Position')
            return

        # 회전 행렬 생성
        if self.ee_rotation == False:
            self.rotation_matrix = T_cur[:3, :3]    #self.calculate_natural_rotation(T_cur, self.human_position)
            self.ee_rotation = True

        # 로봇의 목표 위치 설정
        T_sd = np.eye(4)
        T_sd[:3, :3] = self.rotation_matrix #T_ee[:3,:3] #self.rotation_matrix # T_er[:3, :3]  # 회전 행렬은 단위 행렬로 설정
        det = np.linalg.det(self.rotation_matrix)
        orthogonality_check = np.allclose(self.rotation_matrix.T @ self.rotation_matrix, np.eye(3))

        if not np.isclose(det, 1.0) or not orthogonality_check:
            print("Invalid rotation matrix detected. Normalizing...")
            U, _, Vt = np.linalg.svd(self.rotation_matrix)
            rotation_matrix_normalized = U @ Vt
            T_bd[:3, :3] = rotation_matrix_normalized

        robot_target_position_homogeneous = np.append(self.human_position, 1) 
        robot_target_position_local = T_cur @ robot_target_position_homogeneous
        robot_target_position_ee = robot_target_position_local[:3, 3]  # 3차원으로 변환

        T_sd[0, 3] = (robot_target_position_ee[0] + T_cur[0, 3])/2 #robot_target_position[0]
        T_sd[1, 3] = (robot_target_position_ee[1] + T_cur[1, 3])/2 #robot_target_position[1]
        T_sd[2, 3] = (robot_target_position_ee[2] + T_cur[2, 3])/2 #robot_target_position[2]

        self.desired_trajectory['x'].append(T_sd[0, 3])
        self.desired_trajectory['y'].append(T_sd[1, 3])
        self.desired_trajectory['timestamps'].append(current_time)

        ppppp = list(T_sd[0:3,3])
        # print(ppppp)
        qqqqq = R.from_matrix(T_sd[0:3,0:3]).as_quat()
        # print(qqqqq)
        self.make_tf_msg(ppppp, qqqqq, "world", "desired")

        ppppp_ = list(robot_target_position_ee)
        # print(ppppp)
        qqqqq_ = R.from_matrix(T_sd[0:3,0:3]).as_quat()
        # print(qqqqq)
        self.make_tf_msg(ppppp_, qqqqq_, "world", "human")

        # 각도 계산
        sight_vec = T_cur[:3,0]
        sight_vec /= np.linalg.norm(sight_vec)
        direction_unit_vector = self.human_position - T_cur[:3, 3]
        direction_unit_vector = direction_unit_vector / np.linalg.norm(direction_unit_vector)
        cos_theta = np.dot(direction_unit_vector, sight_vec)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # theta_values.append(np.degrees(theta))

        T_bd = np.linalg.inv(T_sb) @ T_sd  
        # print("T_bd:", T_bd)
        # print("T_bd shape:", T_bd.shape)
        H_desired = SE3(T_bd)  # 목표 end-effector 위치

        F = np.array([[0.0, 1.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0], 
                        [0.0, 0.0],
                        [1.0, 0.0]])

        J_p = base.tr2adjoint(T_be.T) @ F  # 6x2 자코비안 (선형 속도)
        J_a_e = base.tr2adjoint(T_be.T) @ self.ur5e_robot.jacob0(self.q[2:])
        J_mb = np.hstack((J_p, J_a_e))  # 6x8 자코비안 (선형 속도 + 각속도)
        J_mb_v = J_mb[:3, :]  # 3x8 자코비안 (선형 속도)
        J_mb_w = J_mb[3:, :]  # 3x8 자코비안 (각속도)

       

        T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4
        # print(T_error)
        et = np.sum(np.abs(T_error[:3, -1])) 
        # Quadratic component of objective function
        Q = np.eye(self.n_dof + 6)
        # Joint velocity component of Q
        Q[:2, :2] *= 1.0 / max(et * 100, 1e-6)

        # Slack component of Q
        Q[self.n_dof:, self.n_dof:] = (1. / max(et, 1e-6)) * np.eye(6)

        H = np.zeros((self.n_dof-2, 6, self.n_dof-2))  # same as jacobm

        for j in range(self.n_dof-2):
            for i in range(j, self.n_dof-2):
                H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                if i != j:
                        H[i, :3, j] = H[j, :3, i]
                        H[i, 3:, j] = H[j, 3:, i]

        # manipulability only for arm joints
        J_a = self.ur5e_robot.jacob0(self.q[2:])
        m = J_a @ J_a.T 
        m_det = np.linalg.det(m)  
        m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        rank = np.linalg.matrix_rank(J_a @ J_a.T)
        if rank < J_a.shape[0]:
            print("Warning: Jacobian matrix is rank-deficient. Robot may be in a singularity.")
            JJ_inv = np.linalg.pinv(J_a @ J_a.T)  # 유사역행렬 사용
        else:
            JJ_inv = np.linalg.inv(J_a @ J_a.T)  # 역행렬 계산

        # Compute manipulability Jacobian only for arm joints
        J_m = np.zeros((self.n_dof-2,1))
        for i in range(self.n_dof-2):
            c = J_a @ np.transpose(H[i, :, :])  # shape: (6,6)
            J_m[i,0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

        A = np.zeros((self.n_dof + 2 + num_points, self.n_dof + 6))
        B = np.zeros(self.n_dof + 2 + num_points)
        # print(f"Ashape: {A.shape}, B shape: {B.shape}")
        
        J_dj = np.zeros(self.n_dof+6)
        w_p_sum = 0.0
        min_dist_list = []  # 장애물과의 최소 거리 리스트
        g_vec_list = []  # 각 점에서의 장애물 방향 벡터 리스트
        # 초기화
        weighted_g_vec = np.zeros(3)  # 가중치가 적용된 방향 벡터의 합
        total_weight = 0.0  # 총 가중치
        for i , pose in enumerate(xform_pose) :

            distance, index, g_vec_ = self.get_nearest_obstacle_distance(pose, self.obstacles_positions[:, :3], self.obstacle_radius)
            min_dist_ = np.min(distance)
            min_dist_list.append(min_dist_)  # 최소 거리 추가
            g_vec_list.append(g_vec_)  # 해당 최소 거리의 방향 벡터 추가
            # 거리 기반 가중치 계산
            if min_dist_ <= 0.6:
                weight = 0.15  # 거리가 0.3 이하인 경우 동일한 가중치 부여
            else:
                weight = 0. #max(0, 0.1 / (min_dist_ - d_safe + 1e-6))  # 거리의 역수로 가중치 계산
            weighted_g_vec += weight * g_vec_  # 가중치를 곱한 방향 벡터를 합산
            total_weight += weight  # 총 가중치 합산
        # print('min_dist', min_dist_list)
        # 최종 g_vec 계산 (가중치로 정규화)
        if total_weight > 0:
            g_vec = weighted_g_vec / total_weight  # 가중치로 정규화
        else:
            g_vec = np.zeros(3)  # 총 가중치가 0인 경우, 기본값으로 설정
        min_dist = np.min(min_dist_list)
        min_index = np.argmin(min_dist_list)
        # print('min_dist:', min_dist)
        g_vec = g_vec_list[min_index]
        avoid_vec /= np.linalg.norm(g_vec)

        link1_len = (num_mobile + self.ur5e_link_points['shoulder'][0])
        link2_len = (link1_len + self.ur5e_link_points['upper_arm'][0])
        link3_len = (link2_len + self.ur5e_link_points['forearm'][0])
        link4_len = (link3_len + self.ur5e_link_points['wrist_1'][0])
        link5_len = (link4_len + self.ur5e_link_points['wrist_2'][0])
        # link6_len = (link5_len + self.ur5e_link_points['wrist_3'][0])


        for i , pose in enumerate(xform_pose) :    
                if i < num_mobile:  # mobile base wheels

                    d_dot = (avoid_vec) @ J_mb_v   # 장애물 피하는 방향으로의 자코비안
                    
                    A[i, :8] = -d_dot 
                    A[i, 8:] = np.zeros((1, 6)) 
                    B[i] = (min_dist_list[i] - self.d_safe) / ((self.d_influence - self.d_safe))
                    # B[i] = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                    w_p = (self.d_influence-min_dist_list[i])/(self.d_influence - self.d_safe) 
                    # print("min_dist_list[i]:", min_dist_list[i])
                    # print(f"w_p: {w_p}")
                    # w_p = (min_dist_list[i]-d_influence)/(d_influence-d_safe) 
                    J_dj[:8] +=  (-d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                    # print("J_dj[:8]:", J_dj[:8])
                    # print(f"robot {i}th link Distance: {min_dist:.2f} m")
                    # print(f"{i}th J_dj : {A[i, :8] * w_p}")
                    # print(f"{i}th w_p : { w_p}")
                    w_p_sum += np.abs(w_p)
                    # if min_dist < 0.0:
                        # print(f"robot {i+1}th link is too close to an obstacle. Distance: {min_dist:.2f} m")
                        # print(f"A : {A[i, :8]}")
                        # print(f"B : {B[i]:.2f}")
                else:  # UR5e joints
                    if num_mobile <= i < link1_len: 
                        J_mb_arm_v = np.hstack([np.zeros((3, 2)), J_a_e[:3, :]])
                    elif link1_len <= i < link2_len:
                        J_mb_arm_v = np.hstack([np.zeros((3, 3)), J_a_e[:3, 1:]])
                    elif link2_len <= i < link3_len:
                        J_mb_arm_v = np.hstack([np.zeros((3, 4)), J_a_e[:3, 2:]])
                    elif link3_len <= i < link4_len:
                        J_mb_arm_v = np.hstack([np.zeros((3, 5)), J_a_e[:3, 3:]])
                    elif link4_len <= i < link5_len:
                        J_mb_arm_v = np.hstack([np.zeros((3, 6)), J_a_e[:3, 4:]])
                    else : 
                        J_mb_arm_v = np.hstack([np.zeros((3, 7)), J_a_e[:3, 5:]])

                    d_dot = (avoid_vec) @ J_mb_arm_v

                    A[i, :8] = -d_dot
                    A[i, 8:] = np.zeros((1, 6)) 
                    B[i] = (min_dist_list[i] - self.d_safe) / ((self.d_influence - self.d_safe))
                    # B[i] = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                    w_p = (self.d_influence-min_dist_list[i])/(self.d_influence - self.d_safe) 
                    # print("min_dist_list[i]:", min_dist_list[i])
                    # print(f"w_p: {w_p}")
                    # w_p = (min_dist_list[i]-d_influence)/(d_influence-d_safe) 
                    J_dj[:8] += (-d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                    # print("J_dj[:8]:", J_dj[:8])
                    # print(f"robot {i}th link Distance: {min_dist:.2f} m")
                    # print(f"{i}th J_dj : {A[i, :8] * w_p}")
                    # print(f"{i}th w_p : { w_p}")
                    w_p_sum += np.abs(w_p)


        C1 = np.concatenate((np.zeros(2), -J_m.reshape((self.n_dof - 2,)), np.zeros(6)))
        # bTe = self.ur5e_robot.fkine(self.q[2:], include_base=False).A 
        # θε = atan2(bTe[1, -1], bTe[0, -1])
        try:
            bTe = self.ur5e_robot.fkine(self.q[2:], include_base=False).A
            θε = atan2(bTe[1, -1], bTe[0, -1])
            # print(f"θε: {θε}")
            C2 = np.zeros(self.n_dof + 6)
            C2[0] = -5. * θε
        except:
            # print('nonono')
            C2 = np.zeros(self.n_dof + 6)
       
        lambda_max = 5.
        min_distance = np.min(min_dist_list)  # 장애물과의 최소 거리
        if min_distance <= self.d_influence :
            lambda_c = (lambda_max /(self.d_influence - self.d_safe)**2) * (min_distance - self.d_influence)**2
        else:
            lambda_c = 0.0
        J_c = lambda_c * J_dj/w_p_sum
        # print("w_p_sum:", w_p_sum)
        C3 = J_c # 베이스 조인트 속도에 대한 제약 조건 추가
        # 장애물 회피 (간단화)
        # C3 = np.zeros(self.n_dof + 6)
    
        
        # 회전 제어 항
        J_h = np.zeros(self.n_dof + 6)
        J_mb_w_h = direction_unit_vector @ J_mb_w

        epsilon = 1e-6
        lambda_h = self.lambda_h_a_param * max(abs(theta), epsilon)
        J_h[:8] = lambda_h * J_mb_w_h

        C4 = J_h
        C =  self.w1 * C1 + self.w2 * C2 + self.w3 * C3 + self.w4 * C4   # 베이스 조인트 속도에 대한 제약 조건 추가

        J_ = np.c_[J_mb, np.eye(6)]  # J_ 행렬 (예시)

        eTep = T_error  # 현재 위치에서의 오차 행렬

        e = np.zeros(6)

        # Translational error
        e[:3] = eTep[:3, -1]

        # Angular error
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        # perint(f"e: {e}")
        k = np.eye(6)  # gain
        # k[:3,:] *= 8.0 # gain
        v = k @ e
        # v[3:] *= 1.3

        lb = -np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        # print(f"Qshape: {Q.shape}, C shape: {C.shape}, A shape: {A.shape}, B shape: {B.shape}, J_ shape: {J_.shape}, v shape: {v.shape}, lb shape: {lb.shape}, ub shape: {ub.shape}")
        # qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')

     
        
        # qd = [vc, wc, qd1, qd2, qd3, qd4, qd5, qd6]
        # qd = qd[:8]
        # print(f"qd: {qd}")

        # if qd is None:
        #     print("QP solution is None")
        #     qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.]) 


        x_ = cp.Variable(self.n_dof+6)
        objective = cp.Minimize(0.5 * cp.quad_form(x_, Q) + C.T @ x_)
        constraints = [
            x_ >= lb,
            x_ <= ub,
            J_ @ x_ == v,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if x_.value is not None:
            qd = x_.value
        else:
            print("QP solution is None")
            qd = np.zeros(self.n_dof+6)

        if et < self.target_reached_threshold:
            qd = qd[: self.n_dof]
            qd = 0 * qd


        try:
            # use elapsed time as x-axis
            t_now = time.time() - self.start_time
            # ensure numpy arrays
            x_vec = np.array(qd).reshape(-1)
            # quadratic term
            Qterm = 0.5 * float(x_vec.T @ Q @ x_vec)
            # individual C parts (ensure C1..C4 exist in scope)
            c1x = float(C1 @ x_vec) if 'C1' in locals() or 'C1' in globals() else 0.0
            c2x = float(C2 @ x_vec) if 'C2' in locals() or 'C2' in globals() else 0.0
            c3x = float(C3 @ x_vec) if 'C3' in locals() or 'C3' in globals() else 0.0
            c4x = float(C4 @ x_vec) if 'C4' in locals() or 'C4' in globals() else 0.0
            Csum = c1x + c2x + c3x + c4x
            total_obj = Qterm + Csum

            self.obj_timestamps.append(t_now)
            self.obj_Qterm.append(Qterm)
            self.obj_C1.append(c1x)
            self.obj_C2.append(c2x)
            self.obj_C3.append(c3x)
            self.obj_C4.append(c4x)
            self.obj_total.append(total_obj)
        except Exception as e:
            # don't break control loop if logging fails
            print(f"Objective logging failed: {e}")


        wc, vc = qd[0], qd[1]  # 베이스 속도
        qdc = qd[2:]
        # print('qd:', qd)
        # moving base
        twist = Twist()
        twist.linear.x = vc
        twist.angular.z = wc
        # self.scout_publisher.publish(twist)

        # moving arm
        # self.rtde_c.speedJ(qdc, 0.2, self.dt)
        # self.rtde_c.waitPeriod(t_start)

        # joint_vel = JointState()
        # joint_vel.velocity = qd[2:]
        # self.ur5e_publisher.publish(joint_vel)


    def plot_objective_time_series(self, save_path=None, show_plot=False):
        """Plot objective value over time and stacked C1..C4 contributions.
           x axis = elapsed time (s). Stacked area shows contributions of C1..C4,
           overlay Q-term and total objective as lines.
        """
        try:
            if len(self.obj_timestamps) == 0:
                print("No objective history to plot.")
                return

            import matplotlib.pyplot as plt
            ts = np.array(self.obj_timestamps)
            c1 = np.array(self.obj_C1)
            c2 = np.array(self.obj_C2)
            c3 = np.array(self.obj_C3)
            c4 = np.array(self.obj_C4)
            qterm = np.array(self.obj_Qterm)
            total = np.array(self.obj_total)

            # stacked contributions (positive/negative handled by stacking absolute sign but keep sign)
            contributions = np.vstack([c1, c2, c3, c4])

            plt.switch_backend('Agg')  # safe when no display

            fig, ax = plt.subplots(figsize=(10, 6))

            # stacked area for C parts
            labels = ['C1·x', 'C2·x', 'C3·x', 'C4·x']
            colors = ['#ffd700', '#ffa07a', '#87cefa', '#90ee90']
            # stackplot expects sequences of shape (n_series, n_points)
            ax.stackplot(ts, contributions, labels=labels, colors=colors, alpha=0.8)

            # overlay Q-term and total objective
            ax.plot(ts, qterm, color='k', linestyle='--', linewidth=1.5, label='0.5 x^T Q x')
            ax.plot(ts, total, color='r', linestyle='-', linewidth=2, label='Total objective')

            # also plot cumulative sums requested: c1, c1+c2, c1+c2+c3, c1+c2+c3+c4
            cum1 = c1
            cum2 = c1 + c2
            cum3 = c1 + c2 + c3
            cum4 = c1 + c2 + c3 + c4
            ax.plot(ts, cum1, color='#b8860b', linestyle=':', linewidth=1, label='C1')
            ax.plot(ts, cum2, color='#ff7f50', linestyle=':', linewidth=1, label='C1+C2')
            ax.plot(ts, cum3, color='#4682b4', linestyle=':', linewidth=1, label='C1+C2+C3')
            ax.plot(ts, cum4, color='#2e8b57', linestyle=':', linewidth=1, label='C1+C2+C3+C4')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Objective value (scalar)')
            ax.set_title('QP objective and decomposition over time')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)

            plt.tight_layout()

            if save_path is None:
                # default save location near script
                save_path = os.path.expanduser(f'./objective_time_series_{int(time.time())}.png')
            plt.savefig(save_path, dpi=200)
            plt.close(fig)

            print(f"Objective plot saved: {save_path}")
            if show_plot:
                # attempt to open using default viewer (may not be available in headless env)
                try:
                    from subprocess import Popen
                    Popen(['xdg-open', save_path])
                except Exception:
                    pass

        except Exception as e:
            print(f"Failed to plot objective time series: {e}")

    def save_ee_trajectory(self):
        """End effector 및 desired 궤적을 파일로 저장"""
        if len(self.ee_trajectory['x']) == 0:
            print("📊 기록된 궤적이 없습니다.")
            return
            
        # 저장 디렉토리 생성
        save_dir = "/home/nvidia/geon/robotics/jetson_orin/controller/mb_control/hand_trajectories_plots"
        os.makedirs(save_dir, exist_ok=True)
        
        # 타임스탬프로 파일명 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # matplotlib 백엔드를 'Agg'로 설정 (GUI 없이 파일로만 저장)
        plt.switch_backend('Agg')

        plt.figure(figsize=(8, 8))
        plt.plot(self.ee_trajectory['x'], self.ee_trajectory['y'], '-',color ='c',  linewidth=2, alpha=0.7, label='Actual EE Trajectory')
        plt.plot(self.base_trajectory['x'], self.base_trajectory['y'], '-',color='orange', linewidth=2, alpha=0.7, label='Actual Mobile Base Trajectory')
        
        # Desired 궤적 추가
        if len(self.desired_trajectory['x']) > 0:
            plt.plot(self.desired_trajectory['x'], self.desired_trajectory['y'], 'b--', linewidth=2, alpha=0.8, label='Desired Trajectory')
            # Desired waypoints 마커
            plt.scatter(self.desired_trajectory['x'], self.desired_trajectory['y'], 
                       color='#87CEFA', s=80, marker='x', alpha=0.8, zorder=5, label='Desired Points')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Actual and Desired XY Trajectory')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plot_filename_ = f"{save_dir}/trajectories_{self.w1}_{self.w2}_{self.w4}_{timestamp}.png"
        plt.savefig(plot_filename_, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 10))
        
        # 궤적 플롯
        plt.subplot(2, 3, 1)
        plt.plot(self.ee_trajectory['x'], self.ee_trajectory['y'], 'b-', linewidth=2, alpha=0.7, label='Actual EE Trajectory')
        
        # Desired 궤적 추가
        if len(self.desired_trajectory['x']) > 0:
            plt.plot(self.desired_trajectory['x'], self.desired_trajectory['y'], 'r--', linewidth=2, alpha=0.8, label='Desired Trajectory')
            # Desired waypoints 마커
            plt.scatter(self.desired_trajectory['x'], self.desired_trajectory['y'], 
                       color='red', s=80, marker='x', alpha=0.8, zorder=5, label='Desired Points')
            
            # Desired trajectory 시작/끝점
            if len(self.desired_trajectory['x']) > 0:
                plt.scatter(self.desired_trajectory['x'][0], self.desired_trajectory['y'][0], 
                           color='orange', s=120, marker='^', label='Desired Start', zorder=6)
                plt.scatter(self.desired_trajectory['x'][-1], self.desired_trajectory['y'][-1], 
                           color='darkred', s=120, marker='v', label='Desired End', zorder=6)
        
        # Actual trajectory 시작/끝점
        plt.scatter(self.ee_trajectory['x'][0], self.ee_trajectory['y'][0], 
                   color='green', s=100, marker='o', label='Actual Start', zorder=5)
        plt.scatter(self.ee_trajectory['x'][-1], self.ee_trajectory['y'][-1], 
                   color='blue', s=100, marker='s', label='Actual End', zorder=5)
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Actual and Desired XY Trajectory')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # X 좌표 시간 변화
        plt.subplot(2, 3, 2)
        plt.plot(self.ee_trajectory['timestamps'], self.ee_trajectory['x'], 'b-', linewidth=2, label='Actual X')
        if len(self.desired_trajectory['x']) > 0:
            plt.plot(self.desired_trajectory['timestamps'], self.desired_trajectory['x'], 'r--', linewidth=2, label='Desired X')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X Position over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Y 좌표 시간 변화
        plt.subplot(2, 3, 3)
        plt.plot(self.ee_trajectory['timestamps'], self.ee_trajectory['y'], 'b-', linewidth=2, label='Actual Y')
        if len(self.desired_trajectory['y']) > 0:
            plt.plot(self.desired_trajectory['timestamps'], self.desired_trajectory['y'], 'r--', linewidth=2, label='Desired Y')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 속도 분석
        plt.subplot(2, 3, 4)
        if len(self.ee_trajectory['x']) > 1:
            dx = np.diff(self.ee_trajectory['x'])
            dy = np.diff(self.ee_trajectory['y'])
            dt = np.diff(self.ee_trajectory['timestamps'])
            dt[dt == 0] = 1e-6  # divide by zero 방지
            
            vx = dx / dt
            vy = dy / dt
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            plt.plot(self.ee_trajectory['timestamps'][1:], v_magnitude, 'purple', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity Magnitude (m/s)')
            plt.title('EE Velocity Magnitude')
            plt.grid(True, alpha=0.3)
        
        # 추적 오차 분석
        plt.subplot(2, 3, 5)
        if len(self.desired_trajectory['x']) > 0:         
            # Desired trajectory를 actual trajectory 시간에 맞춰 보간
            if len(self.desired_trajectory['timestamps']) > 1:
            
                # 위치 오차 계산
                position_error = np.sqrt((self.ee_trajectory['x'][-1] - self.desired_trajectory['x'][-1])**2 + 
                                       (self.ee_trajectory['y'][-1] - self.desired_trajectory['y'][-1])**2)
                
                plt.plot(self.ee_trajectory['timestamps'], position_error, 'purple', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Position Error (m)')
                plt.title('Tracking Error')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Insufficient desired\ntrajectory data', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Tracking Error')
        else:
            plt.text(0.5, 0.5, 'No desired trajectory\ndata available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Tracking Error')
        
        # 웨이포인트 도달 시간 분석
        plt.subplot(2, 3, 6)
        if len(self.desired_trajectory['timestamps']) > 1:
            waypoint_durations = np.diff(self.desired_trajectory['timestamps'])
            plt.bar(range(len(waypoint_durations)), waypoint_durations, color='orange', alpha=0.7)
            plt.xlabel('Waypoint Transition')
            plt.ylabel('Duration (s)')
            plt.title('Time Between Waypoints')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient waypoint\ndata for analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Time Between Waypoints')
        
        plt.tight_layout()
        
        # 그래프 저장
        plot_filename = f"{save_dir}/ee_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Actual trajectory CSV로 저장
        csv_filename = f"{save_dir}/ee_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            f.write("timestamp,actual_x,actual_y\n")
            for i in range(len(self.ee_trajectory['x'])):
                f.write(f"{self.ee_trajectory['timestamps'][i]:.3f},"
                       f"{self.ee_trajectory['x'][i]:.6f},"
                       f"{self.ee_trajectory['y'][i]:.6f}\n")
        
        # Desired trajectory CSV로 저장
        desired_csv_filename = f"{save_dir}/desired_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.csv"
        if len(self.desired_trajectory['x']) > 0:
            with open(desired_csv_filename, 'w') as f:
                f.write("timestamp,desired_x,desired_y\n")
                for i in range(len(self.desired_trajectory['x'])):
                    f.write(f"{self.desired_trajectory['timestamps'][i]:.3f},"
                           f"{self.desired_trajectory['x'][i]:.6f},"
                           f"{self.desired_trajectory['y'][i]:.6f}\n")
        
        # 통계 정보 출력 및 저장
        total_distance = 0
        if len(self.ee_trajectory['x']) > 1:
            for i in range(1, len(self.ee_trajectory['x'])):
                dx = self.ee_trajectory['x'][i] - self.ee_trajectory['x'][i-1]
                dy = self.ee_trajectory['y'][i] - self.ee_trajectory['y'][i-1]
                total_distance += np.sqrt(dx**2 + dy**2)
        
        x_range = max(self.ee_trajectory['x']) - min(self.ee_trajectory['x'])
        y_range = max(self.ee_trajectory['y']) - min(self.ee_trajectory['y'])
        total_time = self.ee_trajectory['timestamps'][-1] - self.ee_trajectory['timestamps'][0]
        
        # 추적 성능 통계 계산
        avg_tracking_error = 0
        max_tracking_error = 0
        num_waypoints = len(self.desired_trajectory['x'])
        
        if len(self.desired_trajectory['x']) > 1:
            try:
                import scipy.interpolate
                f_x = scipy.interpolate.interp1d(self.desired_trajectory['timestamps'], self.desired_trajectory['x'], 
                                               kind='linear', fill_value='extrapolate')
                f_y = scipy.interpolate.interp1d(self.desired_trajectory['timestamps'], self.desired_trajectory['y'], 
                                               kind='linear', fill_value='extrapolate')
                
                desired_x_interp = f_x(self.ee_trajectory['timestamps'])
                desired_y_interp = f_y(self.ee_trajectory['timestamps'])
                
                tracking_errors = np.sqrt((np.array(self.ee_trajectory['x']) - desired_x_interp)**2 + 
                                        (np.array(self.ee_trajectory['y']) - desired_y_interp)**2)
                
                avg_tracking_error = np.mean(tracking_errors)
                max_tracking_error = np.max(tracking_errors)
            except:
                pass
        
        stats_filename = f"{save_dir}/trajectory_stats_{timestamp}.txt"
        stats_info = f"""궤적 추종 성능 통계 정보
=====================================
기록 시간: {timestamp}
제어 파라미터: w1={self.w1}, w2={self.w2}, w4={self.w4}

=== Actual Trajectory ===
총 이동 거리: {total_distance:.3f} m
X 축 범위: {x_range:.3f} m
Y 축 범위: {y_range:.3f} m
총 시간: {total_time:.1f} s
평균 속도: {total_distance/max(total_time, 1e-6):.3f} m/s
기록된 포인트 수: {len(self.ee_trajectory['x'])} 개

=== Desired Trajectory ===
웨이포인트 수: {num_waypoints} 개
첫 번째 목표: ({self.desired_trajectory['x'][0]:.3f}, {self.desired_trajectory['y'][0]:.3f}) (시작 시 0초) if num_waypoints > 0 else (없음)
마지막 목표: ({self.desired_trajectory['x'][-1]:.3f}, {self.desired_trajectory['y'][-1]:.3f}) (시간 {self.desired_trajectory['timestamps'][-1]:.1f}초) if num_waypoints > 0 else (없음)

=== 추종 성능 ===
평균 추적 오차: {avg_tracking_error:.3f} m
최대 추적 오차: {max_tracking_error:.3f} m
목표 도달 임계값: {self.target_reached_threshold:.3f} m
"""
        
        with open(stats_filename, 'w') as f:
            f.write(stats_info)
        
        print(f"\n📊 궤적 추종 결과 저장 완료:")
        print(f"   그래프: {plot_filename}")
        print(f"   Actual 데이터: {csv_filename}")
        if len(self.desired_trajectory['x']) > 0:
            print(f"   Desired 데이터: {desired_csv_filename}")
        print(f"   통계: {stats_filename}")
        print(f"   총 이동 거리: {total_distance:.3f} m")
        print(f"   웨이포인트 수: {num_waypoints} 개")
        print(f"   평균 추적 오차: {avg_tracking_error:.3f} m")
        print(f"   최대 추적 오차: {max_tracking_error:.3f} m")
        print(f"   총 시간: {total_time:.1f} s")

    def cleanup_and_plot(self):
        """정리 작업 및 궤적 저장"""
        print("\n🛑 프로그램 종료 중...")
        self.save_ee_trajectory()
        try:
            self.plot_objective_time_series()
        except Exception as e:
            print(f"Failed to save objective plot: {e}")

if __name__ == '__main__':
    # 시그널 핸들러 등록
    
    rclpy.init()
    node = QP_mbcontorller()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 감지됨")
    finally:
        # cleanup은 rclpy.ok()가 참일 때만 실행합니다.
        if rclpy.ok():
            print("✨ ROS 2 종료 정리 시작...")
            node.cleanup_and_plot() # 단 한번 저장
            node.destroy_node()
            rclpy.shutdown()
        else:
            # rclpy.shutdown()이 이미 호출되었을 경우 (안전 장치)
            print("✨ ROS 2 이미 종료됨. 추가 정리 생략.")