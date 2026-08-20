import os
from datetime import datetime
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
import numpy as np

# 파라미터 설정
w1 = 1.0 #{w1_val}
w2 = 0.4  #{w2_val}
w3 = 0.5 #{w3_val}
w4 = 0.2 #{w4_val}
obstacles_positions = np.array([[0.8,0.8, 0.],[0.95,0.8, 0.],[1.1,0.8, 0.],[1.25,0.8, 0.],[1.4,0.8, 0.],[1.55,0.8, 0.], [1.7,0.8, 0.]])
obstacle_radius = 0.075
obstacle_height = 0.6
lambda_h_a_param = 0.5 #{lambda_h_a_val}
trj_start_point = (1.0, 0.9, 1.0)
trj_end_point = (2.0, 0.9, 1.0)  #(trj_end_point_val)
# combo_index = {combo_index_val}
# total_combos = {total_combos_val}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "low_obstacle_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FOLDER = OUTPUT_DIR

# print("Running Simulation " + str(combo_index) + "/" + str(total_combos))
print("Parameters: w1=" + str(w1) + ", w2=" + str(w2) + ", w4=" + str(w4) + ", lambda_h_a=" + str(lambda_h_a_param))

# 시뮬레이션 앱 시작
simulation_app = SimulationApp({"headless": False})

try:
    # 모든 import 구문들
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.api import World, SimulationContext
    from isaacsim.core.api.objects import DynamicCylinder, VisualSphere, VisualCylinder, FixedCylinder
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import Articulation, XFormPrim
    from isaacsim.core.utils.types import ArticulationActions
    from isaacsim.core.utils.prims import create_prim
    from pxr import UsdGeom, Gf
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    from roboticstoolbox import DHRobot, RevoluteDH

    from scipy.spatial.transform import Rotation as R
    from math import atan2 as atan2
    import qpsolvers as qp
    from spatialmath import base, SE3
    import roboticstoolbox as rtb
    import matplotlib.pyplot as plt
    import random
    from straight_planner import LinearTrajectoryPlanner  # 직선 궤적 플래너 임포트
    import cvxpy as cp
    from mpl_toolkits.mplot3d import Axes3D
    from carb._carb import Float3, ColorRgba
    from spatialmath.base import trnorm

    # 함수 정의들
    def calculate_natural_rotation(T_cur, target_position):
        """
        현재 자세에서 최소 회전으로 타겟을 향하는 회전 행렬 계산.
        
        Args:
            T_cur: 현재 변환 행렬 (4x4)
            target_position: 목표 위치 (3,)
        
        Returns:
            회전 행렬 (3x3)
        """
        cur_R = T_cur[:3, :3]
        cur_z_axis = cur_R[:, 2]  # 현재 Z축(바라보는 방향)
        current_position = T_cur[:3, 3]
        
        # 타겟 방향 벡터
        direction_vector = target_position - current_position
        direction_vector /= np.linalg.norm(direction_vector)
        
        # 현재 Z축과 타겟 방향 사이의 각도
        cos_angle = np.dot(cur_z_axis, direction_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 회전축 (Z축에서 타겟 방향으로의 최단 경로)
        if angle < 1e-6:
            # 이미 타겟을 향하고 있음
            return cur_R
        
        rotation_axis = np.cross(cur_z_axis, direction_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)
        
        # Rodrigues 공식으로 회전 행렬 계산 (최소 회전)
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # 새로운 회전: 현재 회전에 델타 회전을 적용
        new_R = R_delta @ cur_R
        
        return new_R
    def calculate_natural_rotation_xy_only(T_cur, target_position):
        """
        XY 평면에서만 회전 (Z축 높이 유지).
        로봇이 현재 높이를 유지하면서 타겟을 바라봄.
        """
        current_position = T_cur[:3, 3]
        
        # XY 평면에서의 방향 벡터
        direction_xy = target_position[:2] - current_position[:2]
        direction_xy /= np.linalg.norm(direction_xy)
        
        # 각도 계산
        yaw = np.arctan2(direction_xy[1], direction_xy[0])
        
        # Z축 회전만 적용
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return R_z

    def transform_points(points, position, quaternion):
        """
        점들을 변환 행렬을 사용하여 변환하는 함수.

        Args:
            points (np.ndarray): 변환할 점들의 로컬 좌표 (shape: num_points x 3).
            position (tuple): 변환 행렬의 위치 (x, y, z).
            quaternion (tuple): 변환 행렬의 회전 (쿼터니언).

        Returns:
            np.ndarray: 변환된 점들의 월드 좌표 (shape: num_points x 3).
        """
        # 변환 행렬 생성
        rotation_matrix = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
        translation = np.array(position)
        transformed_points = (rotation_matrix @ points.T).T + translation
        return transformed_points
    
    def generate_rectangle_edge_points(center, quat, width, height, num_points_per_edge):
        """
        직사각형의 네 변에 점들을 생성하고 월드 좌표로 변환하는 함수.

        Args:
            center (tuple or np.ndarray): 직사각형의 중심 좌표 (x, y, z) - 월드 좌표.
            quat (tuple or np.ndarray): 중심의 회전 (쿼터니언: w, x, y, z 또는 x, y, z, w).
            width (float): 직사각형의 너비 (로컬 X 방향).
            height (float): 직사각형의 높이 (로컬 Y 방향).
            num_points_per_edge (int): 각 변에 생성할 점의 개수.

        Returns:
            np.ndarray: 직사각형 네 변의 점들의 월드 좌표 (shape: num_points x 3).
        """
        half_width = width / 2
        half_height = height / 2

        # 로컬 좌표계에서 직사각형 네 변의 점 생성 (XY 평면, Z=0)
        top_edge = np.linspace([-half_width, half_height, 0], [half_width, half_height, 0], num_points_per_edge)
        bottom_edge = np.linspace([-half_width, -half_height, 0], [half_width, -half_height, 0], num_points_per_edge)
        left_edge = np.linspace([-half_width, -half_height, 0], [-half_width, half_height, 0], num_points_per_edge)
        right_edge = np.linspace([half_width, -half_height, 0], [half_width, half_height, 0], num_points_per_edge)

        # 모든 점 합치기 (로컬 좌표)
        points_local = np.vstack([top_edge, bottom_edge, left_edge, right_edge])

        # 쿼터니언을 회전 행렬로 변환
        # Isaac Sim은 보통 (w, x, y, z) 순서 사용, scipy는 (x, y, z, w)
        # 입력 quat 형식 확인 필요 - 여기서는 Isaac Sim 형식 (w, x, y, z) 가정
        rotation_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        
        # center를 numpy 배열로 변환
        center = np.asarray(center, dtype=float)
        
        # 로컬 좌표를 월드 좌표로 변환: P_world = R @ P_local + center
        points_world = (rotation_matrix @ points_local.T).T + center
        
        return points_world
    
    def get_nearest_obstacle_distance(position, obstacles, obstacle_radius, T_e):
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
            if position[2] > obstacle_height:
                obs[2] = obstacle_height
            else:
                obs[2] = position[2]
            obs_real.append(obs)
            obs_homogeneous = np.append(obs, 1)  # 동차 좌표로 확장
            obs_local = np.linalg.inv(T_e) @ obs_homogeneous
            obstacles_local.append(obs_local[:3])  # 3차원으로 변환
            
            
        position_homogeneous = np.append(position, 1)  # 동차 좌표로 확장
        position_local = np.linalg.inv(T_e) @ position_homogeneous
        position_local = position_local[:3]  # 3차원으로 변환
        
        distances = []
        for obse in obstacles_local:
            vec_to_obs = obse - position_local  # 점에서 장애물 중심으로의 벡터
            dist_center = np.linalg.norm(vec_to_obs)  # 중심까지 거리
            signed_dist = dist_center - obstacle_radius  # 양수: 외부, 음수: 내부
            distances.append(signed_dist)
        index = np.argmin(distances)

        g_vec = (position - obs_real[index])
        g_vec /= np.linalg.norm(g_vec) 
        # print('g_vec: ', g_vec)
        return distances, index, g_vec
    
    def sagging_curve_points(p0, p1, num_points=10, sag=0.2):
        """
        p0, p1: shape (3,) numpy 배열 (world 좌표)
        sag   : 처짐 정도 (거리의 비율; 0.0이면 직선)
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        t = np.linspace(0.0, 1.0, num_points)

        # xy는 직선 보간
        xy = p0[:2] * (1 - t)[:, None] + p1[:2] * t[:, None]

        # z는 포물선으로 아래로 처짐
        z0, z1 = p0[2], p1[2]
        horiz_dist = np.linalg.norm(p1[:2] - p0[:2])
        z_sag = sag * horiz_dist * t * (1 - t)  # 가운데가 가장 아래
        z = z0 * (1 - t) + z1 * t - z_sag

        points = np.column_stack([xy, z])
        return points
    
    # 데이터 저장용 리스트들
    et_values = []
    et_x_values = []
    et_y_values = []
    et_z_values = []
    time_values = []
    theta_values = []
    scatter_x = []
    scatter_y = []
    scatter_mob_x = []
    scatter_mob_y = []


    cylinders = []
    for i in range(len(obstacles_positions)):
        cylinder = FixedCylinder(
            prim_path=f"/World/Xform/Cylinder{i+1}",
            name=f"cylinder{i+1}",
            position=obstacles_positions[i][:3],
            radius=obstacle_radius,
            height=obstacle_height,
            color=np.array([0.8, 0.2, 0.2])
        )
        cylinders.append(cylinder)
    # 시뮬레이션 설정
    ur5e_robot = rtb.models.UR5()
    world = World(stage_units_in_meters=1.0)
    scene = world.scene
    
    # 로봇 로드
    robot_asset_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"
    prim_path = "/World/aljnu_mp"
    add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)

    # 파라미터 설정
    aljnu_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    aljnu_body_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    n_dof = 8
    qdlim = np.array([0.7]*8)
    qdlim[:1] = 1.5
    qdlim[1] = 0.6

    world.scene.add_default_ground_plane(z_position=-0.2)
    world.reset()

    # 궤적 생성 (원형 → 직선으로 변경)
    planner = LinearTrajectoryPlanner(start=trj_start_point, end=trj_end_point)  # Z축 고도 1.0m 유지
    traj_num_points = 600  # 테스트용으로 줄임
    human_smoothed_path = planner.generate_linear_path(num_points=traj_num_points)
    human_velocity_profile = planner.generate_velocity_profile(human_smoothed_path, v_max=0.2)
    trajectories = planner.generate_full_trajectory_with_offset(human_smoothed_path, human_velocity_profile, dt=0.1, time_offset=15)
    human_trajectory = trajectories
    human_trajectory_index = 0
    # 시각화 객체 생성
    human_goal_position = np.array([3.5, 2.0, 1.2])
    desired_sphere = VisualSphere(
        prim_path="/World/Xform/sphere",
        name="desired_sphere",
        position=human_goal_position,
        radius=0.02,
        color=np.array([0.2, 0.8, 0.2])
    )
    
    human_position = np.array([0.5922, 0.1332, 0.97])
    human_sphere = VisualSphere(
        prim_path="/World/Xform/human_sphere",
        name="human_sphere",
        position=human_position,
        radius=0.02,
        color=np.array([0.2, 0.2, 0.8])
    )

    # 로봇 초기화
    my_robot = Articulation(prim_path)
    my_robot.initialize()
    
    aljnu_indices = aljnu_body_indices[5:]
    values = np.ones((1, len(aljnu_indices)), dtype=bool)
    my_robot.set_body_disable_gravity(values, indices=[0], body_indices=aljnu_indices)

    joints_default_positions = my_robot.get_joint_positions()
    target_positions = np.copy(joints_default_positions)
    target_positions[0][5] = -np.pi/2
    target_positions[0][6] = np.pi/2
    target_positions[0][8] = np.pi/2
    target_joint_positions = target_positions[0][aljnu_joint_indices]

    # 시뮬레이션 변수들
    reached_default = False
    position_tolerance = 0.1
    T_robot = None
    start_t = None
    H_desired = None
    # collision avoidance parameters
    d_safe = 0.1
    d_influence = 0.5
    # 메인 시뮬레이션 루프
    max_iterations = 5000  # 테스트용으로 줄임
    iteration_count = 0
    
    print("Starting simulation loop...")
    # 직사각형 점 생성
    width = 1.1  # 700 mm
    height = 0.7  # 930 mm
    num_points_per_edge = 5  # 각 변에 5개의 점 생성
    
    
    while simulation_app.is_running() and iteration_count < max_iterations:
        iteration_count += 1
        world.step(render=True)
        
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                reached_default = False

            current_joint_positions = my_robot.get_joint_positions()[0][aljnu_joint_indices]
            mobile_base_pose, mobile_base_quat = my_robot.get_world_poses(indices=[0])
            x = mobile_base_pose[0][0]
            y = mobile_base_pose[0][1]
            z = mobile_base_pose[0][2]

            quat = mobile_base_quat[0]
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

            q = np.zeros(8)
            q[0] = 0.0
            q[1] = 0.0
            q[2:] = current_joint_positions[4:10]

            if not reached_default:
                actions = ArticulationActions(
                    joint_positions=target_joint_positions,
                    joint_indices=aljnu_joint_indices
                )
                my_robot.apply_action(actions)

                if np.all(np.abs(current_joint_positions[4:10] - target_joint_positions[4:10]) < position_tolerance):
                    reached_default = True
                    my_robot.switch_control_mode("velocity")
                    print("Reached default position!")
            else:
                # mobile base links positions and orientations
                rear_right_wheel_link = "/World/aljnu_mp/rear_right_wheel_link"
                prim_rear_right_wheel = XFormPrim(rear_right_wheel_link)
                rear_right_wheel_pose, rear_right_wheel_quat = prim_rear_right_wheel.get_world_poses()
                rear_left_wheel_link = "/World/aljnu_mp/rear_left_wheel_link"
                prim_rear_left_wheel = XFormPrim(rear_left_wheel_link)
                rear_left_wheel_pose, rear_left_wheel_quat = prim_rear_left_wheel.get_world_poses()
                front_right_wheel_link = "/World/aljnu_mp/front_right_wheel_link"
                prim_front_right_wheel = XFormPrim(front_right_wheel_link)
                front_right_wheel_pose, front_right_wheel_quat = prim_front_right_wheel.get_world_poses()
                front_left_wheel_link = "/World/aljnu_mp/front_left_wheel_link"
                prim_front_left_wheel = XFormPrim(front_left_wheel_link)
                front_left_wheel_pose, front_left_wheel_quat = prim_front_left_wheel.get_world_poses()    
                # ur5e joint links positions and orientations
                ur5e_shoulder_link = "/World/aljnu_mp/ur5e_shoulder_link"
                prim_shoulder = XFormPrim(ur5e_shoulder_link)
                ur5e_shoulder_pose, ur5e_shoulder_quat = prim_shoulder.get_world_poses()
                ur5e_upper_arm_link = "/World/aljnu_mp/ur5e_upper_arm_link"
                prim_upper_arm = XFormPrim(ur5e_upper_arm_link)
                ur5e_upper_arm_pose, ur5e_upper_arm_quat = prim_upper_arm.get_world_poses()
                ur5e_forearm_link = "/World/aljnu_mp/ur5e_forearm_link"
                prim_forearm = XFormPrim(ur5e_forearm_link)
                ur5e_forearm_pose, ur5e_forearm_quat = prim_forearm.get_world_poses()
                ur5e_wrist_1_link = "/World/aljnu_mp/ur5e_wrist_1_link"
                prim_wrist_1 = XFormPrim(ur5e_wrist_1_link)
                ur5e_wrist_1_pose, ur5e_wrist_1_quat = prim_wrist_1.get_world_poses()
                ur5e_wrist_2_link = "/World/aljnu_mp/ur5e_wrist_2_link"
                prim_wrist_2 = XFormPrim(ur5e_wrist_2_link)
                ur5e_wrist_2_pose, ur5e_wrist_2_quat = prim_wrist_2.get_world_poses()
                ur5e_wrist_3_link = "/World/aljnu_mp/ur5e_wrist_3_link"
                prim_wrist_3 = XFormPrim(ur5e_wrist_3_link)
                ur5e_wrist_3_pose, ur5e_wrist_3_quat = prim_wrist_3.get_world_poses()
                base_link = "/World/aljnu_mp/base_link"
                prim_base = XFormPrim(base_link)
                base_pose, base_quat = prim_base.get_world_poses()

                # rectangle_points_world를 xform_pose 앞에 추가
                rectangle_edge_points_local = generate_rectangle_edge_points(center=base_pose[0], quat = base_quat[0],  width=width, height=height,
                                                             num_points_per_edge=num_points_per_edge)
                # rectangle_edge_points_local 시각화 (VisualSphere 사용)
                # for i, point in enumerate(rectangle_edge_points_local):
                #     sphere = VisualSphere(
                #         prim_path=f"/World/Xform/rectangle_point_{i}",
                #         name=f"rectangle_point_{i}",
                #         position=point,
                #         radius=0.02,
                #         color=np.array([1.0, 0.5, 0.0])  # 주황색
                #     )
                xform_pose = np.vstack((rectangle_edge_points_local, np.array([
                    rear_right_wheel_pose[0], 
                    rear_left_wheel_pose[0], 
                    front_right_wheel_pose[0], 
                    front_left_wheel_pose[0],
                    ur5e_shoulder_pose[0], 
                    ur5e_upper_arm_pose[0], 
                    ur5e_forearm_pose[0], 
                    ur5e_wrist_1_pose[0], 
                    ur5e_wrist_2_pose[0], 
                    ur5e_wrist_3_pose[0]
                ])))
                # 로봇 기구학 계산
                sb_rot = r.as_matrix()
                T_sb = np.eye(4)
                T_sb[0,3] = x
                T_sb[1,3] = y
                T_sb[2,3] = z
                T_sb[:3, :3] = sb_rot
                
                T_b0 = np.eye(4)
                T_b0[0,3] = 0.1315
                T_b0[2,3] = 0.51921
                
                try:
                    T_0e = ur5e_robot.fkine(q[2:]).A
                    T = T_b0 @ T_0e
                    H_current = SE3(T)
                    T_cur = T_sb @ T
                    cur_p = T_cur[:3, 3]
                    T_e = T_cur
                except Exception as e:
                    print("Forward kinematics error: " + str(e))
                    continue

                if T_robot is None:
                    T_robot = T
                    desired_sphere.set_world_pose(T_robot[:3, 3])
                if start_t is None:
                    start_t = world.current_time

                # 궤적 추적 로직
                if human_trajectory_index >= len(human_trajectory["x"]):
                    human_position = np.array([
                        human_trajectory["x"][-1],
                        human_trajectory["y"][-1],
                        human_trajectory["z"][-1]
                    ])
                else:
                    human_position = np.array([
                        human_trajectory["x"][human_trajectory_index],
                        human_trajectory["y"][human_trajectory_index],
                        human_trajectory["z"][human_trajectory_index]
                    ])
                    robot_target_position = human_position.copy()

                # 거리 기반 타겟 업데이트
                distance_to_target = np.linalg.norm(human_position - T_e[:3, 3])
                distance_threshold = 0.4

                if distance_to_target < distance_threshold:
                    human_sphere.set_world_pose(robot_target_position)
                    human_trajectory_index += 1

                # 제어 로직
                cur_ee = T_e[:3,3]
                sight_vec = T_e[:3,0]
                sight_vec /= np.linalg.norm(sight_vec)

                rotation_matrix =  calculate_natural_rotation_xy_only(T_cur, robot_target_position) #np.eye(3) #calculate_natural_rotation(T_cur, robot_target_position)

                T_sd = np.eye(4)
                T_sd[:3, :3] = rotation_matrix
                T_sd[0, 3] = robot_target_position[0]
                T_sd[1, 3] = robot_target_position[1]
                T_sd[2, 3] = robot_target_position[2]
                desired_sphere.set_world_pose(T_sd[:3, 3])
                # 케이블 시각화 점들 계산
                cable_world = []
                cable_points = sagging_curve_points(cur_ee, robot_target_position, num_points=10, sag=0.5)
                for i, point in enumerate(cable_points):
                    # 점을 월드 좌표계로 변환
                    point_homogeneous = np.append(point, 1)  # 동차 좌표로 확장
         
                    cable_world.append(point)
                
                    # 구 생성
                    sphere = VisualSphere(
                        prim_path=f"/World/Xform/point_sphere_{i}",
                        name=f"point_sphere_{i}",
                        position=point,
                        radius=0.02,
                        color=np.array([0.8, 0.8, 0.2])  # 노란색
                    )
                cable_world = np.array(cable_world)
                xform_pose = np.vstack((xform_pose, cable_world))


                scatter_x.append(T_e[0,3])
                scatter_y.append(T_e[1,3])
                scatter_mob_x.append(x)
                scatter_mob_y.append(y)

                # 각도 계산
                direction_unit_vector = human_position - T_cur[:3, 3]
                direction_unit_vector = direction_unit_vector / np.linalg.norm(direction_unit_vector)
                cos_theta = np.dot(direction_unit_vector, sight_vec)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                theta_values.append(np.degrees(theta))

                T_bd = np.linalg.inv(T_sb) @ T_sd
                try:
                    H_desired = SE3(T_bd)
                except ValueError:
                    T_bd_normalized = trnorm(T_bd)
                    H_desired = SE3(T_bd_normalized)

                # 자코비안 계산
                try:
                    F = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
                    J_p = base.tr2adjoint(T.T) @ F
                    J_a_e = base.tr2adjoint(T.T) @ ur5e_robot.jacob0(q[2:])
                    J_mb = np.hstack((J_p, J_a_e))
                    J_mb_v = J_mb[:3, :]  # 3x8 자코비안 (선형 속도)
                    J_mb_w = J_mb[3:, :]
                except Exception as e:
                    print("Jacobian calculation error: " + str(e))
                    continue

                T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4
                er_sum = 0
                for i in range(3):
                    er = H_current.A[i,3]-H_desired.A[i,3]
                    er_sum += er**2
                et= np.sqrt(er_sum)

                # QP 설정
                Q = np.eye(n_dof + 6)
                # Joint velocity component of Q
                Q[:2, :2] *= 1.0 / (et * 100)


                Q[n_dof :, n_dof :] = (1. / et) * np.eye(6)
                # Q의 고유값 확인 및 수정
                Q = (Q + Q.T) / 2  # 대칭화
                eig_min = np.min(np.linalg.eigvals(Q))
                if eig_min < 0:
                    Q += np.eye(Q.shape[0]) * (-eig_min + 1e-6)
                
                H = np.zeros((n_dof-2, 6, n_dof-2))  # same as jacobm

                for j in range(n_dof-2):
                    for i in range(j, n_dof-2):
                        H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                        H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                        if i != j:
                                H[i, :3, j] = H[j, :3, i]
                                H[i, 3:, j] = H[j, 3:, i]

                # manipulability only for arm joints
                J_a = ur5e_robot.jacob0(q[2:])
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
                J_m = np.zeros((n_dof-2,1))
                for i in range(n_dof-2):
                    c = J_a @ np.transpose(H[i, :, :])  # shape: (6,6)
                    J_m[i,0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

                A = np.zeros((n_dof + 2 + 20 + 10 , n_dof + 6))
                B = np.zeros(n_dof + 2 + 20 + 10)

                J_dj = np.zeros(n_dof+6)
                w_p_sum = 0.0
                min_dist_list = []  # 장애물과의 최소 거리 리스트
                g_vec_list = []  # 각 점에서의 장애물 방향 벡터 리스트
                # 초기화
                weighted_g_vec = np.zeros(3)  # 가중치가 적용된 방향 벡터의 합
                total_weight = 0.0  # 총 가중치
                for i , pose in enumerate(xform_pose) :

                    distance, index, g_vec_ = get_nearest_obstacle_distance(pose, obstacles_positions[:, :3], obstacle_radius, T_cur)
                    min_dist_ = np.min(distance)
                    min_dist_list.append(min_dist_)  # 최소 거리 추가
                    g_vec_list.append(g_vec_) 

                min_dist = np.min(min_dist_list)
                min_index = np.argmin(min_dist_list)
                # 거리가 0.3m 이하인 모든 장애물에 대해 g_vec 평균 계산
                threshold_dist = 0.5
                close_indices = [i for i, d in enumerate(min_dist_list) if d <= threshold_dist]
                
                if len(close_indices) > 0:
                    # 근처 장애물들의 g_vec을 모두 더함
                    g_vec_sum = np.zeros(3)
                    for idx in close_indices:
                        g_vec_sum += g_vec_list[idx]
                    # 평균 계산
                    avg_g_vec = g_vec_sum / len(close_indices)
                    avg_g_vec /= np.linalg.norm(avg_g_vec)  # 정규화
                    g_vec = avg_g_vec
                    # self.get_logger().debug(f"Using avg g_vec from {len(close_indices)} close obstacles")
                else:
                    # 근처 장애물이 없으면 최소 거리 장애물 사용
                    g_vec = g_vec_list[min_index]
                    # self.get_logger().debug(f"No obstacles within {threshold_dist}m, using closest")

                avoid_vec = (g_vec) #+ d_vec_unit #+ perpendicular_weight * perpendicular_vec# 장애물 방향과 목표 방향의 합
                # avoid_vec = (g_vec) + perpendicular_vec + d_vec_unit # 장애물 방향과 목표 방향의 합

                avoid_vec /= np.linalg.norm(avoid_vec)  # 정규화

                # print('min_dist overall:', min_dist)
                pose_closest = xform_pose[min_index]
                
                
                # avoid 시작점과 끝점 정의
                start_point_a = Float3(cur_ee)  # Float3 형식으로 변환
                # print('g_vec', g_vec)
                end_point_a = Float3(
                    start_point_a[0] +  avoid_vec[0],
                    start_point_a[1] +  avoid_vec[1],
                    start_point_a[2] +  avoid_vec[2]
                )  # Float3 형식으로 변환
                # 색상과 크기 정의
                color_a = [ColorRgba(0.0, 0.5, 0.5, 1.0)]  # 빨간색 (RGBA 형식)
                sizes_a = [0.4]  # 크기 리스트
                draw.clear_lines()
                # g_vec 벡터 그리기
                draw.draw_lines(
                    [start_point_a],  # 시작점 리스트
                    [end_point_a],  # 끝점 리스트
                    color_a,  # 색상 리스트
                    sizes_a  # 크기 리스트
                )

                # g_vec의 시작점과 끝점 정의
                start_point = Float3(pose_closest)  # Float3 형식으로 변환
                # print('g_vec', g_vec)
                end_point = Float3(
                    start_point[0] -  g_vec[0],
                    start_point[1] -  g_vec[1],
                    start_point[2] -  g_vec[2]
                )  # Float3 형식으로 변환
                # 색상과 크기 정의
                color = [ColorRgba(1.0, 0.0, 0.0, 1.0)]  # 빨간색 (RGBA 형식)
                sizes = [0.4]  # 크기 리스트

                # g_vec 벡터 그리기
                draw.draw_lines(
                    [start_point],  # 시작점 리스트
                    [end_point],  # 끝점 리스트
                    color,  # 색상 리스트
                    sizes  # 크기 리스트
                )
                
                start_point_1 = Float3(cur_ee)  # Float3 형식으로 변환
                # print('g_vec', g_vec)
                end_point_1 = Float3(
                    start_point_1[0] +  sight_vec[0],
                    start_point_1[1] +  sight_vec[1],
                    start_point_1[2] +  sight_vec[2]
                )  # Float3 형식으로 변환
                # 색상과 크기 정의
                color = [ColorRgba(0.0, 1.0, 0.0, 1.0)]  # 빨간색 (RGBA 형식)
                sizes = [0.4]  # 크기 리스트
                
                # g_vec 벡터 그리기
                draw.draw_lines(
                    [start_point_1],  # 시작점 리스트
                    [end_point_1],  # 끝점 리스트
                    color,  # 색상 리스트
                    sizes  # 크기 리스트
                )
                
                for i , pose in enumerate(xform_pose) :    
                    if i < 24:  # mobile base wheels

                        d_dot = (-avoid_vec) @ J_mb_v   # 장애물 피하는 방향으로의 자코비안
                        
                        A[i, :8] = d_dot 
                        A[i, 8:] = np.zeros((1, 6)) 
                        B[i] = (min_dist_list[i] - d_safe) / ((d_influence - d_safe))
                        # B[i] = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        w_p = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        # print("min_dist_list[i]:", min_dist_list[i])
                        # print(f"w_p: {w_p}")
                        # w_p = (min_dist_list[i]-d_influence)/(d_influence-d_safe) 
                        J_dj[:8] +=  (d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                        # print("J_dj[:8]:", J_dj[:8])
                        # print(f"robot {i}th link Distance: {min_dist:.2f} m")
                        # print(f"{i}th J_dj : {A[i, :8] * w_p}")
                        # print(f"{i}th w_p : { w_p}")
                        w_p_sum += np.abs(w_p)
                        # if min_dist < 0.0:
                            # print(f"robot {i+1}th link is too close to an obstacle. Distance: {min_dist:.2f} m")
                            # print(f"A : {A[i, :8]}")
                            # print(f"B : {B[i]:.2f}")
                    elif 23 < i < 30:  # UR5e joints
                        
                        J_mb_arm_v = np.hstack([np.zeros((3, i - 22)), J_a_e[:3, i - 24:]])

                        d_dot = (-avoid_vec) @ J_mb_arm_v

                        A[i, :8] = d_dot
                        A[i, 8:] = np.zeros((1, 6)) 
                        B[i] = (min_dist_list[i] - d_safe) / ((d_influence - d_safe))
                        # B[i] = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        w_p = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        # print("min_dist_list[i]:", min_dist_list[i])
                        # print(f"w_p: {w_p}")
                        # w_p = (min_dist_list[i]-d_influence)/(d_influence-d_safe) 
                        J_dj[:8] += (d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                        # print("J_dj[:8]:", J_dj[:8])
                        # print(f"robot {i}th link Distance: {min_dist:.2f} m")
                        # print(f"{i}th J_dj : {A[i, :8] * w_p}")
                        # print(f"{i}th w_p : { w_p}")
                        w_p_sum += np.abs(w_p)
                    else: 
                        J_mb_arm_v = np.hstack([np.zeros((3, 7)), J_a_e[:3, 5:]])

                        d_dot = (-avoid_vec) @ J_mb_arm_v

                        A[i, :8] = d_dot
                        A[i, 8:] = np.zeros((1, 6)) 
                        B[i] = (min_dist_list[i] - d_safe) / ((d_influence - d_safe))
                        # B[i] = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        w_p = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                        # print("min_dist_list[i]:", min_dist_list[i])
                        # print(f"w_p: {w_p}")
                        # w_p = (min_dist_list[i]-d_influence)/(d_influence-d_safe) 
                        J_dj[:8] += (d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                        # print("J_dj[:8]:", J_dj[:8])
                        # print(f"robot {i}th link Distance: {min_dist:.2f} m")
                        # print(f"{i}th J_dj : {A[i, :8] * w_p}")
                        # print(f"{i}th w_p : { w_p}")
                        w_p_sum += np.abs(w_p)

                # 비용 함수 항들 계산
                C1 = np.concatenate((np.zeros(2), -J_m.reshape((n_dof - 2,)), np.zeros(6)))
                
                try:
                    bTe = ur5e_robot.fkine(q[2:], include_base=False).A
                    θε = atan2(bTe[1, -1], bTe[0, -1])
                    C2 = np.zeros(n_dof + 6)
                    C2[0] = -5. * θε
                except:
                    C2 = np.zeros(n_dof + 6)

                lambda_max = 10.
                min_distance = np.min(min_dist_list)  # 장애물과의 최소 거리
                if min_distance <= d_influence :
                    lambda_c = (lambda_max /(d_influence - d_safe)**2) * (min_distance - d_influence)**2
                else:
                    lambda_c = 0.0
                
                J_c = lambda_c * J_dj/w_p_sum
                # 장애물 회피 (간단화)
                C3 = J_c

                # 회전 제어 항
                J_h = np.zeros(n_dof+6)
                try:
                    J_mb_w_h = direction_unit_vector @ J_mb_w
                    epsilon = 1e-6
                    lambda_h = lambda_h_a_param * max(abs(theta), epsilon)
                    J_h[:8] = lambda_h * J_mb_w_h
                except:
                    pass
                C4 = J_h

                # 현재 파라미터로 비용 함수 계산
                # w3 = 1.0
                C = w1 * C1 + w2 * C2 + w3 * C3 + w4 * C4

                J_ = np.c_[J_mb, np.eye(6)]

                # 오차 계산
                eTep = np.linalg.inv(T) @ H_desired.A
                e = np.zeros(6)
                e[:3] = eTep[:3, -1]
                try:
                    e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
                except:
                    e[3:] = np.array([0, 0, 0])

                k = np.eye(6)
                k[:3,:] *= 2.0
                v = k @ e

                lb = -np.r_[qdlim[:n_dof], 10 * np.ones(6)]
                ub = np.r_[qdlim[:n_dof], 10 * np.ones(6)]

                # QP 해결
                try:
                    x_ = cp.Variable(n_dof+6)
                    objective = cp.Minimize(0.5 * cp.quad_form(x_, Q) + C.T @ x_)
                    constraints = [
                        x_ >= lb,
                        x_ <= ub,
                        A @ x_ <= B,
                        J_ @ x_ == v,
                    ]

                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.ECOS, verbose=False)

                    if x_.value is not None:
                        qd = x_.value
                    else:
                        print("QP solver failed to find a solution.")
                        qd = np.zeros(n_dof+6)
                except Exception as e:
                    print("QP solver error: " + str(e))
                    qd = np.zeros(n_dof+6)

                qd = qd[:n_dof]
                qd = qd[:8]

                # 오차 값 저장
                current_time = world.current_time
                et_x = abs(eTep[0, -1])
                et_y = abs(eTep[1, -1])
                et_z = abs(eTep[2, -1])
                et = et_x + et_y + et_z

                et_values.append(et)
                et_x_values.append(et_x)
                et_y_values.append(et_y)
                et_z_values.append(et_z)
                time_values.append(current_time)

                # 종료 조건
                if human_trajectory_index >= traj_num_points:
                    print("Trajectory completed!")
                    break

                # 로봇 제어
                try:
                    wc, vc = qd[0], qd[1]
                    r_m = 0.165
                    l_m = 0.582
                    w_R = vc/r_m + l_m*wc/(2*r_m)
                    w_L = vc/r_m - l_m*wc/(2*r_m)
                    joint_velocities = np.zeros(16)
                    joint_velocities[0] = -w_L
                    joint_velocities[2] = -w_L
                    joint_velocities[1] = w_R
                    joint_velocities[3] = w_R
                    joint_velocities[4:10] = qd[2:]

                    actions = ArticulationActions(
                        joint_velocities=joint_velocities,
                        joint_indices=aljnu_joint_indices
                    )
                    my_robot.apply_action(actions)
                except Exception as e:
                    print("Robot control error: " + str(e))

    print("Simulation loop completed")

    # 결과 그래프 저장
    if human_trajectory_index >= traj_num_points or et_x+et_y+et_z < 0.3 or iteration_count >= max_iterations:

        print("Reached desired position!")
        plt.scatter(scatter_x, scatter_y, color='orange', s=10, label="End-Effector Trajectory")
        plt.scatter(T_sd[0,3], T_sd[1,3], color='blue', label="Robot Desired Position", s=100)
        plt.scatter(scatter_mob_x, scatter_mob_y, color='yellow', s=10, label="Robot-Base Trajectory")
        # plt.scatter(human_trajectory["x"], human_trajectory["y"], color='green', s=10, label="Human Trajectory")
        # plt.scatter(obstacles_positions[:,0], obstacles_positions[:,1], color='red', s=100, label="Obstacles")
        for obstacle in obstacles_positions:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle_radius, color='red', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")    
        plt.title("Simulation Environment with Obstacle Avoidance")
        
        # 그래프 저장
        fname = f"simulation_obstaclewall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = os.path.join(RESULTS_FOLDER, fname)
        plt.savefig(out_path)
        print(f"Graph saved: {out_path}")
        plt.close()
        
        
        # theta_values의 평균, 최대값, 최소값 계산
        if theta_values:
            theta_mean = np.mean(theta_values)  # 평균
            theta_max = np.max(theta_values)   # 최대값
            theta_min = np.min(theta_values)   # 최소값

            print(f"Theta Mean: {theta_mean:.2f} degrees")
            print(f"Theta Max: {theta_max:.2f} degrees")
            print(f"Theta Min: {theta_min:.2f} degrees")
        else:
            print("Theta values are empty. No data to calculate.")
        # 두 번째 그래프 저장 (Theta Over Time)
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, theta_values, label="Theta (degrees)", color="purple")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Theta (degrees)")
        plt.title("Theta Over Time")
        plt.legend()
        plt.grid()
        out_path = os.path.join(RESULTS_FOLDER, f"theta_{obstacles_positions[0][0]:.2f}_{obstacles_positions[0][1]:.2f}_w3_{w3:.1f}_over_time_graph.png")
        plt.savefig(out_path)
        print(f"Graph saved: {out_path}")
        plt.close()  # 그래프를 닫아 다음 그래프와 겹치지 않도록 함
        simulation_app.close()
        print("Simulation completed.")

    if len(scatter_x) > 0:
        filename = os.path.join(RESULTS_FOLDER, "str_simulation_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".png")
        
        # Figure 설정을 더 명확하게 지정
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # 축 범위를 먼저 설정
        ax.set_xlim(-1.0, 6.0)
        ax.set_ylim(-1.0, 4.0)
        
        # 제목과 라벨 설정
        ax.set_title("Linear Trajectory (w_m=" + str(w1) + ", w_o=" + str(w2) + ", w_h=" + str(w4) + ", λ=" + str(lambda_h_a_param) + ")")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        
        # 데이터 플롯
        ax.plot(human_trajectory["x"], human_trajectory["y"], 'b--', linewidth=2, label="Target Trajectory (Linear)")
       
        # 범례와 그리드
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 종횡비를 고정하되 xlim, ylim 유지
        ax.set_aspect('equal', adjustable='box')
        for obstacle in obstacles_positions:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle_radius, color='red', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.legend()
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장 (bbox_inches='tight' 제거하여 크기 고정)
        plt.savefig(filename, dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)  # 명시적으로 figure 닫기
        
        print("Graph saved as '" + filename + "'")
    
    # 최종 오차를 파일로 저장
    error_file = os.path.join(RESULTS_FOLDER, "error_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".txt")
    final_error = et_values[-1] if et_values else 1.0  # inf 대신 1.0으로 설정
    
    # theta_values 평균 계산
    avg_theta = np.mean(theta_values) if theta_values else 0.0
    
    # 오차와 평균 theta 값을 함께 저장
    with open(error_file, 'w') as f:
        f.write(str(final_error) + "\n")
        f.write(str(avg_theta))
    
    # theta 평균을 별도 파일로도 저장
    theta_file = os.path.join(RESULTS_FOLDER, "theta_avg_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".txt")
    with open(theta_file, 'w') as f:
        f.write(str(avg_theta))
    
    print("Final error: " + str(final_error))
    print("Average theta: " + str(avg_theta) + " degrees")
    
except Exception as e:
    print("Error in simulation: " + str(e))
    import traceback
    traceback.print_exc()
    
    # 오류 발생시에도 파일 생성
    error_file = os.path.join(RESULTS_FOLDER, "error_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".txt")
    theta_file = os.path.join(RESULTS_FOLDER, "theta_avg_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".txt")
    
    with open(error_file, 'w') as f:
        f.write("1.0\n")  # inf 대신 1.0으로 설정
        f.write("0.0")    # 오류시 theta 평균은 0.0
    
    with open(theta_file, 'w') as f:
        f.write("0.0")    # 오류시 theta 평균은 0.0

finally:
    simulation_app.close()
    print("Simulation completed.")