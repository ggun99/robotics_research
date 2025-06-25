import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.stage import update_stage
from isaacsim.core.api import World, SimulationContext
from isaacsim.core.api.objects import DynamicCylinder, VisualSphere
# from isaacsim.core.api.materials import PhysicsMaterial
# from isaacsim.cortex.framework.robot import CortexUr10
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.core.utils.prims import create_prim
from pxr import UsdGeom, Gf
import isaacsim.core.utils.prims as prim_utils

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import random
from RRTPlanner import RealTime3DTrajectoryPlanner

def joint_velocity_damper(
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
            if q[i] - qlim[0, i] <= pi:
                Bin[i] = -gain * (((qlim[0, i] - q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if qlim[1, i] - q[i] <= pi:
                Bin[i] = gain * ((qlim[1, i] - q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin

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
    # 월드 기준 좌표계에서의 distance, 방향벡터

    for obs in obstacles:
        obs[2] = position[2]
        distances = ((np.linalg.norm(position - obs)) - obstacle_radius)
    index = np.argmin(distances)

    g_vec = (position - obstacles[index])
    g_vec /= np.linalg.norm(g_vec)  # 방향 벡터 정규화

    return distances, index, g_vec

def generate_points_between_positions(start_pos, end_pos, num_points=10, T_e = None):
    """
    두 3차원 위치를 이어주는 선에서 일정한 간격으로 점을 생성하는 함수.

    Args:
        start_pos (np.ndarray): 시작 위치 (3차원 좌표).
        end_pos (np.ndarray): 끝 위치 (3차원 좌표).
        num_points (int): 생성할 점의 개수 (기본값: 10).

    Returns:
        np.ndarray: 생성된 점들의 좌표 배열 (shape: num_points x 3).
    """
    # position을 엔드 이펙터 좌표계로 변환
    # start_pos_homogeneous = np.append(start_pos, 1)  # 동차 좌표로 확장
    # start_pos_local = np.linalg.inv(T_e) @ start_pos_homogeneous
    # start_pos_local = start_pos_local[:3]  # 3차원으로 변환
    start_pos_local = np.zeros(3)   # 엔드 이펙터 좌표계에서 시작 위치
    # position을 엔드 이펙터 좌표계로 변환
    end_pos_homogeneous = np.append(end_pos, 1)  # 동차 좌표로 확장
    end_pos_local = np.linalg.inv(T_e) @ end_pos_homogeneous
    end_pos_local = end_pos_local[:3]  # 3차원으로 변환
    # print('start_pos_local: ', start_pos_local)
    # print('end_pos_local: ', end_pos_local)
    # 시작 위치와 끝 위치를 연결하는 선을 따라 일정한 간격으로 점 생성
    points = np.linspace(start_pos_local, end_pos_local, num_points)
    dist_vec = (end_pos - start_pos)/num_points
    # print('dist_vec: ', dist_vec.shape)
    # dist_vec_local = np.linalg.inv(T)[:3,:3] @ dist_vec
    return points, dist_vec

# et 값과 시간을 저장할 리스트
et_values = []
et_x_values = []
et_y_values = []
et_z_values = []
time_values = []


# Matplotlib 인터랙티브 모드 활성화
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

# 초기 그래프 설정
line_total, = ax.plot([], [], label="Total Error (et)", color="blue")
line_x, = ax.plot([], [], label="Error in X (et_x)", color="red")
line_y, = ax.plot([], [], label="Error in Y (et_y)", color="green")
line_z, = ax.plot([], [], label="Error in Z (et_z)", color="orange")
ax.axhline(y=0.02, color='purple', linestyle='--', label="Threshold (0.02)")
ax.set_xlim(0, 100)  # 초기 x축 범위
ax.set_ylim(0, 1)    # 초기 y축 범위
ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Error(m)")
ax.set_title("Error Reduction Over Time")
ax.legend()
ax.grid()

# # Distant Light 생성
# create_prim(
#     prim_path="/World/MyDistantLight",
#     prim_type="DistantLight",
#     position=[0.0, 0.0, 5.0],  # 위치는 사실 distant light에선 큰 영향 없음   
# )

step_count = 0

ur5e_robot = rtb.models.UR5()
world = World(stage_units_in_meters=1.0)
scene = world.scene
assets_root_path = get_assets_root_path()

# Gf.Quatd 객체를 numpy 배열로 변환
orientation_quat = Gf.Quatd(0.7071068, 0, 0.7071068, 0)  # 기존 쿼터니언
orientation_np = np.array([orientation_quat.GetReal(), *orientation_quat.GetImaginary()])  # numpy 배열로 변환

# create_prim 호출
distant_light_prim = prim_utils.create_prim(
    "/World/DistantLight",
    "DistantLight",
    position=Gf.Vec3d(0, 0, 10),  # 위치
    orientation=orientation_np,  # numpy 배열로 변환된 쿼터니언
    attributes={
        "inputs:intensity": 2000,
        "inputs:color": (1.0, 1.0, 1.0),  # 흰색 빛
    }
)
# use Isaac Sim provided asset
robot_asset_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"#assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
prim_path = "/World/aljnu_mp"
# 참조 추가 전에 순환 참조 확인
if not prim_path.endswith("mobile_manipulator"):
    add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)
# 1. Stage에 USD 추가
# add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)
aljnu_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # aljnu_mp의 조인트 인덱스
aljnu_body_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # aljnu_mp의 바디 인덱스
n_dof = 8 # base(2) + arm(6)
k_e = 0.5
rho_i = 0.9 # influence distance
rho_s = 0.1  # safety factor
eta = 1
qdlim = np.array([1.5]*8)
qdlim[:1] = 1.5  # 베이스 조인트 속도 제한
qdlim[1] = 1.0
qlim = np.array([[-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ np.inf, np.inf, 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])

# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'

world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()


# 파라미터 설정
map_size = 6
num_obstacles = 2
min_distance_between_obstacles = 1.5
obstacle_radius_range = (0.25, 0.25)

planner = RealTime3DTrajectoryPlanner(
    map_size=map_size,
    num_obstacles=num_obstacles,
    min_distance_between_obstacles=min_distance_between_obstacles,
    obstacle_radius_range=obstacle_radius_range,
    step_size=0.1,  # RRT step size
    goal_sample_rate=0.1,  # 목표 샘플링 확률
    max_iter=1000,  # 최대 반복 횟수
    goal_clearance=0.6,  # 목표 지점에서의 여유 공간
    safety_margin=0.2,  # 안전 여유 공간
    z_range=(0.97, 0.97)  # z축 범위 설정 (고정 높이, 예: 0.97m)
)
# 사람의 경로 생성
human_start_position = np.array([0.5922, 0.1332, 0.97])  # 사람의 시작 위치
human_goal_position = np.array([1.5, 1.0, 0.97])  # 사람의 목표 위치

# 시뮬레이션 컨텍스트 초기화
simulation_context = SimulationContext()
obstacles_positions = planner.generate_obstacles(goal=human_goal_position)
# print(f"Generated obstacles: {obstacles_positions}")
# print(obstacles_positions.shape)  # (num_obstacles, 4)
# print(obstacles_positions[:, :3])

# RRTPlanner를 사용하여 사람의 경로 생성
human_path = planner.plan(start=human_start_position, goal=human_goal_position)
if human_path is None:
    print("사람의 경로를 찾을 수 없습니다.")
    simulation_app.close()
    exit()
# 사람의 경로를 부드럽게 만들기
human_smoothed_path = planner.smooth_path(human_path, num_points=300)

# 사람의 속도 프로파일 생성
human_velocity_profile = planner.generate_velocity_profile(human_smoothed_path, v_max=0.2)

# 사람의 전체 경로 생성
human_trajectory = planner.generate_full_trajectory(human_smoothed_path, human_velocity_profile)

# 시뮬레이션 루프에서 사람의 움직임 따라가기
human_trajectory_index = 0

# 원기둥 생성
obstacle_radius = 0.25
obstacle_height = 2.0

cylinder = DynamicCylinder(
    prim_path="/World/Xform/Cylinder1",
    name="cylinder1",
    position=obstacles_positions[0][:3],
    radius=obstacle_radius,
    height=obstacle_height,
    color=np.array([0.8, 0.2, 0.2])
)
cylinder = DynamicCylinder(
    prim_path="/World/Xform/Cylinder2",
    name="cylinder2",
    position=obstacles_positions[1][:3],
    radius=obstacle_radius,
    height=obstacle_height,
    color=np.array([0.8, 0.2, 0.2])
)
# cylinder = DynamicCylinder(
#     prim_path="/World/Xform/Cylinder3",
#     name="cylinder3",
#     position=obstacles_positions[2][:3],
#     radius=obstacle_radius,
#     height=obstacle_height,
#     color=np.array([0.8, 0.2, 0.2])
# )
# desired position with sphere
desired_sphere = VisualSphere(
    prim_path="/World/Xform/sphere",
    name="desired_sphere",
    position=human_goal_position,
    radius=0.02,
    color=np.array([0.2, 0.8, 0.2])
)
# human position with sphere
human_position = np.array([0.5922, 0.1332, 0.97])
human_sphere = VisualSphere(
    prim_path="/World/Xform/human_sphere",
    name="human_sphere",
    position=human_position,
    radius=0.02,
    color=np.array([0.2, 0.2, 0.8])
)
# 2. Articulation 객체로 래핑
my_robot = Articulation(prim_path)
my_robot.initialize()

aljnu_indices = aljnu_body_indices[5:]   
values = np.ones((1, len(aljnu_indices)), dtype=bool)  
my_robot.set_body_disable_gravity(values, indices=[0], body_indices=aljnu_indices) 

print("aljnu_mp_robot is added")

joints_default_positions = my_robot.get_joint_positions()
# 1. 초기 자세로 이동할 목표값
target_positions = np.copy(joints_default_positions)
target_positions[0][5] = -np.pi/2
target_positions[0][6] = np.pi/2
target_positions[0][8] = np.pi/2
target_joint_positions = target_positions[0][aljnu_joint_indices]

# 2. 상태 플래그
reached_default = False
position_tolerance = 0.1  # 허용 오차
mobile_base_pose, mobile_base_quat = my_robot.get_world_poses(indices = [0])
x0 = mobile_base_pose[0][0]
y0 = mobile_base_pose[0][1]
H_desired = None

# collision avoidance parameters
d_safe = 0.1
d_influence = 0.6

# moving human
moving_t = 30.0  # 이동 시간
start_t = None  # 현재 시간
human_desired_position = np.array([-1.5, -1.0, 0.97])  # 목표 위치
T_robot = None  # 로봇의 현재 위치를 저장할 변수

# 마지막 업데이트 시간을 저장할 변수
last_update_time = None
update_interval = 0.1  # 업데이트 간격 (초)

while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        
        if world.current_time_step_index == 0:  
            world.reset()
            reached_default = False  # 시뮬 초기화 시 플래그도 초기화
        current_joint_positions = my_robot.get_joint_positions()[0][aljnu_joint_indices]
        mobile_base_pose, mobile_base_quat = my_robot.get_world_poses(indices = [0])
        x = mobile_base_pose[0][0]
        y = mobile_base_pose[0][1] 
        z = mobile_base_pose[0][2] 

        quat = mobile_base_quat[0]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('zyx', degrees=False)  # 'zyx' 순서로 euler angles 추출

        q = np.zeros(8)
        q[0] = 0.0
        q[1] = 0.0 
        q[2:] = current_joint_positions[4:10]  # UR5e 조인트 위치
        
        if  not reached_default:
            # 1단계: default position으로 이동
            actions = ArticulationActions(
                joint_positions=target_joint_positions,
                joint_indices=aljnu_joint_indices
            )
            my_robot.apply_action(actions)

            # 목표 자세 도달 여부 체크
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

            xform_pose = np.array([rear_right_wheel_pose[0], rear_left_wheel_pose[0], front_right_wheel_pose[0], front_left_wheel_pose[0],
                                ur5e_shoulder_pose[0], ur5e_upper_arm_pose[0], ur5e_forearm_pose[0], ur5e_wrist_1_pose[0], 
                                ur5e_wrist_2_pose[0], ur5e_wrist_3_pose[0]])
            # print("xform_pose: ", xform_pose)  # (10, 3)
            # print("xform_pose: ", xform_pose.shape) (10,3)
            # ['base_link', 'front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link', 
            # 'ur5e_shoulder_link', 'ur5e_upper_arm_link', 'ur5e_forearm_link', 'ur5e_wrist_1_link', 'ur5e_wrist_2_link', 'ur5e_wrist_3_link', 

            sb_rot = r.as_matrix()
            T_sb = np.eye(4)
            T_sb[0,3] = x
            T_sb[1,3] = y
            T_sb[2,3] = z
            T_sb[:3, :3] = sb_rot  # 베이스 프레임 회전 행렬
            T_b0 = np.eye(4)
            T_b0[0,3] = 0.1315 # 0.1015
            # T_b0[1,3] = 
            T_b0[2,3] = 0.51921  # 0.47921
            T_0e = ur5e_robot.fkine(q[2:]).A

            T = T_b0 @ T_0e  # 베이스 프레임 기준 end-effector 위치
            H_current = SE3(T)  # 현재 end-effector 위치

            if start_t is None:
                start_t = world.current_time
            # 사람이 경로를 따라 이동
            if human_trajectory_index < len(human_trajectory["time"]):
                human_position = np.array([
                    human_trajectory["x"][human_trajectory_index],
                    human_trajectory["y"][human_trajectory_index],
                    human_trajectory["z"][human_trajectory_index]
                ])
                # print(f"Human position: {human_position}, Index: {human_trajectory_index}")
            
            
            # 업데이트 간격 확인
            current_time = world.current_time
            if last_update_time is None or (current_time - last_update_time >= update_interval):
                human_sphere.set_world_pose(human_position)  # 사람의 위치 업데이트
                human_trajectory_index += 1 # 인간 위치 업데이트
                # 로봇의 목표 위치를 사람의 현재 위치로 설정

                # 로봇이 사람을 따라가기
                T_cur = T_sb @ T  # 현재 로봇 위치 (월드 좌표계 기준)
                cur_p = T_cur[:3, 3]  # 현재 엔드 이펙터 위치 (월드 좌표계 기준)

                # 엔드 이펙터의 변환 행렬
                T_e = T_cur  # 월드 좌표계에서 엔드 이펙터 좌표계로의 변환

                # human_position을 엔드 이펙터 좌표계로 변환
                human_position_homogeneous = np.append(human_position, 1)  # 동차 좌표로 확장
                human_position_local = np.linalg.inv(T_e) @ human_position_homogeneous
                human_position_local = human_position_local[:3]  # 3차원으로 변환

                # 현재 엔드 이펙터 위치를 엔드 이펙터 좌표계로 변환 (항상 원점)

                # 목표 방향 계산 (엔드 이펙터 좌표계 기준)
                direction_vector = human_position_local # - cur_p_local
                direction_vector /= np.linalg.norm(direction_vector)  # 방향 벡터 정규화

                # # z축은 항상 위쪽을 향한다고 가정 (엔드 이펙터 좌표계 기준)
                # x_axis = np.array([1, 0, 0])

                # z_axis = direction_vector

                # y_axis = np.cross(x_axis, z_axis)
                # y_axis /= np.linalg.norm(y_axis)

                # x_axis = np.cross(z_axis, y_axis)

                # # 회전 행렬 생성
                # rotation_matrix = np.vstack([z_axis, y_axis, x_axis]).T

                # 로봇의 현재 x축 방향 (엔드 이펙터의 x축)
                current_x_axis = T_e[:3, 0]  # 엔드 이펙터 변환 행렬의 첫 번째 열

                # z축은 목표 방향 (human_position - cur_p)
                z_axis = direction_vector

                # y축은 현재 x축 방향과 z축의 외적
                y_axis = np.cross(current_x_axis, z_axis)
                y_axis /= np.linalg.norm(y_axis)  # 정규화

                # x축은 y축과 z축의 외적
                x_axis = np.cross(z_axis, y_axis)
                x_axis /= np.linalg.norm(x_axis)  # 정규화

                # 회전 행렬 생성
                rotation_matrix = np.vstack([z_axis, y_axis, x_axis]).T

                ee_matrix = np.eye(4)
                ee_matrix[:3, :3] = rotation_matrix  # 회전 행렬 설정

                T_ee = T_sb @ T @ ee_matrix
                
                # 로봇의 목표 위치 설정
                T_sd = np.eye(4)
                T_sd[:3, :3] = T_ee[:3,:3] #rotation_matrix # T_er[:3, :3]  # 회전 행렬은 단위 행렬로 설정

                T_sd[0, 3] = human_position[0] #robot_target_position[0]
                T_sd[1, 3] = human_position[1] #robot_target_position[1]
                T_sd[2, 3] = human_position[2] #robot_target_position[2]

                # 마지막 업데이트 시간 기록
                last_update_time = current_time
       
            num_points=10
            points_between, dist_vec = generate_points_between_positions(cur_p, human_position, num_points, T_e)
            # points_between에 있는 점들을 월드 좌표계로 변환하고 구를 생성
            points_world = []
            for i, point in enumerate(points_between):
                # 점을 월드 좌표계로 변환
                point_homogeneous = np.append(point, 1)  # 동차 좌표로 확장
                point_world = T_e @ point_homogeneous  # 엔드 이펙터 좌표계에서 월드 좌표계로 변환
                point_world = point_world[:3]  # 3차원으로 변환
                points_world.append(point_world)
            
                # 구 생성
                sphere = VisualSphere(
                    prim_path=f"/World/Xform/point_sphere_{i}",
                    name=f"point_sphere_{i}",
                    position=point_world,
                    radius=0.02,
                    color=np.array([0.8, 0.8, 0.2])  # 노란색
                )
            points_world = np.array(points_world)  # (num_points, 3) 형태로 변환
            xform_pose = np.vstack((xform_pose, points_world))  # 현재 xform_pose에 점 추가

            T_bd = np.linalg.inv(T_sb) @ T_sd  

            H_desired = SE3(T_bd)  # 목표 end-effector 위치

            F = np.array([[0.0, 1.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0], 
                            [0.0, 0.0],
                            [1.0, 0.0]])
            
            J_p = base.tr2adjoint(T.T) @ F  # 6x2 자코비안 (선형 속도)
            J_a_e = base.tr2adjoint(T.T) @ ur5e_robot.jacob0(q[2:])
            J_mb = np.hstack((J_p, J_a_e))  # 6x8 자코비안 (선형 속도 + 각속도)
            J_mb_v = J_mb[:3, :]  # 3x8 자코비안 (선형 속도)
            J_mb_w = J_mb[3:, :]  # 3x8 자코비안 (각속도)
            
            # print(my_robot.body_names)
            # ['base_link', 'front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link', 
            # 'ur5e_shoulder_link', 'ur5e_upper_arm_link', 'ur5e_forearm_link', 'ur5e_wrist_1_link', 'ur5e_wrist_2_link', 'ur5e_wrist_3_link', 
            # 'robotiq_85_left_inner_knuckle_link', 'robotiq_85_left_knuckle_link', 'robotiq_85_right_inner_knuckle_link', 'robotiq_85_right_knuckle_link', 'robotiq_85_left_finger_tip_link', 'robotiq_85_right_finger_tip_link']
            
            T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

            et = np.sum(np.abs(T_error[:3, -1])) 

            # Gain term (lambda) for control minimisation
            Y = 100

            # Quadratic component of objective function
            Q = np.eye(n_dof + 6)

            # Joint velocity component of Q
            Q[:2, :2] *= 1.0 / (et*100)

            # Slack component of Q
            Q[n_dof :, n_dof :] = (0.3 / et) * np.eye(6)

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

            JJ_inv = np.linalg.inv((J_a @ J_a.T))  #.reshape(-1, order='F')

            # Compute manipulability Jacobian only for arm joints
            J_m = np.zeros((n_dof-2,1))
            for i in range(n_dof-2):
                c = J_a @ np.transpose(H[i, :, :])  # shape: (6,6)
                J_m[i,0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

            A = np.zeros((n_dof + num_points + 6, n_dof + 6))
            B = np.zeros(n_dof + num_points + 6)
            
            J_dj = np.zeros(n_dof+6)
            w_p_sum = 0.0
            min_dist_list = []  # 장애물과의 최소 거리 리스트
            for i , pose in enumerate(xform_pose) :

                distance, index, g_vec = get_nearest_obstacle_distance(pose, obstacles_positions[:, :3], obstacle_radius, T_cur)
                min_dist = np.min(distance)
                min_dist_list.append(min_dist)  # 최소 거리 추가
                
                if i < 4:  # mobile base wheels
                
                    # mobile base 기준 계산
                    J_mb_v_ = F[:3,:]
                    g_vec = np.linalg.inv(T_sb[:3, :3]) @ g_vec # world 에서 mobile base 기준으로 변환
                    d_dot = g_vec @ J_mb_v_
                    
                    A[i, :2] = d_dot 
                    A[i, 2:] = np.zeros((1, 12)) 
                    B[i] = (min_dist - d_safe) / (d_influence - d_safe) 
                    w_p = (d_influence-min_dist)/(d_influence-d_safe) 
                    J_dj[:8] += A[i, :8] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p
                    if min_dist < 0.0:
                        print(f"robot {i+1}th link is too close to an obstacle. Distance: {min_dist:.2f} m")
                        print(f"A : {A[i, :8]}")
                        print(f"B : {B[i]:.2f}")
                elif 3 < i < 10:  # UR5e joints
                    
                    # mobile base 기준 계산
                    J_mb_ = np.hstack((F , ur5e_robot.jacob0(q[2:2 + i - 3],end = ur5e_robot.links[i - 2])))
                    J_mb_v_ = J_mb_[:3,:]
                    g_vec = np.linalg.inv(T_sb[:3, :3]) @ g_vec # world 에서 mobile base 기준으로 변환
                    d_dot = g_vec @ J_mb_v_

                    A[i, :i-1] = d_dot
                    A[i, i-1:] = np.zeros((1, 15 - i)) 
                    B[i] = (min_dist - d_safe) / (d_influence - d_safe)
                    w_p = (d_influence-min_dist)/(d_influence-d_safe) 
                    J_dj[:8] += A[i, :8] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p


                else:
                    # mobile base 기준 계산
                    J_mb_ = np.hstack((F , ur5e_robot.jacob0(q[2:])))
                    J_mb_v_ = J_mb_[:3,:]
                    g_vec = np.linalg.inv(T_sb[:3, :3]) @ g_vec # world 에서 mobile base 기준으로 변환
                    d_dot = g_vec @ J_mb_v_

                    # position_homogeneous = np.append(pose, 1)  # 동차 좌표로 확장
                    # position_local = np.linalg.inv(T_e) @ position_homogeneous
                    # position_local = position_local[:3]  # 3차원으로 변환
                    # dist_T = np.eye(4)
                    # dist_T[:3, 3] = position_local
                    # T_ = T @ dist_T
                    # J_p_ = base.tr2adjoint(T_.T) @ F  # 6x2 자코비안 (선형 속도)
                    # J_a_e_ = base.tr2adjoint(T_.T) @ ur5e_robot.jacob0(q[2:])
                    # J_mb_ = np.hstack((J_p_, J_a_e_))  # 6x8 자코비안 (선형 속도 + 각속도)
                    # J_mb_v_ = J_mb_[:3, :]  # 3x8 자코비안 (선형 속도)
                    # J_mb_arm_v_ = np.hstack([np.zeros((3, 7)), J_mb_v_[:3, 7: ]])

                    # d_dot = g_vec @ J_mb_arm_v_

                    A[i, :8] = d_dot
                    A[i, 8:] = np.zeros((1, 6)) 
                    B[i] = (min_dist - d_safe) / (d_influence - d_safe)
                    w_p = (d_influence-min_dist)/(d_influence-d_safe) 

                    J_dj[:8] += A[i, :8] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p


            C = np.concatenate((np.zeros(2), -J_m.reshape((n_dof - 2,)), np.zeros(6)))
            # C = np.zeros(14)
            bTe = ur5e_robot.fkine(q[2:], include_base=False).A  
            θε = atan2(bTe[1, -1], bTe[0, -1])
            C[0] = - k_e * θε  # 베이스 x 위치 오차
            
            lambda_max = 0.5
            min_distance = np.min(min_dist_list)  # 장애물과의 최소 거리
            lambda_c = (lambda_max /(d_influence - d_safe)**2) * (min_distance - d_influence)**2
            J_c = lambda_c * J_dj/w_p_sum

            C += J_c # 베이스 조인트 속도에 대한 제약 조건 추가
            
            J_ = np.c_[J_mb, np.eye(6)]  # J_ 행렬 (예시)

            eTep = np.linalg.inv(T) @ H_desired.A  # 현재 위치에서의 오차 행렬

            e = np.empty(6)

            # Translational error
            e[:3] = eTep[:3, -1]

            # Angular error
            e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
            
            k = 1.5 * np.eye(6) # gain
            v = k @ e
            v[3:] *= 1.3

            lb = -np.r_[qdlim[: n_dof], 10 * np.ones(6)]
            ub = np.r_[qdlim[: n_dof], 10 * np.ones(6)]
            
            qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
            
            if qd is None:
                print("QP solution is None")
                qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.]) 

            if et > 0.5:
                qd = qd[: n_dof]
                # qd = 2 * qd
                qd[:2] = 2 * qd[:2] 
            elif 0.5 > et > 0.2 : 
                qd = qd[: n_dof]
                qd[:2] = 1.5 * qd[:2] 
            else:
                qd = qd[: n_dof]
                qd[:2] = 0.5 * qd[:2] 

                qd[2:] = 2 * qd[2:]  # UR5e 조인트 속도 증가
                # qd = 0.5 * qd
                print("et:", et)
            if et < 0.03:
                qd = qd[: n_dof]
                qd = 0.0 * qd
                print("Reached desired position!")
                # qd *= 0.0 # 목표 위치에 도달했음을 나타냄

                #  # 그래프 저장
                # plt.savefig("error_reduction_graph_x_2.png")
                # print("Graph saved as 'error_reduction_graph.png'")

                # # 시뮬레이션 종료
                # simulation_app.close()
                # break
            # 현재 시뮬레이션 시간
            current_time = world.current_time

            # Translational error 분리
            et_x = np.abs(eTep[0, -1])
            et_y = np.abs(eTep[1, -1])
            et_z = np.abs(eTep[2, -1])

            # 전체 에러 계산
            et = et_x + et_y + et_z

            # 에러 값 저장
            et_values.append(et)
            et_x_values.append(et_x)
            et_y_values.append(et_y)
            et_z_values.append(et_z)
            time_values.append(current_time)
            step_count += 1

            # 그래프 업데이트
            line_total.set_xdata(time_values)
            line_total.set_ydata(et_values)
            line_x.set_xdata(time_values)
            line_x.set_ydata(et_x_values)
            line_y.set_xdata(time_values)
            line_y.set_ydata(et_y_values)
            line_z.set_xdata(time_values)
            line_z.set_ydata(et_z_values)

            # x축 및 y축 범위 동적 업데이트
            ax.set_xlim(0, max(10, current_time))
            ax.set_ylim(0, max(1, max(et_values) * 1.1))

            plt.pause(0.01)  # 그래프 업데이트 간격
            


            wc, vc = qd[0], qd[1]  # 베이스 속도
            # wc *= 2.0

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


simulation_app.close()