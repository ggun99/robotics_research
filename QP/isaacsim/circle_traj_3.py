import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api import World, SimulationContext
from isaacsim.core.api.objects import DynamicCylinder, VisualSphere, VisualCylinder
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

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import random
from circle_traj_generate import CircularTrajectoryPlanner
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D
from carb._carb import Float3, ColorRgba
from spatialmath.base import trnorm

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
    # 엔드 이펙터의 변환 행렬
    # T_e = T_cur @ T  # 월드 좌표계에서 엔드 이펙터 좌표계로의 변환
    g_vec = np.zeros(3)
    obstacles_local = []
    obs_real = []
    for obs in obstacles:
        obs[2] = position[2]
        obs_real.append(obs)
        obs_homogeneous = np.append(obs, 1)  # 동차 좌표로 확장
        obs_local = np.linalg.inv(T_e) @ obs_homogeneous
        obstacles_local.append(obs_local[:3])  # 3차원으로 변환
        # position_homogeneous = np.append(position, 1)  # 동차 좌표로 확장
        # position_local = np.linalg.inv(T_e) @ position_homogeneous
        # position_local = position_local[:3]  # 3차원으로 변환
        
    position_homogeneous = np.append(position, 1)  # 동차 좌표로 확장
    position_local = np.linalg.inv(T_e) @ position_homogeneous
    position_local = position_local[:3]  # 3차원으로 변환
    # position_local = position  # 3차원으로 변환
    
    distances = [((np.linalg.norm(position_local - obse)) - obstacle_radius) for obse in obstacles_local]
    index = np.argmin(distances)

    g_vec = (position - obs_real[index])
    g_vec /= np.linalg.norm(g_vec) 
    # print('g_vec: ', g_vec)
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
def calculate_natural_rotation(T_cur, target_position):
    """
    로봇의 현재 상태에서 가장 자연스러운 방향으로 회전 행렬을 계산합니다.

    Args:
        T_cur (np.ndarray): 현재 엔드 이펙터의 변환 행렬 (4x4).
        target_position (np.ndarray): 목표 위치 (3차원 벡터).

    Returns:
        np.ndarray: 새로운 회전 행렬 (3x3).
    """
    # 현재 엔드 이펙터의 z축 방향 벡터 (월드 좌표계 기준)
    cur_z_axis = T_cur[:3, 2]

    # 목표 위치로의 방향 벡터 계산
    current_position = T_cur[:3, 3]
    direction_vector = target_position - current_position
    direction_vector /= np.linalg.norm(direction_vector)  # 정규화

    # 새로운 z축 방향 설정 (목표 방향)
    new_z_axis = direction_vector

    # 새로운 y축 계산 (현재 z축과 목표 방향의 외적)
    new_y_axis = np.cross(cur_z_axis, new_z_axis)
    if np.linalg.norm(new_y_axis) < 1e-6:  # 외적 결과가 0에 가까우면, z축과 목표 방향이 평행
        new_y_axis = np.array([0, 1, 0])  # 기본 y축 설정
    new_y_axis /= np.linalg.norm(new_y_axis)  # 정규화

    # 새로운 x축 계산 (y축과 z축의 외적)
    new_x_axis = np.cross(new_y_axis, new_z_axis)
    new_x_axis /= np.linalg.norm(new_x_axis)  # 정규화

    # 새로운 회전 행렬 생성
    rotation_matrix = np.vstack([new_x_axis, new_y_axis, new_z_axis]).T
    return rotation_matrix

# et 값과 시간을 저장할 리스트
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

# Matplotlib 인터랙티브 모드 활성화
# plt.ion()
plt.ioff()
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
obstacles_positions = np.array([
     [1.25,0.5, 1.]])
step_count = 0

ur5e_robot = rtb.models.UR5()
world = World(stage_units_in_meters=1.0)
scene = world.scene
assets_root_path = get_assets_root_path()

# Gf.Quatd 객체를 numpy 배열로 변환
orientation_quat = Gf.Quatd(0.7071068, 0, 0.7071068, 0)  # 기존 쿼터니언
orientation_np = np.array([orientation_quat.GetReal(), *orientation_quat.GetImaginary()])  # numpy 배열로 변환

camera = Camera(
    prim_path="/World/MyCamera",
    resolution=(1280, 720),
    frequency=20,
    position=(4.04, 1.4318, 8.66154),
    orientation=(0.2690812, 0.761412, -0.5560824, -0.1965182)
)
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
# k_e = 0.8
rho_i = 0.9 # influence distance
rho_s = 0.1  # safety factor
eta = 1
qdlim = np.array([0.7]*8)
qdlim[:1] = 1.5  # 베이스 조인트 속도 제한
qdlim[1] = 0.6
qlim = np.array([[-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ np.inf, np.inf, 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])

# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'

world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()


# 파라미터 설정
map_size = 6
num_obstacles = 3
min_distance_between_obstacles = 1.5
obstacle_radius_range = (0.25, 0.25)

planner = CircularTrajectoryPlanner(radius=3.0, center=(0, 0, 0), height=1.0)

# 시뮬레이션 컨텍스트 초기화
simulation_context = SimulationContext()
traj_num_points = 500
human_smoothed_path = planner.generate_circle_path(num_points=traj_num_points)
# 사람의 경로 생성
human_start_position = np.array([0.0, 0.0, 0.0])  # 사람의 시작 위치
human_goal_position = np.array([3.5, 2.0, 1.2])
# RRTPlanner를 사용하여 사람의 경로 생성
# human_path = planner.plan(start=human_start_position, goal=human_goal_position)
# if human_path is None:
#     print("사람의 경로를 찾을 수 없습니다.")
#     simulation_app.close()
#     exit()
 # 속도 프로파일 생성
human_velocity_profile = planner.generate_velocity_profile(human_smoothed_path, v_max=0.2)

# 전체 궤적 생성
trajectories = planner.generate_full_trajectory_with_offset(human_smoothed_path, human_velocity_profile, dt=0.1, time_offset=15)


human_trajectory = trajectories
# robot_trajectory = trajectories["robot_trajectory"]
# 시뮬레이션 루프에서 사람의 움직임 따라가기
human_trajectory_index = 0

# 원기둥 생성
obstacle_radius = 0.2
obstacle_height = 2.3

# 그래프 그리기
plt.figure(figsize=(8, 8))
plt.title("Simulation Environment (X-Y Plane)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)

# 출발 지점 그리기
plt.scatter(human_start_position[0], human_start_position[1], color='blue', label="Robot Start Position", s=100)
# 로봇의 시작 위치
plt.scatter(0.6088934, 0.1442274, color='red', label="End-Effector Start Position", s=100)  # 로봇의 시작 위치

# 목표 지점 그리기
# plt.scatter(human_goal_position[0], human_goal_position[1], color='green', label="Target Goal Position", s=100)

# plt.scatter(robot_trajectory["x"][-1], robot_trajectory["y"][-1], color='orange', label="Robot Goal Position", s=100)  # 로봇의 목표 위치
# Human trajectory 그리기
plt.plot(human_trajectory["x"], human_trajectory["y"], color='blue', linestyle='--', label="Target Trajectory")

# 범례 추가
plt.legend()

# 축 설정
plt.axis("equal")  # x축과 y축 비율 동일하게 설정
plt.xlim(-4.0,4.0)  # x축 범위
plt.ylim(-4.0,4.0)  # y축 범위

first_plot = True

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
d_safe = 0.06
d_influence = 2.0

# moving human
moving_t = 30.0  # 이동 시간
start_t = None  # 현재 시간
human_desired_position = np.array([-1.5, -1.0, 0.97])  # 목표 위치
T_robot = None  # 로봇의 현재 위치를 저장할 변수

# 마지막 업데이트 시간을 저장할 변수
last_update_time = None
update_interval = 0.1  # 업데이트 간격 (초)

def generate_rectangle_edge_points(center, width, height, num_points_per_edge):
    """
    직사각형의 네 변에 점들을 생성하는 함수.

    Args:
        center (tuple): 직사각형의 중심 좌표 (x, y, z).
        width (float): 직사각형의 너비.
        height (float): 직사각형의 높이.
        num_points_per_edge (int): 각 변에 생성할 점의 개수.

    Returns:
        np.ndarray: 직사각형 네 변의 점들의 로컬 좌표 (shape: num_points x 3).
    """
    half_width = width / 2
    half_height = height / 2

    # 각 변에 점 생성
    top_edge = np.linspace([-half_width, half_height, 0], [half_width, half_height, 0], num_points_per_edge)
    bottom_edge = np.linspace([-half_width, -half_height, 0], [half_width, -half_height, 0], num_points_per_edge)
    left_edge = np.linspace([-half_width, -half_height, 0], [-half_width, half_height, 0], num_points_per_edge)
    right_edge = np.linspace([half_width, -half_height, 0], [half_width, half_height, 0], num_points_per_edge)

    # 모든 점 합치기
    points = np.vstack([top_edge, bottom_edge, left_edge, right_edge])
    return points

def set_orientation_from_vector(vector):
    """
    원기둥의 방향을 주어진 벡터에 맞게 설정합니다.
    """
    # 벡터를 정규화
    vector = vector / np.linalg.norm(vector)

    # 기본 z축(0, 0, 1)과 주어진 벡터 사이의 회전 행렬 계산
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, vector)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, vector), -1.0, 1.0))
    rotation_matrix = R.from_rotvec(rotation_axis * rotation_angle).as_matrix()

    return rotation_matrix
 
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

# 직사각형 점 생성
width = 0.6  # 700 mm
height = 0.93  # 930 mm
num_points_per_edge = 5  # 각 변에 5개의 점 생성
rectangle_edge_points_local = generate_rectangle_edge_points(center=(0, 0, 0), width=width, height=height,
                                                             num_points_per_edge=num_points_per_edge)


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

        rectangle_points_world = transform_points(rectangle_edge_points_local, position=(x, y, z), quaternion=quat)

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
            base_link = "/World/aljnu_mp/base_link"
            prim_base = XFormPrim(base_link)
            base_pose, base_quat = prim_base.get_world_poses()
            scatter_mob_x.append(x)
            scatter_mob_y.append(y)
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

            xform_pose = np.vstack((rectangle_points_world, np.array([
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
            T_cur = T_sb @ T  # 현재 로봇 위치 (월드 좌표계 기준)
            cur_p = T_cur[:3, 3]  # 현재 엔드 이펙터 위치 (월드 좌표계 기준)

            # 엔드 이펙터의 변환 행렬
            T_e = T_cur  # 월드 좌표계에서 엔드 이펙터 좌표계로의 변환
            if T_robot is None:
                T_robot = T
                desired_sphere.set_world_pose(T_robot[:3, 3])
            if start_t is None:
                start_t = world.current_time
     
            # 사람이 궤적의 마지막 위치에 도달했는지 확인
            if human_trajectory_index >= len(human_trajectory["x"]):
                # print("Human has reached the final position in the trajectory.")
                human_position = np.array([
                    human_trajectory["x"][-1],
                    human_trajectory["y"][-1],
                    human_trajectory["z"][-1]
                ])
              
            else:
                # 시뮬레이션 루프에서 사용
                human_position = np.array([
                    human_trajectory["x"][human_trajectory_index],
                    human_trajectory["y"][human_trajectory_index],
                    human_trajectory["z"][human_trajectory_index]
                ])

                robot_target_position = np.array([
                    human_trajectory["x"][human_trajectory_index],
                    human_trajectory["y"][human_trajectory_index],
                    human_trajectory["z"][human_trajectory_index]
                ])
            # print('human_trajectory_length: ', len(human_trajectory))
            # 현재 위치와 목표 위치 간의 거리 계산
            distance_to_target = np.linalg.norm(human_position - T_e[:3, 3])

            # 거리 임계값 설정
            distance_threshold = 0.15  # 15cm (필요에 따라 조정)

            # 업데이트 간격 확인 또는 거리 기반 조건
            current_time = world.current_time
            time_condition = last_update_time is None or (current_time - last_update_time >= update_interval)
            distance_condition = distance_to_target < distance_threshold
            
            cur_ee = T_e[:3,3]

            sight_vec = T_e[:3,0]
            sight_vec /= np.linalg.norm(sight_vec)
            # robot_target_position을 엔드 이펙터 좌표계로 변환
            robot_target_position_homogeneous = np.append(human_position, 1)  # 동차 좌표로 확장
            robot_target_position_local = np.linalg.inv(T_e) @ robot_target_position_homogeneous
            robot_target_position_local = robot_target_position_local[:3]  # 3차원으로 변환
            
            # 목표 방향 계산 (엔드 이펙터 좌표계 기준)
            direction_vector = robot_target_position_local
            direction_vector /= np.linalg.norm(direction_vector)  # 방향 벡터 정규화
            draw.clear_lines()  # 이전에 그린 선들을 모두 지웁니다.

            start_point_1 = Float3(cur_ee)  # Float3 형식으로 변환
            end_point_1 = Float3(
                start_point_1[0] +  sight_vec[0],
                start_point_1[1] +  sight_vec[1],
                start_point_1[2] +  sight_vec[2]
            )  # Float3 형식으로 변환

            # 색상과 크기 정의
            color = [ColorRgba(1.0, 0.0, 0.0, 1.0)]  # 빨간색 (RGBA 형식)
            sizes = [0.4]  # 크기 리스트
            
            # g_vec 벡터 그리기
            draw.draw_lines(
                [start_point_1],  # 시작점 리스트
                [end_point_1],  # 끝점 리스트
                color,  # 색상 리스트
                sizes  # 크기 리스트
            )

            # 자연스러운 방향으로 회전 행렬 계산
            rotation_matrix = calculate_natural_rotation(T_cur, robot_target_position)

            # 로봇의 목표 위치 설정
            T_sd = np.eye(4)
            T_sd[:3, :3] = rotation_matrix
            det = np.linalg.det(rotation_matrix)
            orthogonality_check = np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3))

            if not np.isclose(det, 1.0) or not orthogonality_check:
                print("Invalid rotation matrix detected. Normalizing...")
                U, _, Vt = np.linalg.svd(rotation_matrix)
                rotation_matrix_normalized = U @ Vt
                T_sd[:3, :3] = rotation_matrix_normalized

            T_sd[0, 3] = robot_target_position[0]
            T_sd[1, 3] = robot_target_position[1]
            T_sd[2, 3] = robot_target_position[2]
            desired_sphere.set_world_pose(T_sd[:3, 3])  # 목표 위치 업데이트
            scatter_x.append(T_e[0,3])
            scatter_y.append(T_e[1,3])

            if distance_condition:
                # 거리 조건으로 업데이트된 경우 로그 출력
                if distance_condition:
                    print(f"Target reached! Distance: {distance_to_target:.3f}m, Moving to next target: {human_trajectory_index}")
                
                human_sphere.set_world_pose(robot_target_position)  # 사람의 위치 업데이트
                human_trajectory_index += 1  # 인간 위치 업데이               
                # 마지막 업데이트 시간 기록
                last_update_time = current_time
            # 엔드 이펙터의 z축 방향 벡터
            cur_z_axis = T_cur[:3, 0]  # T_cur의 회전 행렬에서 z축 방향 벡터

            # T_cur에서 T_sd로 향하는 단위 벡터
            direction_vector = human_position - T_cur[:3, 3]  # T_cur에서 T_sd로 향하는 벡터
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)  # 단위 벡터
            start_point_2 = Float3(cur_ee)  # Float3 형식으로 변환
            # print('g_vec', g_vec)
            end_point_2 = Float3(
                start_point_2[0] +  direction_unit_vector[0],
                start_point_2[1] +  direction_unit_vector[1],
                start_point_2[2] +  direction_unit_vector[2]
            )  # Float3 형식으로 변환
            # 색상과 크기 정의
            color = [ColorRgba(0.0, 0.0, 1.0, 1.0)]  # 빨간색 (RGBA 형식)
            sizes = [0.4]  # 크기 리스트
            
            # g_vec 벡터 그리기
            draw.draw_lines(
                [start_point_2],  # 시작점 리스트
                [end_point_2],  # 끝점 리스트
                color,  # 색상 리스트
                sizes  # 크기 리스트
            )
            # 두 벡터 사이의 각도 계산
            cos_theta = np.dot(direction_unit_vector, sight_vec)  # 내적 계산
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 각도 계산 (라디안)
            theta_values.append(np.degrees(theta))
           
            T_bd = np.linalg.inv(T_sb) @ T_sd  
            # print('T_bd: ', T_bd)
            try:
                # 1. T_bd가 유효하다면, trnorm 없이 깨끗하게 SE3 객체를 생성합니다.
                H_desired = SE3(T_bd)  # 목표 end-effector 위치
                
            except ValueError:
                # 2. ValueError (bad argument)가 발생했다면, 
                # T_bd의 회전 부분에 오차가 있다는 뜻이므로 trnorm()을 적용하여 교정합니다.
                print("Warning: T_bd matrix required normalization due to floating-point error.")
                T_bd_normalized = trnorm(T_bd)
                H_desired = SE3(T_bd_normalized) # 정규화된 행렬로 다시 객체 생성
            # H_desired = SE3(T_bd)  # 목표 end-effector 위치

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
            

            T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

            et = np.sum(np.abs(T_error[:3, -1])) 

            # Quadratic component of objective function
            Q = np.eye(n_dof + 6)

            # Joint velocity component of Q
            Q[:2, :2] *= 1.0 / (et * 100)

            # Slack component of Q
            Q[n_dof :, n_dof :] = (1. / et) * np.eye(6)
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

            J_dj = np.zeros(n_dof+6)
            J_h = np.zeros(n_dof+6)
            w_p_sum = 0.0
            A = np.zeros((n_dof + 2 + 20 , n_dof + 6))
            B = np.zeros(n_dof + 2 + 20)
            # A = np.zeros((n_dof + 16 , n_dof + 6))
            # B = np.zeros(n_dof + 16)
            
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

            obstacle_center = obstacles_positions[0, :3] # 가장 가까운 장애물의 중심
            relative_position = np.linalg.norm(cur_p - obstacle_center)  # 장애물 중심과의 거리

            # 장애물의 반지름과 영향 거리 계산
            obstacle_radius_effective = obstacle_radius + d_safe
            relative_ratio = relative_position / obstacle_radius_effective  # 상대적 거리 비율 (0 ~ 1)

            # 가중치 계산 (장애물 중심에 가까울수록 perpendicular_vec의 영향이 커짐)
            perpendicular_weight = max(0, 1 - relative_ratio)  # 장애물 중심에서 멀어질수록 영향 감소
            direction_weight = 1 - perpendicular_weight  # 목표 방향의 영향 증가

            # 회피 벡터 계산
            # print("g_vec:", g_vec)
            avoid_vec = (g_vec) 

            avoid_vec /= np.linalg.norm(avoid_vec)  # 정규화

            # print('min_dist overall:', min_dist)
            pose_closest = xform_pose[min_index]

            for i , pose in enumerate(xform_pose) :    
                if i < 24:  # mobile base wheels

                    d_dot = (avoid_vec) @ J_mb_v   # 장애물 피하는 방향으로의 자코비안
                    
                    A[i, :8] = -d_dot 
                    A[i, 8:] = np.zeros((1, 6)) 
                    B[i] = (min_dist_list[i] - d_safe) / ((d_influence - d_safe))
                    w_p = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                    J_dj[:8] +=  (-d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += np.abs(w_p)

                else:  # UR5e joints
                    
                    J_mb_arm_v = np.hstack([np.zeros((3, i - 22)), J_a_e[:3, i - 24:]])

                    d_dot = (avoid_vec) @ J_mb_arm_v

                    A[i, :8] = -d_dot
                    A[i, 8:] = np.zeros((1, 6)) 
                    B[i] = (min_dist_list[i] - d_safe) / ((d_influence - d_safe))
                    w_p = (d_influence-min_dist_list[i])/(d_influence-d_safe) 
                    J_dj[:8] += (-d_dot) * (w_p)  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += np.abs(w_p)

            # 그래프 표시
            if first_plot:
                # 그래프 저장
                plt.savefig("first_simulation_environment_with_trajectories.png")
                print("Graph saved as 'first_simulation_environment_with_trajectories.png'")
                first_plot = False
            
            bTe = ur5e_robot.fkine(q[2:], include_base=False).A  
            θε = atan2(bTe[1, -1], bTe[0, -1])
            weight_param = np.sum(np.abs(human_goal_position - T_e[:3, 3]))
           
            C1 = np.concatenate((np.zeros(2), -J_m.reshape((n_dof - 2,)), np.zeros(6)))
            C2 = np.zeros(n_dof + 6)
            C2[0] = - 5. * θε  # 베이스 x 위치 오차

            lambda_max = 5.
            min_distance = np.min(min_dist_list)  # 장애물과의 최소 거리
            if min_distance <= d_influence :
                lambda_c = (lambda_max /(d_influence - d_safe)**2) * (min_distance - d_influence)**2
            else:
                lambda_c = 0.0
            J_c = lambda_c * J_dj/w_p_sum
            # print("w_p_sum:", w_p_sum)
            C3 = J_c # 베이스 조인트 속도에 대한 제약 조건 추가
            J_mb_w_h = direction_unit_vector @ J_mb_w
            epsilon = 1e-6  # 최소 허용 값
            lambda_h = 0.3 * np.maximum(np.abs(theta), epsilon)
            print("lambda_h:", lambda_h)
            J_h[:8] = lambda_h * J_mb_w_h
            C4 = J_h  # 회전 제어 항

            # C4 = np.concatenate((np.zeros(5), np.ones((n_dof - 5)), np.zeros(6)))
            # C4[4:] *= 1./np.abs(np.degrees(theta)) 
            epsilon = 1e-6  # 최소 허용 값
            # C4[4:] *= 1.5 / np.maximum(np.abs(np.degrees(theta)), epsilon)
            w1 = 1.
            w2 = 1.
            w3 = 0.
            w4 = 1.
            C = w1 * C1 + w2 * C2 + w3 * C3 + w4 * C4
            J_ = np.c_[J_mb, np.eye(6)]  # J_ 행렬 (예시)

            eTep = np.linalg.inv(T) @ H_desired.A  # 현재 위치에서의 오차 행렬

            e = np.zeros(6)

            # Translational error
            e[:3] = eTep[:3, -1]

            # Angular error
            e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
            # print(f"e: {e}")
            k = np.eye(6)  # gain
            k[:3,:] *= 4.0 # gain
            v = k @ e

            lb = -np.r_[qdlim[: n_dof], 10 * np.ones(6)]
            ub = np.r_[qdlim[: n_dof], 10 * np.ones(6)]
            # qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
            
            x_ = cp.Variable(n_dof+6) 
            err_default_q = np.abs(current_joint_positions[4:10] - target_joint_positions[4:10])
          # 5. 목적함수
            objective = cp.Minimize(0.5 * cp.quad_form(x_, Q) + C.T @ x_ ) ##+ w * rotation_control) #+ np.abs(0.3 * theta) ) # + 10. * err_default_q.sum())
            # 예시 제약조건 (속도 제한 등)
            constraints = [
                x_ >= lb,
                x_ <= ub,
                # cp.abs(s) <= 0.1,  # 슬랙이 너무 커지는 걸 방지 (선택 사항) # 예시 제약조건 (속도 제한 등)
                J_ @ x_ == v,  # 엔드이펙터 속도 추종
            ]
            # 풀기
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS)  # ECOS, OSQP, SCS, etc.

            # 결과
            qd = x_.value
            
            if x_.value is not None:
                quad_term = 0.5 * np.dot(x_.value.T, Q @ x_.value)  # 이차항
                linear_term = np.dot(C.T, x_.value)  # 선형항
                # print(f"Cost function linear term C: { np.dot(C.T, x_.value)}, C1: { np.dot(C1.T , x_.value)}, C2: { np.dot(C2.T , x_.value)}, C3: { np.dot(C3.T , x_.value)}, C4: { np.dot(C4.T , x_.value)}")
                # print(f"Cost function linear term C: { np.dot(C.T, x_.value)}, C1: { np.dot(C1.T , x_.value)}, C2: { np.dot(C2.T , x_.value)}, C4: { np.dot(C4.T , x_.value)}")


            if qd is None:
                print("QP solution is None")
                qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.]) 

            if et > 0.5:
                qd = qd[: n_dof]
            elif 0.5 > et > 0.2 : 
                qd = qd[: n_dof]
            else:
                qd = qd[: n_dof]
            qd = qd[:8]
            print(human_trajectory_index)
            # if human_trajectory_index >= traj_num_points:
                
            #         qd = qd[: n_dof]
            #         qd = 0.0 * qd
            #         print("Reached desired position!")
            #         # 그래프 저장
            #         plt.legend()
            #         plt.savefig("simulation_environment_with_trajectories_100.png")
            #         print("Graph saved as 'simulation_environment_with_trajectories.png'")
            #         plt.close()

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
            if human_trajectory_index >= traj_num_points :
                qd = qd[: n_dof]
                qd = 0.0 * qd
                print("Reached desired position!")
                # 그래프 저장
                plt.scatter(scatter_x, scatter_y, color='orange', s=10, label="End-Effector Trajectory")
                plt.scatter(scatter_mob_x, scatter_mob_y, color='cyan', s=10, label="End-Effector Trajectory")
                plt.legend()
                plt.savefig("radius_3.0_with_trajectories_011_0.2_0.2_0.3.png")
                print("Graph saved as 'simulation_environment_with_trajectories.png'")
                plt.close()
                # 첫 번째 그래프 저장 (Error Reduction Graph)
                # plt.figure(figsize=(10, 6))
                # plt.plot(time_values, et_values, label="Total Error (et)", color="blue")
                # plt.plot(time_values, et_x_values, label="Error in X (et_x)", color="red")
                # plt.plot(time_values, et_y_values, label="Error in Y (et_y)", color="green")
                # plt.plot(time_values, et_z_values, label="Error in Z (et_z)", color="orange")
                # plt.axhline(y=0.02, color='purple', linestyle='--', label="Threshold (0.02)")
                # plt.xlabel("Simulation Time (s)")
                # plt.ylabel("Error (m)")
                # plt.title("Error Reduction Over Time")
                # plt.legend()
                # plt.grid()
                # plt.savefig("error_reduction_graph_.png")
                # print("Graph saved as 'error_reduction_graph_.png'")
                # plt.close()  # 그래프를 닫아 다음 그래프와 겹치지 않도록 함
                

                
                # 시뮬레이션 종료
                simulation_app.close()
                break


            wc, vc = qd[0], qd[1]  # 베이스 속도

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