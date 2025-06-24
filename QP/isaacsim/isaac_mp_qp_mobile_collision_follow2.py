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

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb
import matplotlib.pyplot as plt


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

def get_nearest_obstacle_distance(position, obstacles, obstacle_radius):
    """
    Calculate the distance to the nearest obstacle from a given position.
    
    Args:
        position (np.ndarray): The position from which to calculate the distance.
        obstacles (list): A list of obstacle positions.
        
    Returns:
        float: The distance to the nearest obstacle.
        index (int): The index of the nearest obstacle.
    """
    for obs in obstacles:
        obs[2] = position[2]  # Set the z-coordinate of the obstacle to the specified value
    distances = [np.linalg.norm(position - obs)-obstacle_radius for obs in obstacles]
    index = np.argmin(distances)
    # x = position[0] - obstacles[index][0]
    # y = position[1] - obstacles[index][1]
    # z = position[2] - obstacles[index][2]
    # x_norm = x / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    # y_norm = y / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    # z_norm = z / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    # # Create a directional vector
    # g_vec = np.zeros(3)  
    # g_vec[0] = x_norm
    # g_vec[1] = y_norm
    # g_vec[2] = z_norm

    # Create a directional vector
    g_vec = (position - obstacles[index])
    g_vec /= np.linalg.norm(g_vec)
    return min(distances), index, g_vec

def generate_points_between_positions(start_pos, end_pos, num_points=10):
    """
    두 3차원 위치를 이어주는 선에서 일정한 간격으로 점을 생성하는 함수.

    Args:
        start_pos (np.ndarray): 시작 위치 (3차원 좌표).
        end_pos (np.ndarray): 끝 위치 (3차원 좌표).
        num_points (int): 생성할 점의 개수 (기본값: 10).

    Returns:
        np.ndarray: 생성된 점들의 좌표 배열 (shape: num_points x 3).
    """
    # 시작 위치와 끝 위치를 연결하는 선을 따라 일정한 간격으로 점 생성
    points = np.linspace(start_pos, end_pos, num_points)
    dist_vec = (end_pos - start_pos)/num_points
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

# use Isaac Sim provided asset
robot_asset_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"#assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
prim_path = "/World/aljnu_mp"
# 1. Stage에 USD 추가
add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)
aljnu_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # aljnu_mp의 조인트 인덱스
aljnu_body_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # aljnu_mp의 바디 인덱스
n_dof = 8 # base(2) + arm(6)
k_e = 0.5
rho_i = 0.9 # influence distance
rho_s = 0.1  # safety factor
eta = 1
qdlim = np.array([1.0]*8)
qdlim[:1] = 1.5  # 베이스 조인트 속도 제한
qdlim[1] = 1.0
qlim = np.array([[-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ np.inf, np.inf, 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])

# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'

world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()

# 시뮬레이션 컨텍스트 초기화
simulation_context = SimulationContext()
obstacles_positions = np.array([[0.0, 1.0, -0.2], [2.0, 1.0, -0.2], [-1.0, -2.0, -0.2]])
# 원기둥 생성
obstacle_radius = 0.25
obstacle_height = 2.0

cylinder = DynamicCylinder(
    prim_path="/World/Xform/Cylinder1",
    name="cylinder1",
    position=obstacles_positions[0],
    radius=obstacle_radius,
    height=obstacle_height,
    color=np.array([0.8, 0.2, 0.2])
)
cylinder = DynamicCylinder(
    prim_path="/World/Xform/Cylinder2",
    name="cylinder2",
    position=obstacles_positions[1],
    radius=obstacle_radius,
    height=obstacle_height,
    color=np.array([0.8, 0.2, 0.2])
)
cylinder = DynamicCylinder(
    prim_path="/World/Xform/Cylinder3",
    name="cylinder3",
    position=obstacles_positions[2],
    radius=obstacle_radius,
    height=obstacle_height,
    color=np.array([0.8, 0.2, 0.2])
)
# desired position with sphere
desired_sphere = VisualSphere(
    prim_path="/World/Xform/sphere",
    name="desired_sphere",
    position=np.array([1.5, 1.5, 1.5]),
    radius=0.02,
    color=np.array([0.2, 0.8, 0.2])
)
# human position with sphere
human_position = np.array([0.1, 0.2, 0.97])
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
update_interval = 0.05  # 업데이트 간격 (초)

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
            print("T: ", T_sb @ T_b0 @ T_0e)
            T = T_b0 @ T_0e  # 베이스 프레임 기준 end-effector 위치
            H_current = SE3(T)  # 현재 end-effector 위치
            if T_robot is None:
                T_robot = T
                desired_sphere.set_world_pose(T_robot[:3, 3])  # 초기 desired position 설정
            if start_t is None:
                start_t = world.current_time
            # print("start_t: ", start_t)
            # print("taken_t: ", world.current_time - start_t)
            # print("current_time: ", world.current_time)
            # print("moving_t: ", moving_t)
            # taken_t = world.current_time - start_t
            # cur_p = human_sphere.get_world_pose()[0]
            # print("human error",(human_desired_position - cur_p) * taken_t / moving_t)   
            # human_error = human_desired_position - cur_p
            # go_p = cur_p + (human_error * taken_t / moving_t)  # 이동 시간 동안 목표 위치로 이동
            # print("go_p: ", go_p)
            # human_sphere.set_world_pose(go_p)  # 인간 위치 업데이트

            # cur_dp = desired_sphere.get_world_pose()[0]
            # # 목표 end-effector 위치 설정
            # T_sd = np.eye(4)
            # T_sd[0, 3] = cur_dp[0] + human_error[0] * taken_t / moving_t # 목표 x 위치
            # T_sd[1, 3] = cur_dp[1] + human_error[1] * taken_t / moving_t # 목표 y 위치
            # T_sd[2, 3] = cur_dp[2] + human_error[2] * taken_t / moving_t # 목표 z 위치
            
            # desired_position = np.array([T_sd[0, 3], T_sd[1, 3], T_sd[2, 3]])
            # desired_sphere.set_world_pose(desired_position)
            # 업데이트 간격 확인
            current_time = world.current_time
            if last_update_time is None or (current_time - last_update_time >= update_interval):
                # 업데이트 수행
                taken_t = current_time - start_t
                cur_p = human_sphere.get_world_pose()[0]
                human_error = human_desired_position - cur_p
                go_p = cur_p + (human_error * taken_t / moving_t)  # 이동 시간 동안 목표 위치로 이동
                human_sphere.set_world_pose(go_p)  # 인간 위치 업데이트

                cur_dp = desired_sphere.get_world_pose()[0]
                T_sd = np.eye(4)
                T_sd[0, 3] = cur_dp[0] + human_error[0] * taken_t / moving_t  # 목표 x 위치
                T_sd[1, 3] = cur_dp[1] + human_error[1] * taken_t / moving_t  # 목표 y 위치
                T_sd[2, 3] = cur_dp[2] + human_error[2] * taken_t / moving_t  # 목표 z 위치

                desired_position = np.array([T_sd[0, 3], T_sd[1, 3], T_sd[2, 3]])
                desired_sphere.set_world_pose(desired_position)

                # 마지막 업데이트 시간 기록
                last_update_time = current_time
            
            num_points=10
            points_between, dist_vec = generate_points_between_positions(go_p, desired_position, num_points)
            xform_pose = np.vstack((xform_pose, points_between))  # 현재 xform_pose에 점 추가

            distances = [np.linalg.norm(desired_position - obs)-obstacle_radius for obs in obstacles_positions]
            min_distance = np.min(distances)
            if min_distance < d_safe:
                print("Desired position is too close to an obstacle. Adjusting position.")
                # Adjust the desired position to be further away from the nearest obstacle
                # nearest_index = np.argmin(distances)
                # direction_vector = (desired_position - obstacles_positions[nearest_index])
                # direction_vector /= np.linalg.norm(direction_vector)
                

            T_bd = np.linalg.inv(T_sb) @ T_sd  

            H_desired = SE3(T_bd)  # 목표 end-effector 위치 
            # print('H_current: ', H_current)
            # print('H_desired: ', H_desired)
            # print('T_sd: ', T_sd)
            # print('T_sd_cal: ', T_sb @ T_b0 @ T_0e)

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
            Q[n_dof :, n_dof :] = (1.0 / et) * np.eye(6)

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

            J_c = np.zeros(n_dof+6)
            w_p_sum = 0.0
            min_dist = 0.0

            for i , pose in enumerate(xform_pose) :
                distance, index, g_vec = get_nearest_obstacle_distance(pose, obstacles_positions, obstacle_radius)
                min_dist = np.min(distance)
                if i < 4:  # mobile base wheels
                    jac_mobile = F
                    jac_mobile_v = jac_mobile[:3,:]  # 3x2 자코비안 (선형 속도)
                    d_dot = g_vec @ jac_mobile_v[:,:2] 
                    A[i, :2] = d_dot 
                    A[i, 2:] = np.zeros((1, n_dof - 2 + 6))  # arm joints
                    B[i] = (distance - d_safe) / (d_influence - d_safe) 
                    w_p = (d_influence-distance)/(d_influence-d_safe) 
                    J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p
                elif 3 < i < 9:  # UR5e joints
                    T_instant = T_b0 @ ur5e_robot.fkine(q[2:2 + i - 4],end = ur5e_robot.links[i - 4]).A
                    jac_mobile = base.tr2adjoint(T_instant.T) @ F

                    # print(ur5e_robot.links)
                    # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                    jac_arm = base.tr2adjoint(T_instant.T) @ ur5e_robot.jacob0(q[2:2 + i - 3], end = ur5e_robot.links[i - 2])
                    J_mb_instant = np.hstack((jac_mobile, jac_arm))
                    J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                    d_dot = g_vec @ J_mb_v_instant[:, :] 

                    A[i, :i-1] = d_dot
                    A[i, i-1:] = np.zeros((1, n_dof - (i-1) + 6))  # arm joints
                    B[i] = (distance - d_safe) / (d_influence - d_safe)
                    w_p = (d_influence-distance)/(d_influence-d_safe) 
                    J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p

                elif i == 9:
                    T_instant = T_b0 @ ur5e_robot.fkine(q[2:2 + i - 4],end = ur5e_robot.links[i - 4]).A
                    jac_mobile = base.tr2adjoint(T_instant.T) @ F

                    # print(ur5e_robot.links)
                    # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                    jac_arm = base.tr2adjoint(T_instant.T) @ ur5e_robot.jacob0(q[2:2 + i - 3], end = ur5e_robot.links[i - 1])
                    J_mb_instant = np.hstack((jac_mobile, jac_arm))
                    J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                    d_dot = g_vec @ J_mb_v_instant[:, :] 

                    A[i, :i-1] = d_dot
                    A[i, i-1:] = np.zeros((1, n_dof - (i-1) + 6))  # arm joints
                    B[i] = (distance - d_safe) / (d_influence - d_safe)
                    w_p = (d_influence-distance)/(d_influence-d_safe) 
                    J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p

                else:
                    T_topoint = np.eye(4)
                    T_topoint[:3, 3] = (i-9) * dist_vec  # 현재 점에서 목표 점까지의 변환 행렬
                    T_instant = T_b0 @ ur5e_robot.fkine(q[2:2 + 9 - 4],end = ur5e_robot.links[9 - 4]).A @ T_topoint
                    jac_mobile = base.tr2adjoint(T_instant.T) @ F

                    # print(ur5e_robot.links)
                    # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                    jac_arm = base.tr2adjoint(T_instant.T) @ ur5e_robot.jacob0(q[2:2 + 9 - 3], end = ur5e_robot.links[9 - 1])
                    J_mb_instant = np.hstack((jac_mobile, jac_arm))
                    J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                    d_dot = g_vec @ J_mb_v_instant[:, :] 
                    # print("d_dot: ", d_dot.shape)  # (8,)
                    A[i, :8] = d_dot
                    A[i, 8:] = np.zeros((1, 6))  # arm joints
                    B[i] = (distance - d_safe) / (d_influence - d_safe)
                    w_p = (d_influence-distance)/(d_influence-d_safe) 
                    J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                    w_p_sum += w_p


            C = np.concatenate((np.zeros(2), -J_m.reshape((n_dof - 2,)), np.zeros(6)))
            lambda_max = 1.5
            lambda_c = (lambda_max /(d_influence - d_safe)**2) * (min_dist - d_influence)**2
            C += (lambda_c * J_c/w_p_sum)  # 베이스 조인트 속도에 대한 제약 조건 추가

            bTe = ur5e_robot.fkine(q[2:], include_base=False).A  
            θε = atan2(bTe[1, -1], bTe[0, -1])
            C[0] = - k_e * θε  # 베이스 x 위치 오차

            # A[: n_dof, : n_dof], B[: n_dof] = joint_velocity_damper(ps=rho_s, pi=rho_i, n=n_dof, gain=eta)  # joint velocity damper
    
            J_ = np.c_[J_mb, np.eye(6)]  # J_ 행렬 (예시)

            # eTep = np.linalg.inv(H_current) @ H_desired.A  # 현재 위치에서의 오차 행렬

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
            qd = qd[: n_dof]
            # print("qd:", qd)
            if qd is None:
                print("QP solution is None")
                qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.]) 

            if et > 0.5:
                qd *= 1.4
            else:
                qd =qd
                print("et:", et)
            if et < 0.02:
                print("Reached desired position!")
                qd *= 0.0 # 목표 위치에 도달했음을 나타냄

                 # 그래프 저장
                plt.savefig("error_reduction_graph_x_2.png")
                print("Graph saved as 'error_reduction_graph.png'")

                # 시뮬레이션 종료
                simulation_app.close()
                break
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
            wc *= 10.0

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