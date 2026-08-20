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
from QP_class import QP

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb
import matplotlib.pyplot as plt


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
desired_position = desired_sphere.get_world_pose()[0].copy()
human_position = human_sphere.get_world_pose()[0].copy()

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
            current_time = world.current_time
            if start_t is None:
                start_t = current_time

            if last_update_time is None or (current_time - last_update_time >= update_interval):
                taken_t = current_time - start_t
                human_position = human_sphere.get_world_pose()[0].copy()
                human_error = human_desired_position - human_position
                human_position = human_position + (human_error * taken_t / moving_t)
                human_sphere.set_world_pose(human_position)

                desired_position = desired_sphere.get_world_pose()[0].copy()
                desired_position = desired_position + (human_error * taken_t / moving_t)
                desired_sphere.set_world_pose(desired_position)

                last_update_time = current_time

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

            qd, eTep = QP(n_dof, qdlim, q, x, y, z, quat, ur5e_robot, d_safe, d_influence, k_e, obstacles_positions, obstacle_radius, xform_pose).solve(
                desired_position=desired_position,
                human_position=human_position,
            )
            
            # Translational error 분리
            et_x = np.abs(eTep[0, -1])
            et_y = np.abs(eTep[1, -1])
            et_z = np.abs(eTep[2, -1])

            # 전체 에러 계산
            et = et_x + et_y + et_z

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
            joint_velocities[4:10] = qd[2:8]

            actions = ArticulationActions(
                joint_velocities=joint_velocities,
                joint_indices=aljnu_joint_indices
            )
            my_robot.apply_action(actions)


simulation_app.close()