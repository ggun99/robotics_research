import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.stage import update_stage
from isaacsim.core.api import World
# from isaacsim.cortex.framework.robot import CortexUr10
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions

import numpy as np
from qp_ import p_servo,jacobe,joint_velocity_damper, jacobm
from math import atan2 as atan2
import qpsolvers as qp


world = World(stage_units_in_meters=1.0)
scene = world.scene
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
robot_asset_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"#assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
prim_path = "/World/aljnu_mp"
# 1. Stage에 USD 추가
add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)
aljnu_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # aljnu_mp의 조인트 인덱스

# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'

world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()

# 2. Articulation 객체로 래핑
my_robot = Articulation(prim_path)
my_robot.initialize()

ur5e_indices = aljnu_joint_indices[4:]  # 조인트 6개라면
values = np.ones((1, len(ur5e_indices)), dtype=bool)  # 정확히 (1, 6)
my_robot.set_body_disable_gravity(values, indices=[0], body_indices=ur5e_indices)
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

# 3. QP 제어를 위한 변수 설정
n_dof = 9
qdlim = np.array([0.03] * n_dof)

# H_current : 현재 위치 4*4 matrix
H_currnet_tag = None
# H_desired : desired position 4*4 matrix
if H_currnet_tag is None:
        H_desired = H_current
        H_desired.A[:3, :3] = H_current.A[:3, :3]  # 현재 회전 행렬 유지
        H_desired.A[0, -1] -= 0.25
        H_desired.A[2, -1] += 0.125
        H_desired = H_desired
eTep = np.linalg.inv(H_current.A) @ H_desired.A  # 현재 위치에서의 오차 행렬
et = np.sum(np.abs(eTep[:3, -1]))
# Gain term (lambda) for control minimisation
Y = 0.01
# Quadratic component of objective function
Q = np.eye(n_dof + 6)
# Joint velocity component of Q
Q[: n_dof, : n_dof] *= Y
Q[:2, :2] *= 1.0 / et
# Slack component of Q
Q[n_dof :, n_dof :] = (1.0 / et) * np.eye(6)
v, _ = p_servo(H_current, H_desired.A, 1.5)
v[3:] *= 1.3

# The equality contraints
robjac = jacobe(q)  # UR5e 자코비안 계산
print("==========================")
# print(robjac.shape)

Aeq = np.c_[robjac, np.eye(6)]
# print('Aeq.shape', Aeq.shape)
beq = v.reshape((6,))

# The inequality constraints for joint limit avoidance
Ain = np.zeros((n_dof + 6, n_dof + 6))
bin = np.zeros(n_dof + 6)

# The minimum angle (in radians) in which the joint is allowed to approach
# to its limit
ps = 0.1
# The influence angle (in radians) in which the velocity damper
# becomes active
pi = 0.9
# Form the joint limit velocity damper
Ain[: n_dof, : n_dof], bin[: n_dof] = joint_velocity_damper(ps, pi, n_dof)

c = np.concatenate(
    (-jacobm().reshape((n_dof,)), np.zeros(6))
)
# Get base to face end-effector
kε = 0.5
bTe = fkine(q, include_base=False).A
θε = atan2(bTe[1, -1], bTe[0, -1])
ε = kε * θε
c[0] = -ε
# The lower and upper bounds on the joint velocity and slack variable
lb = -np.r_[qdlim[: n_dof], 10 * np.ones(6)]
ub = np.r_[qdlim[: n_dof], 10 * np.ones(6)]

# Solve for the joint velocities dq
qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
qd = qd[: n_dof]
print("qd:", qd)
if et > 0.5:
    qd *= 0.7 / et
else:
    qd *= 1.4

if et < 0.02:
    qd *= 0.0 # 목표 위치에 도달했음을 나타냄

if qd is None:
            print("QP solution is None")
            qd = np.array([0.]*n_dof)  # 기본값으로 초기화


# i = 0
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            reached_default = False  # 시뮬 초기화 시 플래그도 초기화

        current_positions = my_robot.get_joint_positions()[0][aljnu_joint_indices]

        if  not reached_default:
            # 1단계: default position으로 이동
            actions = ArticulationActions(
                joint_positions=target_joint_positions,
                joint_indices=aljnu_joint_indices
            )
            my_robot.apply_action(actions)
            print("current_positions shape:", current_positions.shape)
            print("target_joint_positions shape:", target_joint_positions.shape)
            print("current_positions:", current_positions)
            print("target_joint_positions:", target_joint_positions)
            print("diff:", np.abs(current_positions[4:10] - target_joint_positions[4:10]))
            # 목표 자세 도달 여부 체크
            if np.all(np.abs(current_positions[4:10] - target_joint_positions[4:10]) < position_tolerance):
                reached_default = True
                
                print("Reached default position!")
        else:
            # 2단계: velocity 제어
            # i += 1
            
            joint_velocities = np.zeros(16)
            # joint_velocities = np.zeros_like(current_positions)
            current_velocity = np.array(my_robot.get_joint_velocities()).reshape(-1)
            print("current_velocity: ", current_velocity)
            print("changed current_velocity: ", joint_velocities)
            my_robot.switch_control_mode("velocity")
            actions = ArticulationActions(
                joint_velocities=joint_velocities,
                joint_indices=aljnu_joint_indices
            )
            my_robot.apply_action(actions)

        print("joints : ", my_robot.get_joint_positions())

simulation_app.close()