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
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.types import ArticulationActions
from pxr import UsdGeom

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2 as atan2
import qpsolvers as qp
from spatialmath import base, SE3
import roboticstoolbox as rtb

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
qdlim = np.array([0.03]*8)
qdlim[:1] = 1.5   # 베이스 조인트 속도 제한
qlim = np.array([[-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ np.inf, np.inf, 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])

# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'

world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()

# 2. Articulation 객체로 래핑
my_robot = Articulation(prim_path)
my_robot.initialize()
# print("aljnu_joint_indices:", my_robot.get_body_index('ur5e_shoulder_link') )

aljnu_indices = aljnu_body_indices[1:] #aljnu_joint_indices[4:]  
values = np.ones((1, len(aljnu_indices)), dtype=bool)  
my_robot.set_body_disable_gravity(values, indices=[0], body_indices=aljnu_indices) 

# body_indices = aljnu_indices  # 확인하려는 바디 인덱스
# disable_gravities = my_robot.get_body_disable_gravity()

# print(f"gravity disabled: {disable_gravities}")
# stiffness = np.tile(np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]), (10, 1))  # 조인트 10개에 대한 강성
# dampings = np.tile(np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), (10, 1))  # 조인트 10개에 대한 감쇠

# my_robot.set_gains(kps = stiffness, kds = dampings, joint_indices = aljnu_joint_indices[:10])  # UR5e 조인트에 대한 강성 및 감쇠 설정

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
        
        dx = x - x0
        dy = y - y0
        
        quat = mobile_base_quat[0]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('zyx', degrees=False)  # 'xyz' 순서로 euler angles 추출
        # print("euler:", euler)
        yaw = euler[0]  
        # 초기 heading(θ0) 기준으로 전진 거리
        forward = dx * np.cos(yaw) + dy * np.sin(yaw)
        q = np.zeros(8)
        q[0] = yaw  
        q[1] = forward  
        q[2:] = current_joint_positions[4:10]  # UR5e 조인트 위치
        # print('q:', q)

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
            target_link_prim_path = "/World/aljnu_mp/ur5e_wrist_3_link"
            prim = XFormPrim(target_link_prim_path)
            ur5e_last_link_pose, ur5e_last_link_quat = prim.get_world_poses()
            
            quat_ur = ur5e_last_link_quat[0]
            # 정규화
            quat_xyzw = np.array([quat_ur[1], quat_ur[2], quat_ur[3], quat_ur[0]])
            quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)

            r_ur = R.from_quat(quat_xyzw)

            rot_matrix = r_ur.as_matrix()
            T = np.zeros((4,4))
            T[:3,:3] = rot_matrix
            T[0,3] = ur5e_last_link_pose[0][0]  # x position
            T[1,3] = ur5e_last_link_pose[0][1]
            T[2,3] = ur5e_last_link_pose[0][2]
            T[3,3] = 1.0
            H_current = SE3(T) 

            wTe = ur5e_robot.fkine(q[2:])  # UR5e FK 계산
            print("T:", T)
            print("wTe:", wTe)
            if H_desired is None:
                H_desired = SE3(T)
                H_desired.A[:3, :3] = SE3(T).A[:3, :3]  # 현재 회전 행렬 유지
                H_desired.A[0, -1] -= 0.25
                H_desired.A[2, -1] -= 0.05
                H_desired = H_desired
            else:
                H_desired = H_desired
                print('desired: ', H_desired)
                print('current: ', SE3(T))

            # 전체 자코비안 J (6x8)
            lx = 0.1015
            X_e = T[0, 3]  # x position
            Y_e = T[1, 3]  # y position
            J_p = np.array([[np.cos(yaw), -lx*np.sin(yaw)-(X_e*np.sin(yaw)+Y_e*np.cos(yaw))],
                            [np.sin(yaw), lx*np.cos(yaw)+(X_e*np.cos(yaw)-Y_e*np.sin(yaw))],
                            [        0.0, 0.0],
                            [        0.0, 0.0], 
                            [        0.0, 0.0],
                            [        0.0, 1.0]])
            J_a = ur5e_robot.jacobe(q[2:])
            J_mb = np.hstack((J_p, J_a))  # 6x8 자코비안 (선형 속도 + 각속도)
            J_mb_v = J_mb[:3, :]  # 3x8 자코비안 (선형 속도)
            J_mb_w = J_mb[3:, :]  # 3x8 자코비안 (각속도)


            # J_arm = robot.jacobe(q[2:8])
            # J_mb = J = robot.jacobe(q[2:8])  # 베이스 프레임 기준 자코비안 (6x6)
            # J_mb_v = J_mb[:3, :]  # 3x6 자코비안 (선형 속도)
            # J_mb_w = J_mb[3:, :]  # 3x6 자코비안 (각속도)

            # print(my_robot.body_names)
            # ['base_link', 'front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link', 
            # 'ur5e_shoulder_link', 'ur5e_upper_arm_link', 'ur5e_forearm_link', 'ur5e_wrist_1_link', 'ur5e_wrist_2_link', 'ur5e_wrist_3_link', 
            # 'robotiq_85_left_inner_knuckle_link', 'robotiq_85_left_knuckle_link', 'robotiq_85_right_inner_knuckle_link', 'robotiq_85_right_knuckle_link', 'robotiq_85_left_finger_tip_link', 'robotiq_85_right_finger_tip_link']
            
            T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

            et = np.sum(np.abs(T_error[:3, -1])) #+ np.exp(-16)  # Euclidean distance (예시)

            # Gain term (lambda) for control minimisation
            Y = 0.01

            # Quadratic component of objective function
            Q = np.eye(n_dof + 6)

            # Joint velocity component of Q
            Q[: n_dof, : n_dof] *= Y
            Q[:2, :2] *= 1.0 / et

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
            m = J_a @ J_a.T 
            m_det = np.linalg.det(m)  
            m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

            JJ_inv = np.linalg.inv((J_a @ J_a.T))  #.reshape(-1, order='F')

            # Compute manipulability Jacobian only for arm joints
            J_m = np.zeros((n_dof-2,1))
            for i in range(n_dof-2):
                c = J_a @ np.transpose(H[i, :, :])  # shape: (6,6)
                J_m[i,0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")
            
            C = np.concatenate((np.zeros(2), -J_m.reshape((n_dof - 2,)), np.zeros(6)))

            θε = atan2(T[1, -1], T[0, -1])
            C[0] = - k_e * θε  # 베이스 x 위치 오차

            A = np.zeros((n_dof + 6, n_dof + 6))
            B = np.zeros(n_dof + 6)
            A[: n_dof, : n_dof], B[: n_dof] = joint_velocity_damper(ps=rho_s, pi=rho_i, n=n_dof, gain=eta)  # joint velocity damper
    
            J_ = np.hstack((J_mb, np.eye(6)))  # J_ 행렬 (예시)

            eTep = np.linalg.inv(H_current) @ H_desired.A  # 현재 위치에서의 오차 행렬
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
            # print('A:', A)
            # print('B:', B)
            qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
            qd = qd[: n_dof]
            print("qd:", qd)
            if qd is None:
                print("QP solution is None")
                qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.]) 

            if et > 0.5:
                qd *= 0.7 / et
            else:
                qd *= 1.4

            if et < 0.02:
                qd *= 0.0 # 목표 위치에 도달했음을 나타냄
            
            wc, vc = qd[0], qd[1]  # 베이스 속도

            # vc = float(l_v)
            # wc = -float(a_v)
            # vc = 0.0
            # wc = 0.6  # 작은 속도에는 회전을 하지 않음. 또한 회전 시에 제자리에서 회전하지 않음.
            r = 0.165
            l = 0.582
            w_R = vc/r + l*wc/(2*r)
            w_L = vc/r - l*wc/(2*r)
            print("w_R: ", w_R, "w_L: ", w_L)
            joint_velocities = np.zeros(16)
            joint_velocities[0] = -w_L
            joint_velocities[2] = -w_L
            joint_velocities[1] = w_R
            joint_velocities[3] = w_R
            joint_velocities[4:10] = qd[2:]
            # current_velocity = np.array(my_robot.get_joint_velocities()).reshape(-1)
            # print("current_velocity: ", current_velocity)
            # print("changed current_velocity: ", joint_velocities)
            actions = ArticulationActions(
                joint_velocities=joint_velocities,
                joint_indices=aljnu_joint_indices
            )
            my_robot.apply_action(actions)

        # print("joints : ", my_robot.get_joint_positions())

simulation_app.close()