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
from isaacsim.core.prims import Articulation, XFormPrim

import isaacsim.core.utils.prims as prim_utils
from pxr import Gf
import numpy as np

# 월드에 빛을 추가합니다.
# prim_path는 빛이 생성될 USD 경로입니다.
# prim_type은 "DistantLight"로 지정합니다.
# position은 빛의 위치를 설정하지만, Directional Light의 경우 방향이 더 중요합니다.
# orientation은 빛의 방향을 설정합니다. (쿼터니언 형식)
# attributes는 빛의 속성을 설정합니다.
#   - "inputs:intensity": 빛의 강도
#   - "inputs:color": 빛의 색상 (RGB 튜플)

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


# from omni.isaac.core.utils.prims import is_prim_path_valid
# from omni.isaac.core.utils.prims import get_all_matching_child_prims

world = World(stage_units_in_meters=1.0)
scene = world.scene
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
robot_asset_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"#assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
# gripper_asset_path = assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
prim_path = "/World/aljnu_mp"
# 1. Stage에 USD 추가
add_reference_to_stage(usd_path=robot_asset_path, prim_path=prim_path)
aljnu_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # aljnu_mp의 조인트 인덱스
# articulation_controller = my_robot.get_articulation_controller()
# joint_names = my_robot.joint_names 
# print("Joint names: ", joint_names)
# 0 ~ 3 (mobile) : 'front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel', 
# 4 ~ 9 (UR5e) : 'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint', 'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint', 
# 10 ~ 15 (Gripper) : 'robotiq_85_left_inner_knuckle_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint'
world.scene.add_default_ground_plane(z_position=-0.2)  # 바닥면 추가
world.reset()

# 2. Articulation 객체로 래핑
my_robot = Articulation(prim_path)
my_robot.initialize()
# num_links = my_robot.num_bodies  # 예: 7 for UR5e
# values = np.ones((1, num_links), dtype=bool)  # 전부 중력 제거
# my_robot.set_body_disable_gravity(values, indices=[0], body_indices=aljnu_joint_indices[4:10])  # UR5e의 조인트 인덱스만 중력 제거
ur5e_indices = aljnu_joint_indices[4:]  # 조인트 6개라면
values = np.ones((1, len(ur5e_indices)), dtype=bool)  # 정확히 (1, 6)
my_robot.set_body_disable_gravity(values, indices=[0], body_indices=ur5e_indices)
print("aljnu_mp_robot is added")

# aljnu_mp.set_joints_default_state(positions=joints_default_positions)


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

i = 0
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
            # print("current_positions shape:", current_positions.shape)
            # print("target_joint_positions shape:", target_joint_positions.shape)
            # print("current_positions:", current_positions)
            # print("target_joint_positions:", target_joint_positions)
            # print("diff:", np.abs(current_positions[4:10] - target_joint_positions[4:10]))
            # 목표 자세 도달 여부 체크
            if np.all(np.abs(current_positions[4:10] - target_joint_positions[4:10]) < position_tolerance):
                reached_default = True
                
                print("Reached default position!")
        else:
            # 2단계: velocity 제어
            # i += 1
            # 엔드 이펙터의 변환 행렬 가져오기
            # end_effector_index = aljnu_joint_indices[-1]  # 엔드 이펙터의 조인트 인덱스 (마지막 조인트)
            # end_effector_pose = my_robot.get_world_poses(indices=[end_effector_index])
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
            # print("rear_right_wheel_pose: ", rear_right_wheel_pose)
            # print("rear_left_wheel_pose: ", rear_left_wheel_pose)
            # print("front_right_wheel_pose: ", front_right_wheel_pose)
            # print("front_left_wheel_pose: ", front_left_wheel_pose)
            base_link = "/World/aljnu_mp/base_link"
            prim_base_link = XFormPrim(base_link)
            base_pose, base_quat = prim_base_link.get_world_poses()
            # print("base_pose: ", base_pose)
            ur5e_wrist_3_link = "/World/aljnu_mp/ur5e_wrist_3_link"
            prim_wrist_3 = XFormPrim(ur5e_wrist_3_link)
            ur5e_wrist_3_pose, ur5e_wrist_3_quat = prim_wrist_3.get_world_poses()
            print("ur5e_wrist_3_pose: ", ur5e_wrist_3_pose)
            joint_velocities = np.zeros(16)
            # joint_velocities = np.zeros_like(current_positions)
            current_velocity = np.array(my_robot.get_joint_velocities()).reshape(-1)
            # print("current_velocity: ", current_velocity)
            # current_velocity[0][0] = 0.0  # 예시: 첫 번째 조인트
            # current_velocity[0][1] = 0.0 # 예시: 두 번째 조인트
            # current_velocity[0][2] = 0.0
            # current_velocity[0][3] = 0.0
            # current_velocity[0] = np.zeros_like(current_velocity)  # 모든 조인트 속도를 0으로 초기화
            # joint_velocities[9] = 0.5
            # print("changed current_velocity: ", joint_velocities)
            # my_robot.switch_control_mode("velocity")
            actions = ArticulationActions(
                joint_velocities=joint_velocities,
                joint_indices=aljnu_joint_indices
            )
            # my_robot.apply_action(actions)

        # print("joints : ", my_robot.get_joint_positions())

simulation_app.close()