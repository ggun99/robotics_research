import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.stage import update_stage
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
# from isaacsim.cortex.framework.robot import CortexUr10
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

# from omni.isaac.core.utils.prims import is_prim_path_valid
# from omni.isaac.core.utils.prims import get_all_matching_child_prims
import numpy as np

world = World(stage_units_in_meters=1.0)
scene = world.scene
assets_root_path = get_assets_root_path()
print("Assets root path: ", assets_root_path)

# use Isaac Sim provided asset
robot_asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
# gripper_asset_path = assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"

add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/ur5e")
# add_reference_to_stage(usd_path=gripper_asset_path, prim_path="/World/ur5e/Gripper/Robotiq_2F_85edit")

# parent_path = "/World/ur5e/ur5e/wrist_3_link"  # UR5e의 wrist_3_link
# child_path = "/World/ur5e/Gripper/Robotiq_2F_85_edit/Robotiq_2F_85/base_link"  # 그리퍼의 베이스

my_ur5 = world.scene.add(SingleManipulator(
    prim_path="/World/ur5e",
    name="ur5e_robbot",
    end_effector_prim_path="/World/ur5e/Gripper",
    ))

print("my_ur5 is added")

# my_ur5.set_joints_default_state(positions=joints_default_positions)

world.scene.add_default_ground_plane()
world.reset()
joints_default_positions = my_ur5.get_joint_positions()
arm_joint_indices = np.array([0, 1, 2, 3, 4, 5])
# articulation_controller = my_ur5.set_joint_velocities()

i = 0
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
        i += 1

        # 예시: 2번, 3번 조인트에 속도 부여
        joint_velocities = np.zeros_like(joints_default_positions)
        joint_velocities[1] = 0.5  # 2번 조인트 속도(rad/s)
        joint_velocities[2] = np.sin(i/100)  # 3번 조인트 속도(rad/s)

        # arm_joint_indices에 해당하는 속도만 추출
        # arm_velocities = joint_velocities[arm_joint_indices]
        my_ur5.set_joint_velocities(velocities = joint_velocities, indices=arm_joint_indices)
        # actions = ArticulationAction(
        #     joint_velocities=arm_velocities,
        #     joint_indices=arm_joint_indices
        # )
        articulation_controller.apply_action(actions)
        print("joints : ", my_ur5.get_joint_positions())

simulation_app.close()