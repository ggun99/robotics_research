import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from matplotlib import pyplot as plt

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationActions


def find_indices_by_keywords(names, keywords):
    indices = []
    for index, name in enumerate(names):
        lower_name = str(name).lower()
        if all(keyword in lower_name for keyword in keywords):
            indices.append(index)
    return indices


def main():
    robot_usd_path = "/home/airlab/ros_workspace/src/aljnu_mobile_manipulator/aljnu_description/urdf/aljnu_mp/aljnu_mp.usd"
    # robot_usd_path = "/home/airlab/robotics_research/QP/isaacsim/aljnu_mp.usd"
        
    print(f"Using robot USD: {robot_usd_path}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane(z_position=-0.25)

    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/aljnu_mp")
    world.reset()

    robot = Articulation("/World/aljnu_mp")
    robot.initialize()

    joint_names = list(getattr(robot, "joint_names", []))
    body_names = list(getattr(robot, "body_names", []))
    joint_positions = robot.get_joint_positions()
    num_joints = int(joint_positions.shape[1]) if joint_positions.ndim == 2 else len(joint_names)

    if not joint_names:
        joint_names = [f"joint_{index}" for index in range(num_joints)]

    if not body_names:
        body_names = [f"body_{index}" for index in range(num_joints)]

    print("Joint names:")
    for index, name in enumerate(joint_names):
        print(f"  [{index}] {name}")

    print("Body names:")
    for index, name in enumerate(body_names):
        print(f"  [{index}] {name}")

    if len(joint_names) >= 4:
        wheel_joint_indices = find_indices_by_keywords(joint_names, ["wheel"])
    else:
        wheel_joint_indices = []

    if len(wheel_joint_indices) < 4:
        wheel_joint_indices = [0, 1, 2, 3]

    wheel_joint_indices = np.array(wheel_joint_indices[:4], dtype=int)
    print(f"Wheel joint indices: {wheel_joint_indices.tolist()}")

    start_positions, _ = robot.get_world_poses(indices=[0])
    start_position = np.array(start_positions[0], dtype=float)

    time_values = []
    x_values = []
    y_values = []
    z_values = []

    left_wheel_speed = 1.
    right_wheel_speed = 1.
    zero_debug_duration = 3.0

    if hasattr(robot, "switch_control_mode"):
        try:
            robot.switch_control_mode("velocity")
        except Exception as exc:
            print(f"Velocity mode switch failed: {exc}")

    print("Starting rolling test...")
    print("If the wheel spins but the base does not move, the USD likely lost collision, friction, or articulation setup.")

    max_sim_time = 300.0
    report_interval = 1.0
    next_report_time = 0.0

    while simulation_app.is_running() and world.current_time < max_sim_time:
        world.step(render=True)

        if not world.is_playing():
            continue

        joint_velocities = np.zeros(num_joints, dtype=float)

        if wheel_joint_indices.size >= 4:
            if world.current_time < zero_debug_duration:
                joint_velocities[:] = 0.0
            else:
                # The expected order in the current asset is:
                # front_left, front_right, rear_left, rear_right
                joint_velocities[wheel_joint_indices[0]] = -left_wheel_speed
                joint_velocities[wheel_joint_indices[1]] = right_wheel_speed
                joint_velocities[wheel_joint_indices[2]] = -left_wheel_speed
                joint_velocities[wheel_joint_indices[3]] = right_wheel_speed
        else:
            print("Not enough wheel joints were found to run the drive test.")
            break

        print(f"t={world.current_time:.2f}s commanded joint velocities = {joint_velocities.tolist()}")

        action = ArticulationActions(
            joint_velocities=joint_velocities,
            joint_indices=np.arange(num_joints),
        )
        robot.apply_action(action)

        base_pose, _ = robot.get_world_poses(indices=[0])
        base_position = np.array(base_pose[0], dtype=float)

        time_values.append(world.current_time)
        x_values.append(base_position[0])
        y_values.append(base_position[1])
        z_values.append(base_position[2])

        if world.current_time >= next_report_time:
            displacement = np.linalg.norm(base_position[:2] - start_position[:2])
            print(
                f"t={world.current_time:.2f}s, base=({base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}), "
                f"planar_displacement={displacement:.3f} m"
            )
            next_report_time += report_interval

        if np.linalg.norm(base_position[:2] - start_position[:2]) > 0.5:
            print("Base moved successfully.")
            break

    if time_values:
        output_dir = os.path.join(os.path.dirname(__file__), "rolling_test_results")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, label="Base path")
        plt.scatter([x_values[0]], [y_values[0]], label="Start", color="green")
        plt.scatter([x_values[-1]], [y_values[-1]], label="End", color="red")
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Aljnu mobile base rolling test")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = os.path.join(output_dir, "rolling_path.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved path plot: {plot_path}")

    if time_values:
        total_disp = np.linalg.norm(np.array([x_values[-1], y_values[-1]]) - np.array([x_values[0], y_values[0]]))
        print(f"Final planar displacement: {total_disp:.3f} m")
    else:
        print("No simulation data was collected.")


try:
    main()
finally:
    simulation_app.close()
