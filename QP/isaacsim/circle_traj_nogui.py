import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

# 파라미터 설정
w1 = 1. #{w1_val}
w2 = 0.5 #{w2_val}
w4 = 0.2 #{w4_val}
lambda_h_a_param = 1. #{lambda_h_a_val}
# combo_index = {combo_index_val}
# total_combos = {total_combos_val}
RESULTS_FOLDER = "aaa"

# print("Running Simulation " + str(combo_index) + "/" + str(total_combos))
print("Parameters: w1=" + str(w1) + ", w2=" + str(w2) + ", w4=" + str(w4) + ", lambda_h_a=" + str(lambda_h_a_param))

# 시뮬레이션 앱 시작
simulation_app = SimulationApp({"headless": True})

try:
    # 모든 import 구문들
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

    # 함수 정의들
    def calculate_natural_rotation(T_cur, target_position):
        cur_z_axis = T_cur[:3, 2]
        current_position = T_cur[:3, 3]
        direction_vector = target_position - current_position
        direction_vector /= np.linalg.norm(direction_vector)
        
        new_z_axis = direction_vector
        new_y_axis = np.cross(cur_z_axis, new_z_axis)
        if np.linalg.norm(new_y_axis) < 1e-6:
            new_y_axis = np.array([0, 1, 0])
        new_y_axis /= np.linalg.norm(new_y_axis)
        
        new_x_axis = np.cross(new_y_axis, new_z_axis)
        new_x_axis /= np.linalg.norm(new_x_axis)
        
        rotation_matrix = np.vstack([new_x_axis, new_y_axis, new_z_axis]).T
        return rotation_matrix

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

    # 궤적 생성
    planner = CircularTrajectoryPlanner(radius=3.0, center=(0, 0, 0), height=1.0)
    traj_num_points = 100  # 테스트용으로 줄임
    human_smoothed_path = planner.generate_circle_path(num_points=traj_num_points)
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

    # 메인 시뮬레이션 루프
    max_iterations = 5000  # 테스트용으로 줄임
    iteration_count = 0
    
    print("Starting simulation loop...")
    
    while simulation_app.is_running() and iteration_count < max_iterations:
        iteration_count += 1
        world.step(render=False)
        
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
                distance_threshold = 0.15

                if distance_to_target < distance_threshold:
                    human_sphere.set_world_pose(robot_target_position)
                    human_trajectory_index += 1

                # 제어 로직
                cur_ee = T_e[:3,3]
                sight_vec = T_e[:3,0]
                sight_vec /= np.linalg.norm(sight_vec)

                rotation_matrix = calculate_natural_rotation(T_cur, robot_target_position)

                T_sd = np.eye(4)
                T_sd[:3, :3] = rotation_matrix
                T_sd[0, 3] = robot_target_position[0]
                T_sd[1, 3] = robot_target_position[1]
                T_sd[2, 3] = robot_target_position[2]
                desired_sphere.set_world_pose(T_sd[:3, 3])

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
                    J_mb_w = J_mb[3:, :]
                except Exception as e:
                    print("Jacobian calculation error: " + str(e))
                    continue

                T_error = np.linalg.inv(H_current.A) @ H_desired.A
                et = np.sum(np.abs(T_error[:3, -1]))

                # QP 설정
                Q = np.eye(n_dof + 6)
                Q[:2, :2] *= 1.0 / max(et * 100, 1e-6)
                Q[n_dof:, n_dof:] = (1. / max(et, 1e-6)) * np.eye(6)

                # 비용 함수 항들 계산
                C1 = np.concatenate((np.zeros(2), np.zeros(n_dof - 2), np.zeros(6)))
                
                try:
                    bTe = ur5e_robot.fkine(q[2:], include_base=False).A
                    θε = atan2(bTe[1, -1], bTe[0, -1])
                    C2 = np.zeros(n_dof + 6)
                    C2[0] = -5. * θε
                except:
                    C2 = np.zeros(n_dof + 6)

                # 장애물 회피 (간단화)
                C3 = np.zeros(n_dof + 6)

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
                w3 = 0.
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
                k[:3,:] *= 4.0
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
                        J_ @ x_ == v,
                    ]

                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.ECOS, verbose=False)

                    if x_.value is not None:
                        qd = x_.value
                    else:
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
    if len(scatter_x) > 0:
        filename = os.path.join(RESULTS_FOLDER, "simulation_w1_" + str(w1) + "_w2_" + str(w2) + "_w4_" + str(w4) + "_lambda_" + str(lambda_h_a_param) + ".png")
        
        # Figure 설정을 더 명확하게 지정
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # 축 범위를 먼저 설정
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        
        # 제목과 라벨 설정
        ax.set_title("Trajectory (w_m=" + str(w1) + ", w_o=" + str(w2) + ", w_h=" + str(w4) + ", λ=" + str(lambda_h_a_param) + ")")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        
        # 데이터 플롯
        ax.plot(human_trajectory["x"], human_trajectory["y"], 'b--', linewidth=2, label="Target Trajectory")
        ax.scatter(scatter_x, scatter_y, color='orange', s=10, label="End-Effector Trajectory", alpha=0.7)
        ax.scatter(scatter_mob_x, scatter_mob_y, color='cyan', s=10, label="Mobile Base Trajectory", alpha=0.7)
        
        # 범례와 그리드
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 종횡비를 고정하되 xlim, ylim 유지
        ax.set_aspect('equal', adjustable='box')
        
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