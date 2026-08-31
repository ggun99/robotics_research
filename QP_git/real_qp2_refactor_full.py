from __future__ import annotations

import os
import sys
import time
from math import atan2

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qpsolvers as qp
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseArray, Point, TransformStamped, Twist
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import rtde_control
import rtde_receive
import roboticstoolbox as rtb
from spatialmath import SE3, base
from butter import RealtimeButterworthFilter


class ControllerConfig:
    def __init__(self):
        self.n_dof = 8
        self.dt = 0.05
        self.d_safe = 0.1
        self.d_influence = 1.0
        self.obstacle_radius = 0.25
        self.lambda_max = 0.32
        self.lambda_h_a_param = 0.3
        self.target_reached_threshold = 0.2
        self.w1 = 1.0
        self.w2 = 1.0
        self.w3 = 0.0
        self.w4 = 0.2
        self.qdlim = np.array([0.15] * 8, dtype=float)
        self.qdlim[:1] = 0.05
        self.qdlim[1] = 0.05
        self.robot_ip = "192.160.0.4"


class TrajectoryRecorder:
    def __init__(self):
        self.base_trajectory = {"x": [], "y": [], "timestamps": []}
        self.ee_trajectory = {"x": [], "y": [], "timestamps": []}
        self.desired_trajectory = {"x": [], "y": [], "timestamps": []}
        self.last_desired_position = None
        self.start_time = time.time()
        self.obj_timestamps = []
        self.obj_total = []
        self.obj_Qterm = []
        self.obj_C1 = []
        self.obj_C2 = []
        self.obj_C3 = []
        self.obj_C4 = []
        self.problem_t = []
        self.entire_t = []

    def record_state(self, current_base_position, current_ee_position, T_sd, current_time):
        self.desired_trajectory["x"].append(T_sd[0, 3])
        self.desired_trajectory["y"].append(T_sd[1, 3])
        self.desired_trajectory["timestamps"].append(current_time)

        self.base_trajectory["x"].append(current_base_position[0])
        self.base_trajectory["y"].append(current_base_position[1])
        self.base_trajectory["timestamps"].append(current_time)

        self.ee_trajectory["x"].append(current_ee_position[0])
        self.ee_trajectory["y"].append(current_ee_position[1])
        self.ee_trajectory["timestamps"].append(current_time)

    def record_objective(self, qd, Q, C1, C2, C3, C4):
        try:
            t_now = time.time() - self.start_time
            x_vec = np.array(qd).reshape(-1)
            Qterm = 0.5 * float(x_vec.T @ Q @ x_vec)
            c1x = float(C1 @ x_vec) if C1 is not None else 0.0
            c2x = float(C2 @ x_vec) if C2 is not None else 0.0
            c3x = float(C3 @ x_vec) if C3 is not None else 0.0
            c4x = float(C4 @ x_vec) if C4 is not None else 0.0
            total_obj = Qterm + c1x + c2x + c3x + c4x

            self.obj_timestamps.append(t_now)
            self.obj_Qterm.append(Qterm)
            self.obj_C1.append(c1x)
            self.obj_C2.append(c2x)
            self.obj_C3.append(c3x)
            self.obj_C4.append(c4x)
            self.obj_total.append(total_obj)
        except Exception as e:
            print(f"Objective logging failed: {e}")

    def plot_objective_time_series(self, save_path=None, show_plot=False):
        if len(self.obj_timestamps) == 0:
            print("No objective history to plot.")
            return

        ts = np.array(self.obj_timestamps)
        c1 = np.array(self.obj_C1)
        c2 = np.array(self.obj_C2)
        c3 = np.array(self.obj_C3)
        c4 = np.array(self.obj_C4)
        qterm = np.array(self.obj_Qterm)
        total = np.array(self.obj_total)

        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(10, 6))
        contributions = np.vstack([c1, c2, c3, c4])
        labels = ["C1·x", "C2·x", "C3·x", "C4·x"]
        colors = ["#ffd700", "#ffa07a", "#87cefa", "#90ee90"]
        ax.stackplot(ts, contributions, labels=labels, colors=colors, alpha=0.8)
        ax.plot(ts, qterm, color="k", linestyle="--", linewidth=1.5, label="0.5 x^T Q x")
        ax.plot(ts, total, color="r", linestyle="-", linewidth=2, label="Total objective")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective value (scalar)")
        ax.set_title("QP objective and decomposition over time")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.expanduser(
                "/home/nvidia/geon/robotics/jetson_orin/controller/mb_control/hand_trajectories_plots/objective_time_series_"
                f"{int(time.time())}.png"
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Objective plot saved: {save_path}")

        if show_plot:
            try:
                from subprocess import Popen
                Popen(["xdg-open", save_path])
            except Exception:
                pass

    def save_ee_trajectory(self, obstacles_positions, obstacle_radius, w1, w2, w4, target_reached_threshold):
        from matplotlib.patches import Circle
        if len(self.ee_trajectory["x"]) == 0:
            print("📊 기록된 궤적이 없습니다.")
            return

        save_dir = "/home/nvidia/geon/robotics/jetson_orin/controller/mb_control/hand_trajectories_plots2"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        plt.switch_backend("Agg")
        plt.figure(figsize=(8, 8))
        plt.plot(self.ee_trajectory["x"], self.ee_trajectory["y"], "-", color="c", linewidth=2, alpha=0.7, label="Actual EE Trajectory")
        plt.plot(self.base_trajectory["x"], self.base_trajectory["y"], "-", color="orange", linewidth=2, alpha=0.7, label="Actual Mobile Base Trajectory")

        if len(self.desired_trajectory["x"]) > 0:
            plt.plot(self.desired_trajectory["x"], self.desired_trajectory["y"], "b--", linewidth=2, alpha=0.8, label="Desired Trajectory")
            plt.scatter(self.desired_trajectory["x"], self.desired_trajectory["y"], color="#87CEFA", s=80, marker="x", alpha=0.8, zorder=5, label="Desired Points")

        ox, oy = obstacles_positions[0, 0], obstacles_positions[0, 1]
        r = obstacle_radius
        circle = Circle((ox, oy), r, color="red", alpha=0.5)
        plt.gca().add_patch(circle)

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Actual and Desired XY Trajectory")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plot_filename_ = f"{save_dir}/trajectories_{w1}_{w2}_{w4}_{timestamp}.png"
        plt.savefig(plot_filename_, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.plot(self.ee_trajectory["x"], self.ee_trajectory["y"], "b-", linewidth=2, alpha=0.7, label="Actual EE Trajectory")
        if len(self.desired_trajectory["x"]) > 0:
            plt.plot(self.desired_trajectory["x"], self.desired_trajectory["y"], "r--", linewidth=2, alpha=0.8, label="Desired Trajectory")
            plt.scatter(self.desired_trajectory["x"], self.desired_trajectory["y"], color="red", s=80, marker="x", alpha=0.8, zorder=5, label="Desired Points")
            if len(self.desired_trajectory["x"]) > 0:
                plt.scatter(self.desired_trajectory["x"][0], self.desired_trajectory["y"][0], color="orange", s=120, marker="^", label="Desired Start", zorder=6)
                plt.scatter(self.desired_trajectory["x"][-1], self.desired_trajectory["y"][-1], color="darkred", s=120, marker="v", label="Desired End", zorder=6)
        plt.scatter(self.ee_trajectory["x"][0], self.ee_trajectory["y"][0], color="green", s=100, marker="o", label="Actual Start", zorder=5)
        plt.scatter(self.ee_trajectory["x"][-1], self.ee_trajectory["y"][-1], color="blue", s=100, marker="s", label="Actual End", zorder=5)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Actual and Desired XY Trajectory")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")

        plt.subplot(2, 3, 2)
        plt.plot(self.ee_trajectory["timestamps"], self.ee_trajectory["x"], "b-", linewidth=2, label="Actual X")
        if len(self.desired_trajectory["x"]) > 0:
            plt.plot(self.desired_trajectory["timestamps"], self.desired_trajectory["x"], "r--", linewidth=2, label="Desired X")
        plt.xlabel("Time (s)")
        plt.ylabel("X Position (m)")
        plt.title("X Position over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(self.ee_trajectory["timestamps"], self.ee_trajectory["y"], "b-", linewidth=2, label="Actual Y")
        if len(self.desired_trajectory["y"]) > 0:
            plt.plot(self.desired_trajectory["timestamps"], self.desired_trajectory["y"], "r--", linewidth=2, label="Desired Y")
        plt.xlabel("Time (s)")
        plt.ylabel("Y Position (m)")
        plt.title("Y Position over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 3, 4)
        if len(self.ee_trajectory["x"]) > 1:
            dx = np.diff(self.ee_trajectory["x"])
            dy = np.diff(self.ee_trajectory["y"])
            dt = np.diff(self.ee_trajectory["timestamps"])
            dt[dt == 0] = 1e-6
            vx = dx / dt
            vy = dy / dt
            v_magnitude = np.sqrt(vx ** 2 + vy ** 2)
            plt.plot(self.ee_trajectory["timestamps"][1:], v_magnitude, "purple", linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity Magnitude (m/s)")
            plt.title("EE Velocity Magnitude")
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        if len(self.desired_trajectory["x"]) > 0:
            if len(self.desired_trajectory["timestamps"]) > 1:
                ee_x = np.array(self.ee_trajectory["x"])
                ee_y = np.array(self.ee_trajectory["y"])
                ee_t = np.array(self.ee_trajectory["timestamps"])
                des_x = np.array(self.desired_trajectory["x"])
                des_y = np.array(self.desired_trajectory["y"])
                des_t = np.array(self.desired_trajectory["timestamps"])
                if len(des_t) > 1:
                    des_x_interp = np.interp(ee_t, des_t, des_x)
                    des_y_interp = np.interp(ee_t, des_t, des_y)
                    position_error = np.sqrt((ee_x - des_x_interp) ** 2 + (ee_y - des_y_interp) ** 2)
                    plt.plot(self.ee_trajectory["timestamps"], position_error, "purple", linewidth=2)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Position Error (m)")
                    plt.title("Tracking Error")
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, "Insufficient desired\ntrajectory data", ha="center", va="center", transform=plt.gca().transAxes)
                    plt.title("Tracking Error")
            else:
                plt.text(0.5, 0.5, "No desired trajectory\ndata available", ha="center", va="center", transform=plt.gca().transAxes)
                plt.title("Tracking Error")

        plt.subplot(2, 3, 6)
        if len(self.desired_trajectory["timestamps"]) > 1:
            waypoint_durations = np.diff(self.desired_trajectory["timestamps"])
            plt.bar(range(len(waypoint_durations)), waypoint_durations, color="orange", alpha=0.7)
            plt.xlabel("Waypoint Transition")
            plt.ylabel("Duration (s)")
            plt.title("Time Between Waypoints")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Insufficient waypoint\ndata for analysis", ha="center", va="center", transform=plt.gca().transAxes)
            plt.title("Time Between Waypoints")

        plt.tight_layout()
        plot_filename = f"{save_dir}/ee_trajectory_{w1},{w2},{w4}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        csv_filename = f"{save_dir}/ee_trajectory_{w1},{w2},{w4}_{timestamp}.csv"
        with open(csv_filename, "w") as f:
            f.write("timestamp,actual_x,actual_y\n")
            for i in range(len(self.ee_trajectory["x"])):
                f.write(
                    f"{self.ee_trajectory['timestamps'][i]:.3f},"
                    f"{self.ee_trajectory['x'][i]:.6f},"
                    f"{self.ee_trajectory['y'][i]:.6f}\n"
                )

        desired_csv_filename = f"{save_dir}/desired_trajectory_{w1},{w2},{w4}_{timestamp}.csv"
        if len(self.desired_trajectory["x"]) > 0:
            with open(desired_csv_filename, "w") as f:
                f.write("timestamp,desired_x,desired_y\n")
                for i in range(len(self.desired_trajectory["x"])):
                    f.write(
                        f"{self.desired_trajectory['timestamps'][i]:.3f},"
                        f"{self.desired_trajectory['x'][i]:.6f},"
                        f"{self.desired_trajectory['y'][i]:.6f}\n"
                    )

        total_distance = 0
        if len(self.ee_trajectory["x"]) > 1:
            for i in range(1, len(self.ee_trajectory["x"])):
                dx = self.ee_trajectory["x"][i] - self.ee_trajectory["x"][i - 1]
                dy = self.ee_trajectory["y"][i] - self.ee_trajectory["y"][i - 1]
                total_distance += np.sqrt(dx ** 2 + dy ** 2)

        x_range = max(self.ee_trajectory["x"]) - min(self.ee_trajectory["x"])
        y_range = max(self.ee_trajectory["y"]) - min(self.ee_trajectory["y"])
        total_time = self.ee_trajectory["timestamps"][-1] - self.ee_trajectory["timestamps"][0]

        avg_tracking_error = 0
        max_tracking_error = 0
        num_waypoints = len(self.desired_trajectory["x"])

        if len(self.desired_trajectory["x"]) > 1:
            try:
                import scipy.interpolate
                f_x = scipy.interpolate.interp1d(self.desired_trajectory["timestamps"], self.desired_trajectory["x"], kind="linear", fill_value="extrapolate")
                f_y = scipy.interpolate.interp1d(self.desired_trajectory["timestamps"], self.desired_trajectory["y"], kind="linear", fill_value="extrapolate")
                desired_x_interp = f_x(self.ee_trajectory["timestamps"])
                desired_y_interp = f_y(self.ee_trajectory["timestamps"])
                tracking_errors = np.sqrt((np.array(self.ee_trajectory["x"]) - desired_x_interp) ** 2 + (np.array(self.ee_trajectory["y"]) - desired_y_interp) ** 2)
                avg_tracking_error = np.mean(tracking_errors)
                max_tracking_error = np.max(tracking_errors)
            except Exception:
                pass

        stats_filename = f"{save_dir}/trajectory_stats_{timestamp}.txt"
        stats_info = f"""궤적 추종 성능 통계 정보
=====================================
기록 시간: {timestamp}
제어 파라미터: w1={w1}, w2={w2}, w4={w4}

=== Actual Trajectory ===
총 이동 거리: {total_distance:.3f} m
X 축 범위: {x_range:.3f} m
Y 축 범위: {y_range:.3f} m
총 시간: {total_time:.1f} s
평균 속도: {total_distance/max(total_time, 1e-6):.3f} m/s
기록된 포인트 수: {len(self.ee_trajectory['x'])} 개

=== Desired Trajectory ===
웨이포인트 수: {num_waypoints} 개
첫 번째 목표: ({self.desired_trajectory['x'][0]:.3f}, {self.desired_trajectory['y'][0]:.3f}) (시작 시 0초) if num_waypoints > 0 else (없음)
마지막 목표: ({self.desired_trajectory['x'][-1]:.3f}, {self.desired_trajectory['y'][-1]:.3f}) (시간 {self.desired_trajectory['timestamps'][-1]:.1f}초) if num_waypoints > 0 else (없음)

=== 추종 성능 ===
평균 추적 오차: {avg_tracking_error:.3f} m
최대 추적 오차: {max_tracking_error:.3f} m
목표 도달 임계값: {target_reached_threshold:.3f} m

=== QP 속도 성능 ===
평균 풀이 시간: {sum(self.problem_t)/len(self.problem_t)} s
전체 과정 시간: {sum(self.entire_t)/len(self.entire_t)}
"""

        with open(stats_filename, "w") as f:
            f.write(stats_info)

        print(f"\n📊 궤적 추종 결과 저장 완료:")
        print(f"   그래프: {plot_filename}")
        print(f"   Actual 데이터: {csv_filename}")
        if len(self.desired_trajectory["x"]) > 0:
            print(f"   Desired 데이터: {desired_csv_filename}")
        print(f"   통계: {stats_filename}")
        print(f"   총 이동 거리: {total_distance:.3f} m")
        print(f"   웨이포인트 수: {num_waypoints} 개")
        print(f"   평균 추적 오차: {avg_tracking_error:.3f} m")
        print(f"   최대 추적 오차: {max_tracking_error:.3f} m")
        print(f"   총 시간: {total_time:.1f} s")


class RobotKinematics:
    def __init__(self, ur5e_robot):
        self.ur5e_robot = ur5e_robot
        self.define_robot_collision_points()
        self.ur5e_frame_correction = np.eye(4)
        self.ur5e_frame_correction[:3, :3] = R.from_euler("z", np.pi).as_matrix()

    def define_robot_collision_points(self):
        self.ur5e_link_points = {
            "base": np.array([
                [0.0, 0.0, 0.0], [0.08, 0.08, 0.05], [-0.08, 0.08, 0.05], [0.08, -0.08, 0.05], [-0.08, -0.08, 0.05], [0.06, 0.0, 0.1], [-0.06, 0.0, 0.1], [0.0, 0.06, 0.1], [0.0, -0.06, 0.1]
            ]),
            "shoulder": np.array([
                [0.0, 0.0, 0.0], [0.06, 0.06, 0.08], [-0.06, 0.06, 0.08], [0.06, -0.06, 0.08], [-0.06, -0.06, 0.08], [0.0, 0.0, 0.12]
            ]),
            "upper_arm": np.array([
                [0.0, 0.0, 0.0], [0.05, 0.0, 0.1], [-0.05, 0.0, 0.1], [0.0, 0.05, 0.1], [0.0, -0.05, 0.1], [0.04, 0.0, 0.2], [-0.04, 0.0, 0.2], [0.0, 0.04, 0.25], [0.0, -0.04, 0.25]
            ]),
            "forearm": np.array([
                [0.4, 0.0, 0.0], [0.04, 0.0, 0.08], [-0.04, 0.0, 0.08], [0.0, 0.04, 0.08], [0.0, -0.04, 0.08], [0.03, 0.0, 0.15], [-0.03, 0.0, 0.15], [0.0, 0.03, 0.17], [0.0, -0.03, 0.17]
            ]),
            "wrist_1": np.array([
                [0.0, 0.0, 0.0], [0.03, 0.03, 0.0], [-0.03, 0.03, 0.0], [0.03, -0.03, 0.0], [-0.03, -0.03, 0.0]
            ]),
            "wrist_2": np.array([
                [0.0, 0.0, 0.0], [0.03, 0.0, 0.03], [-0.03, 0.0, 0.03], [0.0, 0.03, 0.03], [0.0, -0.03, 0.03]
            ]),
            "wrist_3": np.array([
                [0.0, 0.0, 0.0], [0.025, 0.025, 0.02], [-0.025, 0.025, 0.02], [0.025, -0.025, 0.02], [-0.025, -0.025, 0.02]
            ]),
        }

        self.mobile_base_points = np.array([
            [0.35, 0.25, 0.1], [0.25, 0.25, 0.1], [0.15, 0.25, 0.1], [0.05, 0.25, 0.1], [-0.05, 0.25, 0.1], [-0.15, 0.25, 0.1], [-0.25, 0.25, 0.1], [-0.35, 0.25, 0.1],
            [0.35, -0.25, 0.1], [0.25, -0.25, 0.1], [0.15, -0.25, 0.1], [0.05, -0.25, 0.1], [-0.05, -0.25, 0.1], [-0.15, -0.25, 0.1], [-0.25, -0.25, 0.1], [-0.35, -0.25, 0.1],
            [0.35, 0.15, 0.1], [0.35, 0.05, 0.1], [0.35, -0.05, 0.1], [0.35, -0.15, 0.1], [-0.35, 0.15, 0.1], [-0.35, 0.05, 0.1], [-0.35, -0.05, 0.1], [-0.35, -0.15, 0.1],
            [0.3, 0.2, 0.3], [-0.3, 0.2, 0.3], [0.3, -0.2, 0.3], [-0.3, -0.2, 0.3]
        ])

    def build_base_transform(self, base_position, base_quaternion):
        T_sb = np.eye(4)
        T_sb[0, 3] = base_position[0]
        T_sb[1, 3] = base_position[1]
        T_sb[2, 3] = base_position[2]
        T_sb[:3, :3] = R.from_quat(base_quaternion).as_matrix()
        return T_sb

    def _dh_transform(self, a, alpha, d, theta):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)
        T = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        return T

    def ur5e_fk_chain_manual(self, q_arm):
        a = [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0]
        alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
        d = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]
        q = np.asarray(q_arm, dtype=float).reshape(-1)
        q_pad = np.zeros(6, dtype=float)
        q_pad[:q.size] = q

        transforms = [np.eye(4)]
        T_acc = np.eye(4)
        for i in range(6):
            T_link = self._dh_transform(a[i], alpha[i], d[i], q_pad[i])
            T_acc = T_acc @ T_link
            transforms.append(T_acc.copy())
            if i + 1 >= q.size:
                break
        return transforms

    def compute_dynamic_robot_collision_points(self, q, base_position, base_quaternion):
        if base_position is None or base_quaternion is None:
            return np.empty((0, 3), dtype=float), 0, 0

        collision_points_world = []
        T_sb = self.build_base_transform(base_position, base_quaternion)
        T_b0 = np.eye(4)
        T_b0[0, 3] = 0.1315
        T_b0[2, 3] = 0.51921

        for point in self.mobile_base_points:
            point_homogeneous = np.append(point, 1)
            point_world = T_sb @ point_homogeneous
            collision_points_world.append(point_world[:3])
        num_mobile = len(collision_points_world)

        link_names = ["base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
        num_mani = 0
        fk_chain = self.ur5e_fk_chain_manual(q[2:])
        for i, link_name in enumerate(link_names):
            if i < len(fk_chain):
                T_0i = fk_chain[i]
            else:
                T_0i = fk_chain[-1] if len(fk_chain) > 0 else np.eye(4)
            T_si = T_sb @ T_b0 @ self.ur5e_frame_correction @ T_0i
            if link_name in self.ur5e_link_points:
                for point in self.ur5e_link_points[link_name]:
                    point_homogeneous = np.append(point, 1)
                    point_world = T_si @ point_homogeneous
                    collision_points_world.append(point_world[:3])
                    num_mani += 1

        return np.asarray(collision_points_world, dtype=float), num_mobile, num_mani

    def get_nearest_obstacle_distance(self, position, obstacles, obstacle_radius):
        distances = []
        obs_real = []
        for obs in obstacles:
            ox, oy = float(obs[0]), float(obs[1])
            obs_vis = np.array([ox, oy, float(position[2])])
            obs_real.append(obs_vis)
            dx = position[0] - ox
            dy = position[1] - oy
            distances.append(np.hypot(dx, dy) - obstacle_radius)

        distances = np.asarray(distances, dtype=float)
        index = int(np.argmin(distances))
        dx = position[0] - obs_real[index][0]
        dy = position[1] - obs_real[index][1]
        vec2d = np.array([dx, dy, 0.0], dtype=float)
        norm = np.linalg.norm(vec2d)
        if norm == 0.0:
            g_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            g_vec = vec2d / norm
        return distances, index, g_vec


class QPController:
    def __init__(self, config):
        self.config = config

    def joint_velocity_damper(self, q, qlim, ps=0.05, pi=0.1, n=8, gain=1.0):
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

    def calculate_natural_rotation(self, T_cur, target_position):
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

    def solve_qp(self, state, kin, T_cur, T_be, T_sd, ur5e_robot, q, human_position, base_position, base_quaternion, obstacles_positions):
        n_dof = self.config.n_dof

        # 1) compute qd and robot state for solver
        if base_position is None or base_quaternion is None:
            return np.zeros(n_dof + 6)

        robot_collision_check, num_mobile, num_mani = kin.compute_dynamic_robot_collision_points(q, base_position, base_quaternion)
        xform_pose = [np.array(p) for p in robot_collision_check]

        num_cable = len(state.cable_points) if state.cable_points is not None else 0
        if num_cable > 0:
            for p in state.cable_points:
                xform_pose.append(np.array(p))

        T_sb = np.eye(4)
        T_sb[0, 3] = base_position[0]
        T_sb[1, 3] = base_position[1]
        T_sb[2, 3] = base_position[2]
        T_sb[:3, :3] = R.from_quat(base_quaternion).as_matrix()
        T_b0 = np.eye(4)
        T_b0[0, 3] = 0.1315
        T_b0[2, 3] = 0.51921

        T_0e = ur5e_robot.fkine(q[2:]).A
        T = T_b0 @ T_0e
        T_be = T
        H_current = SE3(T_be)

        xform_pose = np.array(xform_pose) if len(xform_pose) > 0 else np.array([]).reshape(0, 3)
        T_cur = T_sb @ T_be
        current_ee_position = T_cur[:3, 3]

        if human_position is None:
            return np.zeros(n_dof + 6)

        if not hasattr(self, "ee_rotation"):
            self.rotation_matrix = T_cur[:3, :3]
            self.ee_rotation = True

        rotation_matrix = self.rotation_matrix
        det = np.linalg.det(rotation_matrix)
        orthogonality_check = np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3))
        if not np.isclose(det, 1.0) or not orthogonality_check:
            U, _, Vt = np.linalg.svd(rotation_matrix)
            rotation_matrix = U @ Vt

        robot_target_position_homogeneous = np.append(human_position, 1)
        robot_target_position_local = T_cur @ robot_target_position_homogeneous
        robot_target_position_ee = robot_target_position_local[:3]
        T_sd = np.eye(4)
        T_sd[:3, :3] = rotation_matrix
        T_sd[0, 3] = (robot_target_position_ee[0] + T_cur[0, 3]) / 2
        T_sd[1, 3] = (robot_target_position_ee[1] + T_cur[1, 3]) / 2
        T_sd[2, 3] = (robot_target_position_ee[2] + T_cur[2, 3]) / 2

        T_cur_l = T_cur
        sight_vec = T_cur_l[:3, 0]
        sight_vec /= np.linalg.norm(sight_vec)
        direction_unit_vector = np.array(human_position) - T_cur_l[:3, 3]
        direction_unit_vector = direction_unit_vector / np.linalg.norm(direction_unit_vector)
        cos_theta = np.dot(direction_unit_vector, sight_vec)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        T_bd = np.linalg.inv(T_sb) @ T_sd
        H_desired = SE3(T_bd)
        F = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        J_p = base.tr2adjoint(T_be.T) @ F
        J_a_e = base.tr2adjoint(T_be.T) @ ur5e_robot.jacob0(q[2:])
        J_mb = np.hstack((J_p, J_a_e))
        J_mb_v = J_mb[:3, :]
        J_mb_w = J_mb[3:, :]

        T_error = np.linalg.inv(H_current.A) @ H_desired.A
        et = np.sum(np.abs(T_error[:3, -1]))

        Q = np.eye(self.config.n_dof + 6)
        Q[:2, :2] *= 1.0 / max(et * 100, 1e-6)
        Q[n_dof:, n_dof:] = (1.0 / max(et, 1e-6)) * np.eye(6)

        H = np.zeros((self.config.n_dof - 2, 6, self.config.n_dof - 2))
        for j in range(self.config.n_dof - 2):
            for i in range(j, self.config.n_dof - 2):
                H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                if i != j:
                    H[i, :3, j] = H[j, :3, i]
                    H[i, 3:, j] = H[j, 3:, i]

        J_a = ur5e_robot.jacob0(q[2:])
        m = J_a @ J_a.T
        m_det = np.linalg.det(m)
        m_t = np.sqrt(m_det)
        rank = np.linalg.matrix_rank(J_a @ J_a.T)
        if rank < J_a.shape[0]:
            JJ_inv = np.linalg.pinv(J_a @ J_a.T)
        else:
            JJ_inv = np.linalg.inv(J_a @ J_a.T)

        J_m = np.zeros((self.config.n_dof - 2, 1))
        for i in range(self.config.n_dof - 2):
            c = J_a @ np.transpose(H[i, :, :])
            J_m[i, 0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

        A = np.zeros((len(xform_pose), self.config.n_dof + 6))
        B = np.zeros(len(xform_pose))
        J_dj = np.zeros(self.config.n_dof + 6)
        w_p_sum = 0.0
        min_dist_list = []
        g_vec_list = []
        weighted_g_vec = np.zeros(3)
        total_weight = 0.0

        for i, pose in enumerate(xform_pose):
            distance, index, g_vec_ = kin.get_nearest_obstacle_distance(pose, obstacles_positions[:, :3], self.config.obstacle_radius)
            min_dist_ = np.min(distance)
            min_dist_list.append(min_dist_)
            g_vec_list.append(g_vec_)
            if min_dist_ <= 0.6:
                weight = 0.15
            else:
                weight = 0.0
            weighted_g_vec += weight * g_vec_
            total_weight += weight

        if total_weight > 0:
            g_vec = weighted_g_vec / total_weight
        else:
            g_vec = np.zeros(3)
        min_dist = np.min(min_dist_list)
        min_index = np.argmin(min_dist_list)
        g_vec = g_vec_list[min_index]
        g_vec /= np.linalg.norm(g_vec)

        link1_len = (num_mobile + len(kin.ur5e_link_points["shoulder"]))
        link2_len = (link1_len + len(kin.ur5e_link_points["upper_arm"]))
        link3_len = (link2_len + len(kin.ur5e_link_points["forearm"]))
        link4_len = (link3_len + len(kin.ur5e_link_points["wrist_1"]))
        link5_len = (link4_len + len(kin.ur5e_link_points["wrist_2"]))

        for i, pose in enumerate(xform_pose):
            if i < num_mobile:
                d_dot = g_vec @ J_mb_v
                A[i, :8] = d_dot
                A[i, 8:] = np.zeros((1, 6))
                B[i] = (min_dist_list[i] - self.config.d_safe) / ((self.config.d_influence - self.config.d_safe))
                w_p = (self.config.d_influence - min_dist_list[i]) / (self.config.d_influence - self.config.d_safe)
                J_dj[:8] += (d_dot) * (w_p)
                w_p_sum += np.abs(w_p)
            else:
                if num_mobile <= i < link1_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 2)), J_a_e[:3, :]])
                elif link1_len <= i < link2_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 3)), J_a_e[:3, 1:]])
                elif link2_len <= i < link3_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 4)), J_a_e[:3, 2:]])
                elif link3_len <= i < link4_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 5)), J_a_e[:3, 3:]])
                elif link4_len <= i < link5_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 6)), J_a_e[:3, 4:]])
                else:
                    J_mb_arm_v = np.hstack([np.zeros((3, 7)), J_a_e[:3, 5:]])

                d_dot = g_vec @ J_mb_arm_v
                A[i, :8] = d_dot
                A[i, 8:] = np.zeros((1, 6))
                B[i] = (min_dist_list[i] - self.config.d_safe) / ((self.config.d_influence - self.config.d_safe))
                w_p = (self.config.d_influence - min_dist_list[i]) / (self.config.d_influence - self.config.d_safe)
                J_dj[:8] += (d_dot) * (w_p)
                w_p_sum += np.abs(w_p)

        C1 = np.concatenate((np.zeros(2), -J_m.reshape((self.config.n_dof - 2,)), np.zeros(6)))
        try:
            bTe = ur5e_robot.fkine(q[2:], include_base=False).A
            theta_e = atan2(bTe[1, -1], bTe[0, -1])
            C2 = np.zeros(self.config.n_dof + 6)
            C2[0] = -5. * theta_e
        except Exception:
            C2 = np.zeros(self.config.n_dof + 6)

        lambda_max = 5.
        min_distance = np.min(min_dist_list) if len(min_dist_list) > 0 else 1e9
        if min_distance <= self.config.d_influence:
            lambda_c = (lambda_max / (self.config.d_influence - self.config.d_safe) ** 2) * (min_distance - self.config.d_influence) ** 2
        else:
            lambda_c = 0.0
        J_c = lambda_c * J_dj / w_p_sum if w_p_sum != 0 else np.zeros_like(J_dj)
        C3 = J_c

        J_h = np.zeros(self.config.n_dof + 6)
        J_mb_w_h = direction_unit_vector @ J_mb_w
        epsilon = 1e-6
        lambda_h = self.config.lambda_h_a_param * max(abs(theta), epsilon)
        J_h[:8] = lambda_h * J_mb_w_h
        C4 = J_h

        C = self.config.w1 * C1 + self.config.w2 * C2 + self.config.w3 * C3 + self.config.w4 * C4
        J_ = np.c_[J_mb, np.eye(6)]
        eTep = T_error
        e = np.zeros(6)
        e[:3] = eTep[:3, -1]
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        k = np.eye(6)
        v = k @ e

        lb = -np.r_[self.config.qdlim[: self.config.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.config.qdlim[: self.config.n_dof], 10 * np.ones(6)]

        x_ = cp.Variable(self.config.n_dof + 6)
        objective = cp.Minimize(0.5 * cp.quad_form(x_, Q) + C.T @ x_)
        constraints = [x_ >= lb, x_ <= ub, J_ @ x_ == v, A @ x_ <= B]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if x_.value is not None:
            qd = np.asarray(x_.value, dtype=float)
        else:
            qd = np.zeros(self.config.n_dof + 6)

        if et < self.config.target_reached_threshold:
            qd = np.zeros(self.config.n_dof + 6)

        return qd, Q, C1, C2, C3, C4, T_sd, current_ee_position


class QP_mbcontorller(Node):
    def __init__(self):
        super().__init__("mbcontroller")
        self.config = ControllerConfig()
        self.ur5e_robot = rtb.models.UR5()
        self.kinematics = RobotKinematics(self.ur5e_robot)
        self.qp = QPController(self.config)
        self.tracker = TrajectoryRecorder()

        self.ROBOT_IP = self.config.robot_ip
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            print(f"✅ UR 로봇 연결 성공 (IP: {self.ROBOT_IP})")
        except Exception as e:
            print(f"❌ UR 로봇 연결 실패: {e}")
            sys.exit(1)

        self.n_dof = self.config.n_dof
        self.base_position = None
        self.base_quaternion = None
        self.human_position = None
        self.cable_points = []
        self.obstacles_positions = np.array([[0.6, 0.0, 0.5]])
        self.current_joint_positions = None
        self.q = None
        self.qdlim = self.config.qdlim
        self.qlim = np.array([
            [-np.inf, -np.inf, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
            [np.inf, np.inf, 3.14159265, 3.14159265, 3.14159265, 3.14159265, 3.14159265, 3.14159265],
        ])
        self.H_desired = None
        self.current_waypoint_id = 0
        self.last_target_position = None
        self.target_reached = False
        self.target_reached_time = None
        self.ee_rotation = False
        self.rotation_matrix = np.eye(3)

        self.base_position_sub = self.create_subscription(Pose, "/mobile_base/pose", self.set_base_position, 10)
        self.cable_position_sub = self.create_subscription(PoseArray, "/cable_points", self.set_cable_positions, 10)
        self.human_position_sub = self.create_subscription(Pose, "/hand_pose", self.set_human_position, 10)

        self.scout_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/robot_collision_markers", 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.g_vec_pub = self.create_publisher(Marker, "g_vec_marker", 10)

        self.butter_x = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_y = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_z = RealtimeButterworthFilter(order=1, cutoff=3.0, fs=30.0)
        self.butter_hand_x = RealtimeButterworthFilter(order=1, cutoff=5.0, fs=30.0)
        self.butter_hand_y = RealtimeButterworthFilter(order=1, cutoff=5.0, fs=30.0)
        self.butter_hand_z = RealtimeButterworthFilter(order=1, cutoff=5.0, fs=30.0)
        self.cable_filter = [
            RealtimeButterworthFilter(order=1, cutoff=5.0, fs=30.0),
            RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0),
            RealtimeButterworthFilter(order=1, cutoff=10.0, fs=30.0),
        ]

        self.create_timer(self.config.dt, self.QP_real)
        self.create_timer(0.1, self.publish_collision_markers)

        self.target_reached_threshold = self.config.target_reached_threshold
        self.target_reached_debounce_time = 0.5
        self.last_target_position = None
        self._g_vec_ns = "g_vec"
        self._g_vec_id = 0

        print("🎯 사람 추종 시스템 초기화 완료")

    def define_robot_collision_points(self):
        self.kinematics.define_robot_collision_points()

    def trajectory_completed_callback(self, msg):
        pass

    def current_waypoint_callback(self, msg):
        prev_waypoint_id = self.current_waypoint_id
        self.current_waypoint_id = msg.data
        if prev_waypoint_id != self.current_waypoint_id:
            self.target_reached = False
            self.target_reached_time = None
            print(f"📍 새 웨이포인트 ID: {self.current_waypoint_id} (상태 리셋)")

    def check_target_reached(self, current_pos, target_pos):
        if target_pos is None or current_pos is None:
            return False
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        self.get_logger().info(f"Distance to target: {distance:.3f} m")
        return distance < self.target_reached_threshold

    def publish_collision_markers(self):
        try:
            pts_world, num_mobile, num_mani = self.kinematics.compute_dynamic_robot_collision_points(self.q, self.base_position, self.base_quaternion) if self.q is not None else (np.empty((0, 3), dtype=float), 0, 0)
            if pts_world is None or len(pts_world) == 0:
                return
            ma = MarkerArray()
            now = self.get_clock().now().to_msg()
            lifetime = Duration(sec=0, nanosec=200_000_000)
            scale = 0.05
            for i in range(num_mobile):
                p = pts_world[i]
                m = Marker()
                m.header.stamp = now
                m.header.frame_id = "world"
                m.ns = "mobile_base_points"
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(p[0])
                m.pose.position.y = float(p[1])
                m.pose.position.z = float(p[2])
                m.pose.orientation.w = 1.0
                m.scale.x = scale
                m.scale.y = scale
                m.scale.z = scale
                m.color.r = 0.0
                m.color.g = 0.2
                m.color.b = 1.0
                m.color.a = 0.9
                m.lifetime = lifetime
                ma.markers.append(m)
            for j, idx in enumerate(range(num_mobile, len(pts_world))):
                p = pts_world[idx]
                m = Marker()
                m.header.stamp = now
                m.header.frame_id = "world"
                m.ns = "manipulator_points"
                m.id = j
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(p[0])
                m.pose.position.y = float(p[1])
                m.pose.position.z = float(p[2])
                m.pose.orientation.w = 1.0
                m.scale.x = scale * 0.9
                m.scale.y = scale * 0.9
                m.scale.z = scale * 0.9
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.2
                m.color.a = 0.9
                m.lifetime = lifetime
                ma.markers.append(m)
            self.marker_pub.publish(ma)
        except Exception as e:
            self.get_logger().warn(f"publish_collision_markers failed: {e}")

    def set_base_position(self, msg):
        buttered_x = self.butter_x.update(msg.position.x)
        buttered_y = self.butter_y.update(msg.position.y)
        buttered_z = self.butter_z.update(msg.position.z)

        H_world_aruco = np.eye(4)
        H_world_aruco[0, 3] = buttered_x
        H_world_aruco[1, 3] = buttered_y
        H_world_aruco[2, 3] = buttered_z
        H_world_aruco[:3, :3] = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
        H_aruco_base = np.eye(4)
        H_aruco_base[0, 3] = 0.015
        H_aruco_base[1, 3] = -0.16
        H_aruco_base[2, 3] = -0.51921
        H_aruco_base[:3, :3] = np.eye(3)
        H_world_base = H_world_aruco @ H_aruco_base

        self.base_position = [H_world_base[0, 3], H_world_base[1, 3], H_world_base[2, 3]]
        self.base_quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def set_human_position(self, msg):
        if self.base_position is None:
            return
        if msg is None:
            self.human_position = None
            return

        buttered_hand_x = self.butter_hand_x.update(msg.position.x)
        buttered_hand_y = self.butter_hand_y.update(msg.position.y)
        buttered_hand_z = self.butter_hand_z.update(msg.position.z)
        self.human_position = [buttered_hand_x, buttered_hand_y, buttered_hand_z]

    def set_cable_positions(self, msg):
        self.cable_points = []
        for pose in msg.poses:
            filtered = [
                f.update(v)
                for f, v in zip(
                    self.cable_filter,
                    [pose.position.x, pose.position.y, pose.position.z],
                )
            ]
            filtered = np.array(filtered, dtype=float)
            self.cable_points.append(filtered)

    def make_tf_msg(self, pos, quat, parent_name, child_frame_name):
        tfmsg = TransformStamped()
        tfmsg.header.stamp = self.get_clock().now().to_msg()
        tfmsg.header.frame_id = parent_name
        tfmsg.child_frame_id = child_frame_name
        tfmsg.transform.translation.x = pos[0]
        tfmsg.transform.translation.y = pos[1]
        tfmsg.transform.translation.z = pos[2]
        tfmsg.transform.rotation.x = quat[0]
        tfmsg.transform.rotation.y = quat[1]
        tfmsg.transform.rotation.z = quat[2]
        tfmsg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(tfmsg)

    def QP_real(self):
        entire_start_t = time.time()
        t_start = self.rtde_c.initPeriod()
        current_joint_positions = self.rtde_r.getActualQ()

        self.q = np.zeros(8)
        self.q[0] = 0.0
        self.q[1] = 0.0
        self.q[2:] = current_joint_positions

        if self.base_position is None or self.base_quaternion is None:
            return

        robot_collision_check, num_mobile, num_mani = self.kinematics.compute_dynamic_robot_collision_points(self.q, self.base_position, self.base_quaternion)
        xform_pose = [np.array(p) for p in robot_collision_check]
        num_cable = len(self.cable_points)
        if num_cable > 0:
            for p in self.cable_points:
                xform_pose.append(np.array(p))

        T_sb = np.eye(4)
        T_sb[0, 3] = self.base_position[0]
        T_sb[1, 3] = self.base_position[1]
        T_sb[2, 3] = self.base_position[2]
        T_sb[:3, :3] = R.from_quat(self.base_quaternion).as_matrix()
        T_b0 = np.eye(4)
        T_b0[0, 3] = 0.1315
        T_b0[2, 3] = 0.51921

        pppp = list(T_sb[0:3, 3])
        qqqq = R.from_matrix(T_sb[0:3, 0:3]).as_quat()
        self.make_tf_msg(pppp, qqqq, "world", "base_world")

        ppose = list(T_b0[0:3, 3])
        qqua = R.from_matrix(T_b0[0:3, 0:3]).as_quat()
        self.make_tf_msg(ppose, qqua, "base_world", "base_0")

        T_0e = self.ur5e_robot.fkine(self.q[2:]).A
        T = T_b0 @ T_0e
        T_be = T
        pppose = list(T_be[0:3, 3])
        qqqua = R.from_matrix(T_be[0:3, 0:3]).as_quat()
        self.make_tf_msg(pppose, qqqua, "base_world", "ee_base")
        H_current = SE3(T_be)

        xform_pose = np.array(xform_pose) if len(xform_pose) > 0 else np.array([]).reshape(0, 3)
        T_cur = T_sb @ T_be
        self.T_cur = T_cur
        current_ee_position = T_cur[:3, 3]
        current_base_position = T_sb[:3, 3]
        current_time = time.time() - self.tracker.start_time

        ppp = list(T_cur[0:3, 3])
        qqq = R.from_matrix(T_cur[0:3, 0:3]).as_quat()
        self.make_tf_msg(ppp, qqq, "world", "ee")

        p_obs = [self.obstacles_positions[:, :3][0][0], self.obstacles_positions[:, :3][0][1], self.obstacles_positions[:, :3][0][2]]
        q_obs = R.from_matrix(T_cur[0:3, 0:3]).as_quat()
        self.make_tf_msg(p_obs, q_obs, "world", "obstacle")

        if self.human_position is None:
            print("No Desired Position")
            return

        if not self.ee_rotation:
            self.rotation_matrix = T_cur[:3, :3]
            self.ee_rotation = True

        T_sd = np.eye(4)
        T_sd[:3, :3] = self.rotation_matrix
        det = np.linalg.det(self.rotation_matrix)
        orthogonality_check = np.allclose(self.rotation_matrix.T @ self.rotation_matrix, np.eye(3))
        if not np.isclose(det, 1.0) or not orthogonality_check:
            U, _, Vt = np.linalg.svd(self.rotation_matrix)
            rotation_matrix_normalized = U @ Vt
            T_sd[:3, :3] = rotation_matrix_normalized

        robot_target_position_homogeneous = np.append(self.human_position, 1)
        robot_target_position_local = T_cur @ robot_target_position_homogeneous
        robot_target_position_ee = robot_target_position_local[:3]
        T_sd[0, 3] = (robot_target_position_ee[0] + T_cur[0, 3]) / 2
        T_sd[1, 3] = (robot_target_position_ee[1] + T_cur[1, 3]) / 2
        T_sd[2, 3] = (robot_target_position_ee[2] + T_cur[2, 3]) / 2

        self.tracker.record_state(current_base_position, current_ee_position, T_sd, current_time)

        ppppp = list(T_sd[0:3, 3])
        qqqqq = R.from_matrix(T_sd[0:3, 0:3]).as_quat()
        self.make_tf_msg(ppppp, qqqqq, "world", "desired")

        ppppp_ = list(robot_target_position_ee)
        qqqqq_ = R.from_matrix(T_sd[0:3, 0:3]).as_quat()
        self.make_tf_msg(ppppp_, qqqqq_, "world", "human")

        sight_vec = T_cur[:3, 0]
        sight_vec /= np.linalg.norm(sight_vec)
        direction_unit_vector = np.array(self.human_position) - T_cur[:3, 3]
        direction_unit_vector = direction_unit_vector / np.linalg.norm(direction_unit_vector)
        cos_theta = np.dot(direction_unit_vector, sight_vec)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        T_bd = np.linalg.inv(T_sb) @ T_sd
        H_desired = SE3(T_bd)

        F = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        J_p = base.tr2adjoint(T_be.T) @ F
        J_a_e = base.tr2adjoint(T_be.T) @ self.ur5e_robot.jacob0(self.q[2:])
        J_mb = np.hstack((J_p, J_a_e))
        J_mb_v = J_mb[:3, :]
        J_mb_w = J_mb[3:, :]

        T_error = np.linalg.inv(H_current.A) @ H_desired.A
        et = np.sum(np.abs(T_error[:3, -1]))

        Q = np.eye(self.n_dof + 6)
        Q[:2, :2] *= 1.0 / max(et * 100, 1e-6)
        Q[self.n_dof:, self.n_dof:] = (1.0 / max(et, 1e-6)) * np.eye(6)

        H = np.zeros((self.n_dof - 2, 6, self.n_dof - 2))
        for j in range(self.n_dof - 2):
            for i in range(j, self.n_dof - 2):
                H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                if i != j:
                    H[i, :3, j] = H[j, :3, i]
                    H[i, 3:, j] = H[j, 3:, i]

        J_a = self.ur5e_robot.jacob0(self.q[2:])
        m = J_a @ J_a.T
        m_det = np.linalg.det(m)
        m_t = np.sqrt(m_det)
        rank = np.linalg.matrix_rank(J_a @ J_a.T)
        if rank < J_a.shape[0]:
            JJ_inv = np.linalg.pinv(J_a @ J_a.T)
        else:
            JJ_inv = np.linalg.inv(J_a @ J_a.T)

        J_m = np.zeros((self.n_dof - 2, 1))
        for i in range(self.n_dof - 2):
            c = J_a @ np.transpose(H[i, :, :])
            J_m[i, 0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

        A = np.zeros((len(xform_pose), self.n_dof + 6))
        B = np.zeros(len(xform_pose))
        J_dj = np.zeros(self.n_dof + 6)
        w_p_sum = 0.0
        min_dist_list = []
        g_vec_list = []
        weighted_g_vec = np.zeros(3)
        total_weight = 0.0

        for i, pose in enumerate(xform_pose):
            distance, index, g_vec_ = self.kinematics.get_nearest_obstacle_distance(pose, self.obstacles_positions[:, :3], self.config.obstacle_radius)
            min_dist_ = np.min(distance)
            min_dist_list.append(min_dist_)
            g_vec_list.append(g_vec_)
            if min_dist_ <= 0.6:
                weight = 0.15
            else:
                weight = 0.0
            weighted_g_vec += weight * g_vec_
            total_weight += weight

        if total_weight > 0:
            g_vec = weighted_g_vec / total_weight
        else:
            g_vec = np.zeros(3)
        min_dist = np.min(min_dist_list)
        min_index = np.argmin(min_dist_list)
        g_vec = g_vec_list[min_index]
        if np.linalg.norm(g_vec) > 0:
            g_vec /= np.linalg.norm(g_vec)

        link1_len = (num_mobile + len(self.kinematics.ur5e_link_points["shoulder"]))
        link2_len = (link1_len + len(self.kinematics.ur5e_link_points["upper_arm"]))
        link3_len = (link2_len + len(self.kinematics.ur5e_link_points["forearm"]))
        link4_len = (link3_len + len(self.kinematics.ur5e_link_points["wrist_1"]))
        link5_len = (link4_len + len(self.kinematics.ur5e_link_points["wrist_2"]))

        for i, pose in enumerate(xform_pose):
            if i < num_mobile:
                d_dot = g_vec @ J_mb_v
                A[i, :8] = d_dot
                A[i, 8:] = np.zeros((1, 6))
                B[i] = (min_dist_list[i] - self.config.d_safe) / ((self.config.d_influence - self.config.d_safe))
                w_p = (self.config.d_influence - min_dist_list[i]) / (self.config.d_influence - self.config.d_safe)
                J_dj[:8] += (d_dot) * (w_p)
                w_p_sum += np.abs(w_p)
            else:
                if num_mobile <= i < link1_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 2)), J_a_e[:3, :]])
                elif link1_len <= i < link2_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 3)), J_a_e[:3, 1:]])
                elif link2_len <= i < link3_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 4)), J_a_e[:3, 2:]])
                elif link3_len <= i < link4_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 5)), J_a_e[:3, 3:]])
                elif link4_len <= i < link5_len:
                    J_mb_arm_v = np.hstack([np.zeros((3, 6)), J_a_e[:3, 4:]])
                else:
                    J_mb_arm_v = np.hstack([np.zeros((3, 7)), J_a_e[:3, 5:]])

                d_dot = g_vec @ J_mb_arm_v
                A[i, :8] = d_dot
                A[i, 8:] = np.zeros((1, 6))
                B[i] = (min_dist_list[i] - self.config.d_safe) / ((self.config.d_influence - self.config.d_safe))
                w_p = (self.config.d_influence - min_dist_list[i]) / (self.config.d_influence - self.config.d_safe)
                J_dj[:8] += (d_dot) * (w_p)
                w_p_sum += np.abs(w_p)

        C1 = np.concatenate((np.zeros(2), -J_m.reshape((self.n_dof - 2,)), np.zeros(6)))
        try:
            bTe = self.ur5e_robot.fkine(self.q[2:], include_base=False).A
            theta_e = atan2(bTe[1, -1], bTe[0, -1])
            C2 = np.zeros(self.n_dof + 6)
            C2[0] = -5. * theta_e
        except Exception:
            C2 = np.zeros(self.n_dof + 6)

        lambda_max = 5.
        min_distance = np.min(min_dist_list) if len(min_dist_list) > 0 else 1e9
        if min_distance <= self.config.d_influence:
            lambda_c = (lambda_max / (self.config.d_influence - self.config.d_safe) ** 2) * (min_distance - self.config.d_influence) ** 2
        else:
            lambda_c = 0.0
        J_c = lambda_c * J_dj / w_p_sum if w_p_sum != 0 else np.zeros_like(J_dj)
        C3 = J_c

        J_h = np.zeros(self.n_dof + 6)
        J_mb_w_h = direction_unit_vector @ J_mb_w
        epsilon = 1e-6
        lambda_h = self.config.lambda_h_a_param * max(abs(theta), epsilon)
        J_h[:8] = lambda_h * J_mb_w_h
        C4 = J_h

        C = self.config.w1 * C1 + self.config.w2 * C2 + self.config.w3 * C3 + self.config.w4 * C4
        J_ = np.c_[J_mb, np.eye(6)]
        eTep = T_error
        e = np.zeros(6)
        e[:3] = eTep[:3, -1]
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        k = np.eye(6)
        v = k @ e

        lb = -np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]

        x_ = cp.Variable(self.n_dof + 6)
        objective = cp.Minimize(0.5 * cp.quad_form(x_, Q) + C.T @ x_)
        constraints = [x_ >= lb, x_ <= ub, J_ @ x_ == v, A @ x_ <= B]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if x_.value is not None:
            qd = np.asarray(x_.value, dtype=float)
        else:
            qd = np.zeros(self.n_dof + 6)

        if et < self.target_reached_threshold:
            qd = np.zeros(self.n_dof + 6)

        self.tracker.record_objective(qd, Q, C1, C2, C3, C4)
        self.tracker.problem_t.append(prob.value if prob.value is not None else 0.0)
        self.tracker.entire_t.append(time.time() - entire_start_t)

        wc, vc = qd[0], qd[1]
        qdc = qd[2:8]
        twist = Twist()
        twist.linear.x = vc
        twist.angular.z = wc
        self.scout_publisher.publish(twist)
        self.rtde_c.speedJ(qdc, 0.2, self.config.dt)
        self.rtde_c.waitPeriod(t_start)

    def cleanup_and_plot(self):
        print("\n🛑 프로그램 종료 중...")
        self.tracker.save_ee_trajectory(self.obstacles_positions, self.config.obstacle_radius, self.config.w1, self.config.w2, self.config.w4, self.target_reached_threshold)
        try:
            self.tracker.plot_objective_time_series()
        except Exception as e:
            print(f"Failed to save objective plot: {e}")


def main():
    rclpy.init()
    node = QP_mbcontorller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 감지됨")
    finally:
        if rclpy.ok():
            print("✨ ROS 2 종료 정리 시작...")
            node.cleanup_and_plot()
            node.destroy_node()
            rclpy.shutdown()
        else:
            print("✨ ROS 2 이미 종료됨. 추가 정리 생략.")


if __name__ == "__main__":
    main()
