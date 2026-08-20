import os
import time

import matplotlib.pyplot as plt
import numpy as np


class TrajectoryLogger:
    def __init__(self, w1, w2, w4, obstacle_radius, obstacles_positions, save_dir=None):
        self.w1 = w1
        self.w2 = w2
        self.w4 = w4
        self.obstacle_radius = obstacle_radius
        self.obstacles_positions = np.asarray(obstacles_positions)
        self.save_dir = save_dir or "/home/nvidia/geon/robotics/jetson_orin/controller/mb_control/hand_trajectories_plots"

        self.base_trajectory = {"x": [], "y": [], "timestamps": []}
        self.ee_trajectory = {"x": [], "y": [], "timestamps": []}
        self.desired_trajectory = {"x": [], "y": [], "timestamps": []}

        self.obj_timestamps = []
        self.obj_total = []
        self.obj_Qterm = []
        self.obj_C1 = []
        self.obj_C2 = []
        self.obj_C3 = []
        self.obj_C4 = []
        self.problem_t = []
        self.entire_t = []
        self.start_time = time.time()

    def record_trajectory(self, current_time, base_position, ee_position, desired_position):
        self.base_trajectory["x"].append(float(base_position[0]))
        self.base_trajectory["y"].append(float(base_position[1]))
        self.base_trajectory["timestamps"].append(float(current_time))

        self.ee_trajectory["x"].append(float(ee_position[0]))
        self.ee_trajectory["y"].append(float(ee_position[1]))
        self.ee_trajectory["timestamps"].append(float(current_time))

        self.desired_trajectory["x"].append(float(desired_position[0]))
        self.desired_trajectory["y"].append(float(desired_position[1]))
        self.desired_trajectory["timestamps"].append(float(current_time))

    def record_objective(self, qd, Q, C1=None, C2=None, C3=None, C4=None):
        try:
            qd_vec = np.asarray(qd).reshape(-1)
            qterm = 0.5 * float(qd_vec.T @ Q @ qd_vec)
            c1 = float(C1 @ qd_vec) if C1 is not None else 0.0
            c2 = float(C2 @ qd_vec) if C2 is not None else 0.0
            c3 = float(C3 @ qd_vec) if C3 is not None else 0.0
            c4 = float(C4 @ qd_vec) if C4 is not None else 0.0

            self.obj_timestamps.append(time.time() - self.start_time)
            self.obj_Qterm.append(qterm)
            self.obj_C1.append(c1)
            self.obj_C2.append(c2)
            self.obj_C3.append(c3)
            self.obj_C4.append(c4)
            self.obj_total.append(qterm + c1 + c2 + c3 + c4)
        except Exception as exc:
            print(f"Objective logging failed: {exc}")

    def record_timing(self, problem_time, entire_time):
        self.problem_t.append(float(problem_time))
        self.entire_t.append(float(entire_time))

    def plot_objective_time_series(self, save_path=None, show_plot=False):
        if not self.obj_timestamps:
            print("No objective history to plot.")
            return

        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(10, 6))
        ts = np.asarray(self.obj_timestamps)
        ax.plot(ts, self.obj_Qterm, color="k", linestyle="--", linewidth=1.5, label="0.5 x^T Q x")
        ax.plot(ts, self.obj_total, color="r", linewidth=2, label="Total objective")
        ax.fill_between(ts, self.obj_C1, alpha=0.20, label="C1")
        ax.fill_between(ts, self.obj_C2, alpha=0.20, label="C2")
        ax.fill_between(ts, self.obj_C3, alpha=0.20, label="C3")
        ax.fill_between(ts, self.obj_C4, alpha=0.20, label="C4")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective value")
        ax.set_title("QP objective over time")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        save_path = save_path or os.path.join(self.save_dir, f"objective_time_series_{int(time.time())}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Objective plot saved: {save_path}")

        if show_plot:
            try:
                from subprocess import Popen

                Popen(["xdg-open", save_path])
            except Exception:
                pass

    def save_ee_trajectory(self):
        if not self.ee_trajectory["x"]:
            print("📊 기록된 궤적이 없습니다.")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.ee_trajectory["x"], self.ee_trajectory["y"], "c-", linewidth=2, label="Actual EE Trajectory")
        if self.desired_trajectory["x"]:
            ax.plot(self.desired_trajectory["x"], self.desired_trajectory["y"], "b--", linewidth=2, label="Desired Trajectory")
        if self.obstacles_positions.size:
            obstacle = self.obstacles_positions[0]
            circle = plt.Circle((obstacle[0], obstacle[1]), self.obstacle_radius, color="red", alpha=0.4)
            ax.add_patch(circle)

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Actual and Desired XY Trajectory")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        trajectory_plot = os.path.join(self.save_dir, f"ee_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.png")
        plt.savefig(trajectory_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)

        csv_filename = os.path.join(self.save_dir, f"ee_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.csv")
        with open(csv_filename, "w", encoding="utf-8") as file_handle:
            file_handle.write("timestamp,actual_x,actual_y\n")
            for index in range(len(self.ee_trajectory["x"])):
                file_handle.write(
                    f"{self.ee_trajectory['timestamps'][index]:.3f},"
                    f"{self.ee_trajectory['x'][index]:.6f},"
                    f"{self.ee_trajectory['y'][index]:.6f}\n"
                )

        if self.desired_trajectory["x"]:
            desired_csv = os.path.join(self.save_dir, f"desired_trajectory_{self.w1},{self.w2},{self.w4}_{timestamp}.csv")
            with open(desired_csv, "w", encoding="utf-8") as file_handle:
                file_handle.write("timestamp,desired_x,desired_y\n")
                for index in range(len(self.desired_trajectory["x"])):
                    file_handle.write(
                        f"{self.desired_trajectory['timestamps'][index]:.3f},"
                        f"{self.desired_trajectory['x'][index]:.6f},"
                        f"{self.desired_trajectory['y'][index]:.6f}\n"
                    )
        else:
            desired_csv = None

        print("\n📊 궤적 추종 결과 저장 완료:")
        print(f"   그래프: {trajectory_plot}")
        print(f"   Actual 데이터: {csv_filename}")
        if desired_csv is not None:
            print(f"   Desired 데이터: {desired_csv}")
