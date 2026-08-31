from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ObjectivePlotter:
    """목적함수 값 기록 및 그래프 저장만 담당하는 class."""

    def __init__(self):
        self.obj_timestamps = []
        self.obj_Qterm = []
        self.obj_C1 = []
        self.obj_C2 = []
        self.obj_C3 = []
        self.obj_C4 = []
        self.obj_total = []

    def record(self, qd, Q, C1, C2, C3, C4):
        x_vec = np.asarray(qd, dtype=float).reshape(-1)
        Qterm = 0.5 * float(x_vec.T @ Q @ x_vec)
        c1x = float(C1 @ x_vec) if C1 is not None else 0.0
        c2x = float(C2 @ x_vec) if C2 is not None else 0.0
        c3x = float(C3 @ x_vec) if C3 is not None else 0.0
        c4x = float(C4 @ x_vec) if C4 is not None else 0.0

        self.obj_timestamps.append(time.time())
        self.obj_Qterm.append(Qterm)
        self.obj_C1.append(c1x)
        self.obj_C2.append(c2x)
        self.obj_C3.append(c3x)
        self.obj_C4.append(c4x)
        self.obj_total.append(Qterm + c1x + c2x + c3x + c4x)

    def plot_objective_time_series(self, save_path=None, show_plot=False):
        if len(self.obj_timestamps) == 0:
            print("No objective history to plot.")
            return

        ts = np.asarray(self.obj_timestamps)
        ts = ts - ts[0]
        c1 = np.asarray(self.obj_C1)
        c2 = np.asarray(self.obj_C2)
        c3 = np.asarray(self.obj_C3)
        c4 = np.asarray(self.obj_C4)
        qterm = np.asarray(self.obj_Qterm)
        total = np.asarray(self.obj_total)

        fig, ax = plt.subplots(figsize=(10, 6))
        contributions = np.vstack([c1, c2, c3, c4])
        labels = ["C1·x", "C2·x", "C3·x", "C4·x"]
        colors = ["#ffd700", "#ffa07a", "#87cefa", "#90ee90"]
        ax.stackplot(ts, contributions, labels=labels, colors=colors, alpha=0.8)
        ax.plot(ts, qterm, color="k", linestyle="--", linewidth=1.5, label="0.5 x^T Q x")
        ax.plot(ts, total, color="r", linestyle="-", linewidth=2, label="Total objective")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective value")
        ax.set_title("QP objective and decomposition over time")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_path is None:
            save_dir = "/tmp"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"objective_{int(time.time())}.png")

        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Objective plot saved: {save_path}")

        if show_plot:
            try:
                import subprocess
                subprocess.Popen(["xdg-open", save_path])
            except Exception:
                pass
