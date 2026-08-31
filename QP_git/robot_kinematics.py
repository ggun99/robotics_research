from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


class RobotKinematics:
    """FK, collision-point 계산, nearest-obstacle 기준 계산을 담당."""

    def __init__(self, ur5e_robot):
        self.ur5e_robot = ur5e_robot
        self.define_robot_collision_points()
        self.ur5e_frame_correction = np.eye(4)
        self.ur5e_frame_correction[:3, :3] = R.from_euler("z", np.pi).as_matrix()

    def define_robot_collision_points(self):
        self.ur5e_link_points = {
            "base": np.array([
                [0.0, 0.0, 0.0],
                [0.08, 0.08, 0.05],
                [-0.08, 0.08, 0.05],
                [0.08, -0.08, 0.05],
                [-0.08, -0.08, 0.05],
                [0.06, 0.0, 0.1],
                [-0.06, 0.0, 0.1],
                [0.0, 0.06, 0.1],
                [0.0, -0.06, 0.1],
            ]),
            "shoulder": np.array([
                [0.0, 0.0, 0.0],
                [0.06, 0.06, 0.08],
                [-0.06, 0.06, 0.08],
                [0.06, -0.06, 0.08],
                [-0.06, -0.06, 0.08],
                [0.0, 0.0, 0.12],
            ]),
            "upper_arm": np.array([
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.1],
                [-0.05, 0.0, 0.1],
                [0.0, 0.05, 0.1],
                [0.0, -0.05, 0.1],
                [0.04, 0.0, 0.2],
                [-0.04, 0.0, 0.2],
                [0.0, 0.04, 0.25],
                [0.0, -0.04, 0.25],
            ]),
            "forearm": np.array([
                [0.4, 0.0, 0.0],
                [0.04, 0.0, 0.08],
                [-0.04, 0.0, 0.08],
                [0.0, 0.04, 0.08],
                [0.0, -0.04, 0.08],
                [0.03, 0.0, 0.15],
                [-0.03, 0.0, 0.15],
                [0.0, 0.03, 0.17],
                [0.0, -0.03, 0.17],
            ]),
            "wrist_1": np.array([
                [0.0, 0.0, 0.0],
                [0.03, 0.03, 0.0],
                [-0.03, 0.03, 0.0],
                [0.03, -0.03, 0.0],
                [-0.03, -0.03, 0.0],
            ]),
            "wrist_2": np.array([
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.03],
                [-0.03, 0.0, 0.03],
                [0.0, 0.03, 0.03],
                [0.0, -0.03, 0.03],
            ]),
            "wrist_3": np.array([
                [0.0, 0.0, 0.0],
                [0.025, 0.025, 0.02],
                [-0.025, 0.025, 0.02],
                [0.025, -0.025, 0.02],
                [-0.025, -0.025, 0.02],
            ]),
        }

        self.mobile_base_points = np.array([
            [0.35, 0.25, 0.1],
            [0.25, 0.25, 0.1],
            [0.15, 0.25, 0.1],
            [0.05, 0.25, 0.1],
            [-0.05, 0.25, 0.1],
            [-0.15, 0.25, 0.1],
            [-0.25, 0.25, 0.1],
            [-0.35, 0.25, 0.1],
            [0.35, -0.25, 0.1],
            [0.25, -0.25, 0.1],
            [0.15, -0.25, 0.1],
            [0.05, -0.25, 0.1],
            [-0.05, -0.25, 0.1],
            [-0.15, -0.25, 0.1],
            [-0.25, -0.25, 0.1],
            [-0.35, -0.25, 0.1],
            [0.35, 0.15, 0.1],
            [0.35, 0.05, 0.1],
            [0.35, -0.05, 0.1],
            [0.35, -0.15, 0.1],
            [-0.35, 0.15, 0.1],
            [-0.35, 0.05, 0.1],
            [-0.35, -0.05, 0.1],
            [-0.35, -0.15, 0.1],
            [0.3, 0.2, 0.3],
            [-0.3, 0.2, 0.3],
            [0.3, -0.2, 0.3],
            [-0.3, -0.2, 0.3],
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
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def ur5e_fk_chain_manual(self, q_arm):
        a = [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0]
        alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
        d = [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]

        q = np.asarray(q_arm, dtype=float).reshape(-1)
        q_pad = np.zeros(6, dtype=float)
        q_pad[: q.size] = q

        transforms = [np.eye(4)]
        T_acc = np.eye(4)
        for i in range(6):
            T_link = self._dh_transform(a[i], alpha[i], d[i], q_pad[i])
            T_acc = T_acc @ T_link
            transforms.append(T_acc.copy())
            if i + 1 >= q.size:
                break
        return transforms

    def compute_dynamic_robot_collision_points(self, state):
        if state.base_position is None or state.base_quaternion is None:
            return np.empty((0, 3), dtype=float), 0, 0

        collision_points_world = []
        T_sb = self.build_base_transform(state.base_position, state.base_quaternion)

        T_b0 = np.eye(4)
        T_b0[0, 3] = 0.1315
        T_b0[2, 3] = 0.51921

        for point in self.mobile_base_points:
            point_h = np.append(point, 1.0)
            point_world = T_sb @ point_h
            collision_points_world.append(point_world[:3])
        num_mobile = len(collision_points_world)

        link_names = ["base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
        num_mani = 0

        fk_chain = self.ur5e_fk_chain_manual(state.q[2:])
        for i, link_name in enumerate(link_names):
            if i < len(fk_chain):
                T_0i = fk_chain[i]
            else:
                T_0i = fk_chain[-1] if len(fk_chain) > 0 else np.eye(4)

            T_si = T_sb @ T_b0 @ self.ur5e_frame_correction @ T_0i
            if link_name in self.ur5e_link_points:
                for point in self.ur5e_link_points[link_name]:
                    p_h = np.append(point, 1.0)
                    p_world = T_si @ p_h
                    collision_points_world.append(p_world[:3])
                    num_mani += 1

        return np.asarray(collision_points_world, dtype=float), num_mobile, num_mani

    def get_nearest_obstacle_distance(self, position, obstacles, obstacle_radius):
        """position: (3,), obstacles: (N,3)"""
        obstacles = np.asarray(obstacles, dtype=float)
        if obstacles.size == 0:
            return np.array([], dtype=float), -1, np.array([0.0, 0.0, 1.0])

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
