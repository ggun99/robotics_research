import time
from math import atan2

import cvxpy as cp
import numpy as np
from spatialmath import base


class MobileManipulatorQPSolver:
    def __init__(
        self,
        n_dof,
        qdlim,
        d_safe,
        d_influence,
        obstacle_radius,
        lambda_h_a_param,
        w1,
        w2,
        w3,
        w4,
        ur5e_link_points,
    ):
        self.n_dof = n_dof
        self.qdlim = qdlim
        self.d_safe = d_safe
        self.d_influence = d_influence
        self.obstacle_radius = obstacle_radius
        self.lambda_h_a_param = lambda_h_a_param
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.ur5e_link_points = ur5e_link_points

    @staticmethod
    def _nearest_obstacle_distance(position, obstacles, obstacle_radius):
        g_vec = np.zeros(3)
        obstacles_local = []
        obs_real = []

        for obs in obstacles:
            obs_local = np.array(obs, dtype=float).copy()
            obs_local[2] = position[2]
            obs_real.append(obs_local)
            obstacles_local.append(obs_local)

        distances = [np.linalg.norm(position - obs) - obstacle_radius for obs in obstacles_local]
        index = int(np.argmin(distances))

        g_vec = position - obs_real[index]
        norm = np.linalg.norm(g_vec)
        if norm > 1e-9:
            g_vec /= norm
        return distances, index, g_vec

    def solve(
        self,
        q,
        ur5e_robot,
        T_be,
        H_current,
        H_desired,
        xform_pose,
        num_mobile,
        obstacles_positions,
        direction_unit_vector,
        theta,
    ):
        F = np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        J_p = base.tr2adjoint(T_be.T) @ F
        J_a_e = base.tr2adjoint(T_be.T) @ ur5e_robot.jacob0(q[2:])
        J_mb = np.hstack((J_p, J_a_e))
        J_mb_v = J_mb[:3, :]
        J_mb_w = J_mb[3:, :]

        T_error = np.linalg.inv(H_current.A) @ H_desired.A
        et = np.sum(np.abs(T_error[:3, -1]))

        Q = np.eye(self.n_dof + 6)
        Q[:2, :2] *= 1.0 / max(et * 1000.0, 1e-6)
        Q[self.n_dof :, self.n_dof :] = (1.0 / max(et, 1e-6)) * np.eye(6)

        H = np.zeros((self.n_dof - 2, 6, self.n_dof - 2))
        for j in range(self.n_dof - 2):
            for i in range(j, self.n_dof - 2):
                H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                if i != j:
                    H[i, :3, j] = H[j, :3, i]
                    H[i, 3:, j] = H[j, 3:, i]

        J_a = ur5e_robot.jacob0(q[2:])
        m = J_a @ J_a.T
        m_det = np.linalg.det(m)
        m_t = np.sqrt(max(m_det, 0.0))

        rank = np.linalg.matrix_rank(J_a @ J_a.T)
        if rank < J_a.shape[0]:
            JJ_inv = np.linalg.pinv(J_a @ J_a.T)
        else:
            JJ_inv = np.linalg.inv(J_a @ J_a.T)

        J_m = np.zeros((self.n_dof - 2, 1))
        for i in range(self.n_dof - 2):
            c = J_a @ np.transpose(H[i, :, :])
            J_m[i, 0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

        num_points = len(xform_pose)
        A = np.zeros((num_points, self.n_dof + 6))
        B = np.zeros(num_points)

        J_dj = np.zeros(self.n_dof + 6)
        w_p_sum = 0.0
        min_dist_list = []
        g_vec_list = []

        for pose in xform_pose:
            distance, _, g_vec_each = self._nearest_obstacle_distance(
                pose, obstacles_positions[:, :3], self.obstacle_radius
            )
            min_dist_list.append(np.min(distance))
            g_vec_list.append(g_vec_each)

        if len(min_dist_list) == 0:
            min_dist_list = [self.d_influence + 1.0]
            g_vec_list = [np.zeros(3)]

        min_index = int(np.argmin(min_dist_list))
        g_vec = g_vec_list[min_index]
        norm = np.linalg.norm(g_vec)
        if norm > 1e-9:
            g_vec = g_vec / norm

        link1_len = num_mobile + len(self.ur5e_link_points["shoulder"])
        link2_len = link1_len + len(self.ur5e_link_points["upper_arm"])
        link3_len = link2_len + len(self.ur5e_link_points["forearm"])
        link4_len = link3_len + len(self.ur5e_link_points["wrist_1"])
        link5_len = link4_len + len(self.ur5e_link_points["wrist_2"])

        for i, _ in enumerate(xform_pose):
            if i < num_mobile:
                d_dot = g_vec @ J_mb_v
                A[i, :8] = d_dot
                B[i] = (min_dist_list[i] - self.d_safe) / (self.d_influence - self.d_safe)
                w_p = (
                    (self.d_influence - min_dist_list[i]) / (self.d_influence - self.d_safe)
                    if min_dist_list[i] < self.d_safe
                    else 1.0
                )
                J_dj[:8] += d_dot * w_p
                w_p_sum += abs(w_p)
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
                B[i] = (min_dist_list[i] - self.d_safe) / (self.d_influence - self.d_safe)
                w_p = (
                    (self.d_influence - min_dist_list[i]) / (self.d_influence - self.d_safe)
                    if min_dist_list[i] < self.d_safe
                    else 1.0
                )
                J_dj[:8] += d_dot * w_p
                w_p_sum += abs(w_p)

        C1 = np.concatenate((np.zeros(2), -J_m.reshape((self.n_dof - 2,)), np.zeros(6)))
        try:
            bTe = ur5e_robot.fkine(q[2:], include_base=False).A
            theta_e = atan2(bTe[1, -1], bTe[0, -1])
            C2 = np.zeros(self.n_dof + 6)
            C2[0] = -5.0 * theta_e
        except Exception:
            C2 = np.zeros(self.n_dof + 6)

        min_distance = np.min(min_dist_list)
        lambda_max = 5.0
        if min_distance <= self.d_influence:
            lambda_c = (lambda_max / (self.d_influence - self.d_safe) ** 2) * (
                min_distance - self.d_influence
            ) ** 2
        else:
            lambda_c = 0.0

        w_p_denom = max(w_p_sum, 1e-9)
        C3 = lambda_c * J_dj / w_p_denom

        J_h = np.zeros(self.n_dof + 6)
        J_mb_w_h = direction_unit_vector @ J_mb_w
        epsilon = 1e-6
        lambda_h = self.lambda_h_a_param * max(abs(theta), epsilon)
        J_h[:8] = lambda_h * J_mb_w_h
        C4 = J_h

        C = self.w1 * C1 + self.w2 * C2 + self.w3 * C3 + self.w4 * C4

        J_ = np.c_[J_mb, np.eye(6)]
        e = np.zeros(6)
        e[:3] = T_error[:3, -1]
        e[3:] = base.tr2rpy(T_error, unit="rad", order="zyx", check=False)
        v = np.eye(6) @ e

        lb = -np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]

        x_var = cp.Variable(self.n_dof + 6)
        objective = cp.Minimize(0.5 * cp.quad_form(x_var, Q) + C.T @ x_var)
        constraints = [
            x_var >= lb,
            x_var <= ub,
            J_ @ x_var == v,
            A @ x_var <= B,
        ]

        t0 = time.time()
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        problem_solve_time = time.time() - t0

        if x_var.value is None:
            qd = np.zeros(self.n_dof + 6)
        else:
            qd = x_var.value

        return {
            "qd": qd,
            "et": et,
            "Q": Q,
            "C1": C1,
            "C2": C2,
            "C3": C3,
            "C4": C4,
            "problem_solve_time": problem_solve_time,
        }
