from __future__ import annotations

import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
from spatialmath import base


class QPController:
    """QP 수식 및 해석을 담당하는 계산 class."""

    def __init__(self, config):
        self.config = config

    def compute_tracking_error(self, T_cur, T_des):
        T_error = np.linalg.inv(T_cur) @ T_des
        return T_error

    def compute_jacobian(self, T_be, ur5e_robot, q_arm):
        F = np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )

        J_p = base.tr2adjoint(T_be.T) @ F
        J_a_e = base.tr2adjoint(T_be.T) @ ur5e_robot.jacob0(q_arm)
        J_mb = np.hstack((J_p, J_a_e))
        J_mb_v = J_mb[:3, :]
        J_mb_w = J_mb[3:, :]
        return J_mb, J_mb_v, J_mb_w

    def solve_qp(self, state, kin, T_cur, T_be, T_des):
        """현재 힙한 수식 구조를 유지하면서, QP 계산만 분리한다.

        여기서는 기존 `QP_real` 함수에서 핵심이 되는 QP 구성 부분을
        class로 옮기는 구조를 보여준다.
        """
        if state.base_position is None or state.base_quaternion is None:
            raise ValueError("base position / quaternion is required")

        n_dof = self.config.n_dof

        # 기본 위치/회전 오차 계산
        H_current = T_cur
        T_error = self.compute_tracking_error(H_current, T_des)
        et = np.sum(np.abs(T_error[:3, -1]))

        Q = np.eye(n_dof + 6)
        Q[:2, :2] *= 1.0 / max(et * 100, 1e-6)
        Q[n_dof:, n_dof:] = (1.0 / max(et, 1e-6)) * np.eye(6)

        # 예시: 실제 로봇 Jacobian / 장애물 avoidance는 외부에서 계산해서 전달해야 함
        # 여기서는 기본 형태만 남긴다.
        J_mb, J_mb_v, J_mb_w = self.compute_jacobian(T_be, kin.ur5e_robot, state.q[2:])
        J_ = np.c_[J_mb, np.eye(6)]

        e = np.zeros(6)
        e[:3] = T_error[:3, -1]
        e[3:] = base.tr2rpy(T_error, unit="rad", order="zyx", check=False)
        k = np.eye(6)
        v = k @ e

        lb = -np.r_[self.config.qdlim[: n_dof], 10 * np.ones(6)]
        ub = np.r_[self.config.qdlim[: n_dof], 10 * np.ones(6)]

        x_ = cp.Variable(n_dof + 6)
        objective = cp.Minimize(0.5 * cp.quad_form(x_, Q))
        constraints = [
            x_ >= lb,
            x_ <= ub,
            J_ @ x_ == v,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if x_.value is None:
            return np.zeros(n_dof + 6)

        qd = np.asarray(x_.value, dtype=float)
        return qd

    def build_objective_terms(self, qd, Q, C1, C2, C3, C4):
        x_vec = np.asarray(qd, dtype=float).reshape(-1)
        Qterm = 0.5 * float(x_vec.T @ Q @ x_vec)
        c1x = float(C1 @ x_vec) if C1 is not None else 0.0
        c2x = float(C2 @ x_vec) if C2 is not None else 0.0
        c3x = float(C3 @ x_vec) if C3 is not None else 0.0
        c4x = float(C4 @ x_vec) if C4 is not None else 0.0
        return {
            "Qterm": Qterm,
            "C1": c1x,
            "C2": c2x,
            "C3": c3x,
            "C4": c4x,
            "total": Qterm + c1x + c2x + c3x + c4x,
        }
