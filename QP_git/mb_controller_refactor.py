from __future__ import annotations

import numpy as np

from controller_config import ControllerConfig
from objective_plotter import ObjectivePlotter
from qp_controller import QPController
from robot_kinematics import RobotKinematics


class MBControllerRefactor:
    """실제 ROS 노드에서 사용하는 조립용 구조 예시.

    - ROS 통신은 이 class에서 처리
    - 계산은 QPController / RobotKinematics 로 분리
    - 그래프는 ObjectivePlotter 로 분리
    """

    def __init__(self, robot_model):
        self.config = ControllerConfig()
        self.kinematics = RobotKinematics(robot_model)
        self.qp = QPController(self.config)
        self.plotter = ObjectivePlotter()

        self.state = None

    def update_state(self, state):
        self.state = state

    def step(self):
        if self.state is None:
            return None

        q = np.asarray(self.state.q, dtype=float)
        T_cur = np.eye(4)
        T_be = np.eye(4)
        T_des = np.eye(4)

        qd = self.qp.solve_qp(self.state, self.kinematics, T_cur, T_be, T_des)

        # 실험용 간단 기록
        Q = np.eye(self.config.n_dof + 6)
        C1 = np.zeros(self.config.n_dof + 6)
        C2 = np.zeros(self.config.n_dof + 6)
        C3 = np.zeros(self.config.n_dof + 6)
        C4 = np.zeros(self.config.n_dof + 6)
        self.plotter.record(qd, Q, C1, C2, C3, C4)

        return qd


if __name__ == "__main__":
    # 예시용 실행 코드
    print("refactor skeleton ready")
