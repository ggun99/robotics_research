from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ControllerConfig:
    """전반적인 QP controller 설정값을 한 곳에서 관리한다."""

    n_dof: int = 8
    dt: float = 0.05
    d_safe: float = 0.1
    d_influence: float = 1.0
    lambda_max: float = 5.0
    lambda_h_a_param: float = 0.3
    obstacle_radius: float = 0.25
    target_reached_threshold: float = 0.2
    robot_ip: str = "192.160.0.4"

    # objective weights
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 0.0
    w4: float = 0.2

    qdlim: np.ndarray = field(
        default_factory=lambda: np.array([0.15] * 8, dtype=float)
    )

    def __post_init__(self) -> None:
        self.qdlim = np.asarray(self.qdlim, dtype=float)
        self.qdlim[:1] = 0.05
        self.qdlim[1] = 0.05


@dataclass
class RobotState:
    """ROS callback에서 받아온 로봇 상태를 묶어서 전달한다."""

    q: Optional[np.ndarray] = None
    base_position: Optional[np.ndarray] = None
    base_quaternion: Optional[np.ndarray] = None
    human_position: Optional[np.ndarray] = None
    cable_points: list = field(default_factory=list)
    obstacles_positions: np.ndarray = field(
        default_factory=lambda: np.array([[0.6, 0.0, 0.5]], dtype=float)
    )

    def copy(self) -> "RobotState":
        return RobotState(
            q=None if self.q is None else np.array(self.q, dtype=float).copy(),
            base_position=None
            if self.base_position is None
            else np.array(self.base_position, dtype=float).copy(),
            base_quaternion=None
            if self.base_quaternion is None
            else np.array(self.base_quaternion, dtype=float).copy(),
            human_position=None
            if self.human_position is None
            else np.array(self.human_position, dtype=float).copy(),
            cable_points=[np.array(p, dtype=float).copy() for p in self.cable_points],
            obstacles_positions=np.array(self.obstacles_positions, dtype=float).copy(),
        )
