from omni.isaac.kit import SimulationApp

# 시뮬레이션 앱 시작
simulation_app = SimulationApp({"headless": False})  # GUI를 보려면 headless=False

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers import BaseController
import numpy as np
import time

# World 생성
world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

# UR5e 로봇 경로 (NVIDIA 기본 경로 기준)
ur5e_path = assets_root_path + "/Robots/UR/ur5e/ur5e.usd"
robot_prim_path = "/World/UR5e"

# 스테이지에 로봇 추가
add_reference_to_stage(ur5e_path, robot_prim_path)

# 시뮬레이션 초기화
world.scene.add_default_ground_plane()
world.reset()

# UR5e 로봇 객체 생성
ur5e_robot = world.scene.get_articulation(robot_prim_path)

# 사용자 정의 컨트롤러 클래스
class MyVelocityController(BaseController):
    def forward(self, joint_positions: np.ndarray) -> np.ndarray:
        # 예: 간단한 proportional controller (원점 복귀)
        target_position = np.zeros_like(joint_positions)
        error = target_position - joint_positions
        velocity_command = 1.0 * error  # gain = 1.0
        return velocity_command

# 컨트롤러 인스턴스
controller = MyVelocityController()

# 시뮬레이션 루프
for step in range(500):
    world.step(render=True)

    # 현재 조인트 상태 읽기
    joint_positions = ur5e_robot.get_joint_positions()
    
    # 컨트롤러로부터 조인트 속도 명령 생성
    joint_velocities = controller.forward(joint_positions)

    # 조인트 속도 명령 적용
    ur5e_robot.set_joint_velocities(joint_velocities)
    ur5e_robot.apply_action()

    time.sleep(1.0 / 60.0)

simulation_app.close()
