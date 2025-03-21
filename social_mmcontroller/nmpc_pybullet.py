import pybullet as p
import pybullet_data
import time
import numpy as np

# PyBullet 시뮬레이터 연결
p.connect(p.GUI)  # GUI 모드 실행
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 기본 데이터 경로 추가
p.setGravity(0, 0, -9.8)  # 중력 설정

# 평면 로드
plane_id = p.loadURDF("plane.urdf")

# 모바일 매니퓰레이터 로드 (예제: UR5 + 모바일 베이스)
robot_urdf = "urdf/your_robot.urdf"  # 사용자의 로봇 URDF 경로 설정
robot_id = p.loadURDF(robot_urdf, basePosition=[0, 0, 0])

# 장애물 추가 (예제: 박스)
obstacle_id = p.loadURDF("cube_small.urdf", basePosition=[2, 2, 0.5])

import casadi as ca

# 상태 변수: (x, y, theta, q1, q2, q3) - 모바일 베이스 + 매니퓰레이터
n_states = 6
n_controls = 3
T = 10
dt = 0.1  # 100ms 샘플링

X = ca.MX.sym("X", n_states, T+1)  # 상태 변수
U = ca.MX.sym("U", n_controls, T)  # 제어 입력

# 목표 위치 (예: x=3, y=3, 매니퓰레이터 q1=30도)
goal_state = np.array([3, 3, 0, np.pi/6, np.pi/6, np.pi/6])

# 비용 함수 정의
Q_pos = np.diag([10, 10, 1])  # 위치 가중치
Q_manip = np.diag([5, 5, 5])  # 매니퓰레이터 가중치
R = np.diag([0.1, 0.1, 0.1])  # 제어 입력 가중치

cost = 0
for k in range(T):
    cost += ca.mtimes((X[:3, k] - goal_state[:3]).T, Q_pos, (X[:3, k] - goal_state[:3]))
    cost += ca.mtimes((X[3:, k] - goal_state[3:]).T, Q_manip, (X[3:, k] - goal_state[3:]))
    cost += ca.mtimes(U[:, k].T, R, U[:, k])

# 장애물 회피 조건 추가
obstacle = np.array([2, 2])  # 장애물 위치
safe_distance = 0.5
constraints = []
for k in range(T):
    dist_to_obstacle = ca.sqrt((X[0, k] - obstacle[0])**2 + (X[1, k] - obstacle[1])**2)
    constraints.append(dist_to_obstacle - safe_distance)  # 최소 거리 유지

# 최적화 문제 정의
nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), 
       'f': cost, 
       'g': ca.vertcat(*constraints)}

solver = ca.nlpsol('solver', 'ipopt', nlp)


# 초기 상태
x_current = np.array([0, 0, 0, 0, 0, 0])

# PyBullet에서 제어할 joint index
base_joint_indices = [0, 1]  # x, y 방향
arm_joint_indices = [2, 3, 4]  # 매니퓰레이터 관절

for i in range(100):  # 100번 반복 (10초 실행)
    # NMPC 최적화 실행
    x_init = np.tile(x_current, (T+1, 1)).flatten()
    u_init = np.zeros((T, n_controls)).flatten()
    sol = solver(x0=np.concatenate([x_init, u_init]))

    # 최적 경로에서 첫 번째 제어 입력을 적용
    u_opt = np.reshape(sol['x'][n_states*(T+1):], (n_controls, T))
    x_next = x_current.copy()
    x_next[0] += u_opt[0, 0] * dt  # x 이동
    x_next[1] += u_opt[1, 0] * dt  # y 이동
    x_next[2] += u_opt[2, 0] * dt  # 회전
    x_next[3] += 0.1 * dt  # q1
    x_next[4] += 0.1 * dt  # q2
    x_next[5] += 0.1 * dt  # q3

    # PyBullet에 적용
    p.setJointMotorControlArray(robot_id, base_joint_indices, p.VELOCITY_CONTROL, targetVelocities=[u_opt[0, 0], u_opt[1, 0]])
    p.setJointMotorControlArray(robot_id, arm_joint_indices, p.POSITION_CONTROL, targetPositions=x_next[3:])

    # 현재 상태 업데이트
    x_current = x_next

    # 시뮬레이션 스텝
    p.stepSimulation()
    time.sleep(dt)

p.disconnect()
