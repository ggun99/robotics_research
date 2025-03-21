import casadi as ca
import numpy as np

# 상태 변수: [x, y, theta] (모바일 베이스 위치 + 방향)
#            [q1, q2, q3] (매니퓰레이터 관절 각도)
# 입력 변수: [v, omega] (선속도, 회전속도)
n_states = 6   # (x, y, theta, q1, q2, q3)
n_controls = 3 # (vx, vy, w)

T = 10         # 예측 시간 구간 (10 x 0.1s = 1초)
dt = 0.1       # 샘플링 시간 (100ms)

# 상태 및 제어 변수 정의
X = ca.MX.sym("X", n_states, T+1)  # 상태 변수 (미래 상태 예측)
U = ca.MX.sym("U", n_controls, T)  # 제어 입력 (미래 입력 예측)

# 초기 상태 (x0)
x0 = np.array([0, 0, 0, 0, 0, 0])  

# 시스템 모델 (이동 로봇 + 매니퓰레이터)
def system_dynamics(x, u):
    x_next = x.copy()
    x_next[0] += u[0] * dt  # x 이동
    x_next[1] += u[1] * dt  # y 이동
    x_next[2] += u[2] * dt  # 회전
    x_next[3] += 0.1 * dt   # q1 (매니퓰레이터)
    x_next[4] += 0.1 * dt   # q2
    x_next[5] += 0.1 * dt   # q3
    return x_next

Q_pos = np.diag([10, 10, 1])  # 베이스의 위치 및 방향 가중치
Q_manip = np.diag([5, 5, 5])  # 매니퓰레이터의 조작 가능성 가중치
R = np.diag([0.1, 0.1, 0.1])  # 제어 입력 가중치

# 목표 상태 설정 (예: x=5, y=5, q1=30도)
goal_state = np.array([5, 5, 0, np.pi/6, np.pi/6, np.pi/6])

cost = 0
for k in range(T):
    cost += ca.mtimes((X[:3, k] - goal_state[:3]).T, Q_pos, (X[:3, k] - goal_state[:3]))  # 위치 오차 비용
    cost += ca.mtimes((X[3:, k] - goal_state[3:]).T, Q_manip, (X[3:, k] - goal_state[3:]))  # 매니퓰레이터 오차 비용
    cost += ca.mtimes(U[:, k].T, R, U[:, k])  # 제어 입력 비용

# 장애물 위치 (예: 장애물이 (3,3) 위치에 있음)
obstacle = np.array([3, 3])
safe_distance = 0.5  # 최소 거리

constraints = []
for k in range(T):
    dist_to_obstacle = ca.sqrt((X[0, k] - obstacle[0])**2 + (X[1, k] - obstacle[1])**2)
    constraints.append(dist_to_obstacle - safe_distance)  # 장애물과 최소 거리 유지

# 최적화 문제 설정
nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), 
       'f': cost, 
       'g': ca.vertcat(*constraints)}

# IPOPT Solver 설정
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 500}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# 초기 guess
x_init = np.tile(x0, (T+1, 1)).flatten()
u_init = np.zeros((T, n_controls)).flatten()

# 경계 조건
lbx = np.full(x_init.shape, -ca.inf)  # 상태 하한
ubx = np.full(x_init.shape, ca.inf)   # 상태 상한
lbg = np.zeros(len(constraints))      # 장애물 충돌 방지 조건
ubg = np.full(len(constraints), ca.inf)

# 최적화 실행
sol = solver(x0=np.concatenate([x_init, u_init]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

# 결과 추출
x_opt = np.reshape(sol['x'][:n_states*(T+1)], (n_states, T+1))
u_opt = np.reshape(sol['x'][n_states*(T+1):], (n_controls, T))

print("최적 경로:", x_opt)
print("최적 제어 입력:", u_opt)
