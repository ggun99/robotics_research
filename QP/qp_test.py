import numpy as np
import cvxpy as cp
from jacobian_v1 import Jacobian
from scipy.spatial.transform import Rotation as R
Jacobian = Jacobian()

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):
    # 회전 행렬 생성 (ZYX 순: yaw → pitch → roll)
    rot = R.from_euler('xyz', [roll, pitch, yaw])
    R_mat = rot.as_matrix()  # 3x3 회전 행렬

    # 4x4 변환 행렬 구성
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_xyzrpy(T):
    assert T.shape == (4, 4)
    x, y, z = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz')
    return x, y, z, roll, pitch, yaw
# ====== 가정된 값들 ======
# 베이스 + 매니퓰레이터 총 자유도
n_dof = 8  # base(2) + arm(6)

# 현재 조인트 상태 (예: 센서에서 읽은 값)
q = np.array([0.0, 0.0, 1.2, -0.5, 0.3, 0.0, 1.2, 0.0])  # 예시값


# 현재 ee 위치
p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.2])  # 예시값 (x, y, z, ax, ay, az)
H_current = xyzrpy_to_matrix(p[0], p[1], p[2], p[3], p[4], p[5])  # 현재 위치 변환 행렬 (4x4)
# 목표 위치
p_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # 예시값 (x, y, z, ax, ay, az)
H_desired = xyzrpy_to_matrix(p_desired[0], p_desired[1], p_desired[2], p_desired[3], p_desired[4], p_desired[5])  # 목표 위치 변환 행렬 (4x4)

# 전체 자코비안 J (6x9)
# J = [ J_base | J_arm ]  (예: 엔드이펙터 속도 = J * x)
J_arm = Jacobian.jacobian( 0.0, 0.1, 0.5, 0., 0., 0.)  # 예시용 무작위 자코비안 (실제 FK에서 계산해야 함)
J_arm_v = J_arm[:3, :]  # 3x6 자코비안 (선형 속도)
J_arm_w = J_arm[3:, :]  # 3x6 자코비안 (각속도)

J_base = np.zeros((6, 2))  # 베이스 자코비안 (예시)
J = np.hstack((J_base, J_arm))  # 6x9 자코비안 행렬

# Hessian matrix의 translational component 생성
H = np.zeros((6, 3, 6))  # Hessian 행렬 초기화

for i in range(6):
    for j in range(6):
        a, b = min(i, j), max(i, j)  # 작은 값이 a, 큰 값이 b
        H[i, :, j] = np.cross(J_arm_w[:, a], J_arm_v[:, b]) # 외적 계산
# manipulability
m = J_arm_v @ J_arm_v.T 
m_det = np.linalg.det(m)  
m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

# ====== QP 구성 ======
x = cp.Variable(n_dof+6)  # joint velocity (n+6)
# s = cp.Variable(6) # slack variable (optional, for soft constraints)
# euqlidian distance
e = np.sqrt(np.sum((p - p_desired)**2))  # Euclidean distance (예시)
k_a = 0.01
beta = 1
e_base_distance = np.sqrt(np.sum((p[:2] - p_desired[:2])**2))  # 베이스 위치의 유클리드 거리 (예시)
e_base_theta = np.sqrt(np.sum((p[2:] - p_desired[2:])**2))  # 베이스 회전의 유클리드 거리 (예시)
# 각 행마다 곱할 값
scale = np.array([e_base_theta, e_base_distance, k_a, k_a, k_a, k_a, k_a, k_a]).reshape(-1, 1)  # shape (8, 1)

Q_1 = np.hstack((scale*np.eye(n_dof), np.zeros((n_dof, 6))))
Q_2 = np.hstack((np.zeros((6,n_dof)), 1/e*np.eye(6))) 

Q = np.vstack((Q_1, Q_2))  # QP 행렬 (예시)

# Hessian matrix의 translational component 생성
H = np.zeros((6, 3, 6))  # Hessian 행렬 초기화

for i in range(6):
    for j in range(6):
        a, b = min(i, j), max(i, j)  # 작은 값이 a, 큰 값이 b
        H[i, :, j] = np.cross(J_arm_w[:, a], J_arm_v[:, b]) # 외적 계산

# manipulability
m = J_arm_v @ J_arm_v.T 
m_det = np.linalg.det(m)  
m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

A = ((J_arm_v @ H[0, :, :].T).reshape(-1, order='F')).T
JJ_inv = np.linalg.pinv((J_arm_v @ J_arm_v.T)).reshape(-1, order='F')

J_m_T = np.array([[m_t*((J_arm_v @ H[0, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                 [m_t*((J_arm_v @ H[1, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                 [m_t*((J_arm_v @ H[2, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                 [m_t*((J_arm_v @ H[3, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                 [m_t*((J_arm_v @ H[4, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                 [m_t*((J_arm_v @ H[5, :, :].T).reshape(-1, order='F')).T @ JJ_inv]])

J_m_hat = np.vstack((np.zeros((2,1)), J_m_T))
J_ = np.hstack((J, np.eye(6)))  # J_ 행렬 (예시)

k_e = 0.5
theta_e = np.array([-k_e*np.rad2deg(q[2]-q[1])])
epsilon = np.vstack((theta_e, np.zeros((n_dof-1,1))))

C = np.vstack(((J_m_hat+epsilon), np.zeros((6, 1))))

eta = 1
rho_i = 50 #influence distance
rho_s = 2 #safety factor
# Joint limits (예시)
joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])        # 상한

# 현재 조인트 상태와 joint limit 비교
rho = np.minimum(q - joint_limits_lower, joint_limits_upper - q)
A = np.ones(shape=(n_dof,n_dof+6), dtype=np.float16)  # 예시용 제약조건 행렬
B = np.array([0., 0., eta * (rho[2]-rho_s)/(rho_i-rho_s), eta * (rho[3]-rho_s)/(rho_i-rho_s), eta * (rho[4]-rho_s)/(rho_i-rho_s), eta * (rho[5]-rho_s)/(rho_i-rho_s), eta * (rho[6]-rho_s)/(rho_i-rho_s), eta * (rho[7]-rho_s)/(rho_i-rho_s)])  # 예시용 제약조건 벡터

T = np.linalg.inv(H_current) @ H_desired  # 변환 행렬 (4x4)
slack = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 슬랙 변수 (예시)
v_desired_star = beta * matrix_to_xyzrpy(T)
v_desired = v_desired_star - slack
# 5. 목적함수
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + C.T @ x)
# 예시 제약조건 (속도 제한 등)
constraints = [
    x >= -1.0,
    x <= 1.0,
    # cp.abs(s) <= 0.1,  # 슬랙이 너무 커지는 걸 방지 (선택 사항)
    A @ x <= B,  # 예시 제약조건 (속도 제한 등)
    J_ @ x == v_desired,  # 엔드이펙터 속도 추종
]

# 풀기
prob = cp.Problem(objective, constraints)
prob.solve()

# 결과
x_opt = x.value
v_base = x_opt[:2]
q_dot = x_opt[2:8]

print("Optimal base velocity:", v_base)
print("Optimal joint velocity:", q_dot)