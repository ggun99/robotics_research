import numpy as np
import osqp
from scipy import sparse

# 로봇 파라미터
l1, l2 = 1.0, 1.0  # 링크 길이

# 현재 관절각 (예시)
q = np.array([np.pi/4, np.pi/4])  # 45도, 45도

# Jacobian 계산
J = np.array([
    [-l1*np.sin(q[0]) - l2*np.sin(q[0]+q[1]), -l2*np.sin(q[0]+q[1])],
    [l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]), l2*np.cos(q[0]+q[1])]
])

# 목표 엔드이펙터 속도 (x 방향으로 1 m/s, y 방향 0)
xdot_desired = np.array([1.0, 0.0])

# QP 문제 설정
H = J.T @ J  # 목적함수 (이차항)
f = -J.T @ xdot_desired  # 목적함수 (선형항)

# 관절속도 제한
v_max = 2.0  # rad/s

A = np.vstack([np.eye(2), -np.eye(2)])  # 속도 상한, 하한
b = np.hstack([v_max*np.ones(2), v_max*np.ones(2)])  # 상한값 (하한은 -v_max)

# OSQP Solver 설정
prob = osqp.OSQP()
prob.setup(P=sparse.csc_matrix(H), q=f, A=sparse.csc_matrix(A), l=None, u=b, verbose=False)

# QP 풀기
res = prob.solve()

# 결과
dq = res.x
print("계산된 관절속도 dq:", dq)