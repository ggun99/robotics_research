import numpy as np
import cvxpy as cp
from jacobian_v1 import Jacobian
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, WrenchStamped, PoseStamped

def ClassName(Node):
    def __init__(self):
        super().__init__('QP_UR5e')
        self.ROBOT_IP = '192.168.0.3'
        self.Jacobian = Jacobian()
        self.k_a = 0.01
        self.beta = 1
        self.n_dof = 8  # base(2) + arm(6)
        self.k_e = 0.5
        self.rho_i = 0.9  # influence distance
        self.rho_s = 0.1  # safety factor
        self.eta = 1
        self.q = []
        self.p = []
        self.slack = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 슬랙 변수 (예시)
        self.p_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Hessian matrix의 translational component 생성
        self.H = np.zeros((6, 3, 6))  # Hessian 행렬 초기화
        self.joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
        self.joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])  # 상한
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub  # prevent unused variable warning
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_callback, 10)
        self.q_dot_pub = self.create_publisher(Twist, '/forward_velocity_controller/commands', 10)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.qp_solver)

    def joint_callback(self, msg):
        # 조인트 상태 업데이트
        self.q = msg.position
        # self.q_dot = msg.velocity
    def tcp_callback(self, msg):
        # TCP 위치 업데이트
        self.p = msg.pose.position
        # self.p_orientation = msg.pose.orientation
        # self.p_velocity = msg.twist.linear
        # self.p_angular_velocity = msg.twist.angular

    def xyzrpy_to_matrix(self, x, y, z, roll, pitch, yaw):
        # 회전 행렬 생성 (ZYX 순: yaw → pitch → roll)
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        R_mat = rot.as_matrix()  # 3x3 회전 행렬

        # 4x4 변환 행렬 구성
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [x, y, z]
        return T

    def matrix_to_xyzrpy(self,T):
        assert T.shape == (4, 4)
        x, y, z = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        roll, pitch, yaw = rot.as_euler('xyz')
        return x, y, z, roll, pitch, yaw

    def qp_solver(self):
        H_current = self.xyzrpy_to_matrix(self.p[0], self.p[1], self.p[2], self.p[3], self.p[4], self.p[5])  # 현재 위치 변환 행렬 (4x4)
        H_desired = self.xyzrpy_to_matrix(self.p_desired[0], self.p_desired[1], self.p_desired[2], self.p_desired[3], self.p_desired[4], self.p_desired[5])  # 목표 위치 변환 행렬 (4x4)

        # 전체 자코비안 J (6x9)
        J_arm = Jacobian.jacobian( self.q[0], self.q[1], self.q[2], self.q[3], self.q[4], self.q[5])  # 예시용 무작위 자코비안 (실제 FK에서 계산해야 함)
        J_arm_v = J_arm[:3, :]  # 3x6 자코비안 (선형 속도)
        J_arm_w = J_arm[3:, :]  # 3x6 자코비안 (각속도)

        J_base = np.zeros((6, 2))  # 베이스 자코비안 (예시)
        J = np.hstack((J_base, J_arm))  # 6x9 자코비안 행렬

        for i in range(6):
            for j in range(6):
                a, b = min(i, j), max(i, j)  # 작은 값이 a, 큰 값이 b
                self.H[i, :, j] = np.cross(J_arm_w[:, a], J_arm_v[:, b]) # 외적 계산
        # manipulability
        m = J_arm_v @ J_arm_v.T 
        m_det = np.linalg.det(m)  
        m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        # ====== QP 구성 ======
        x = cp.Variable(self.n_dof+6)  # joint velocity (n+6)
        # euqlidian distance
        e = np.sqrt(np.sum((self.p - self.p_desired)**2))  # Euclidean distance (예시)
        e_base_distance = np.sqrt(np.sum((self.p[:2] - self.p_desired[:2])**2))  # 베이스 위치의 유클리드 거리 (예시)
        e_base_theta = np.sqrt(np.sum((self.p[2:] - self.p_desired[2:])**2))  # 베이스 회전의 유클리드 거리 (예시)
        # 각 행마다 곱할 값
        scale = np.array([e_base_theta, e_base_distance, self.k_a, self.k_a, self.k_a, self.k_a, self.k_a, self.k_a]).reshape(-1, 1)  # shape (8, 1)

        Q_1 = np.hstack((scale*np.eye(self.n_dof), np.zeros((self.n_dof, 6))))
        Q_2 = np.hstack((np.zeros((6,self.n_dof)), 1/e*np.eye(6))) 

        Q = np.vstack((Q_1, Q_2))  # QP 행렬 (예시)

        for i in range(6):
            for j in range(6):
                a, b = min(i, j), max(i, j)  # 작은 값이 a, 큰 값이 b
                self.H[i, :, j] = np.cross(J_arm_w[:, a], J_arm_v[:, b]) # 외적 계산

        # manipulability
        m = J_arm_v @ J_arm_v.T 
        m_det = np.linalg.det(m)  
        m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        A = ((J_arm_v @ self.H[0, :, :].T).reshape(-1, order='F')).T
        JJ_inv = np.linalg.pinv((J_arm_v @ J_arm_v.T)).reshape(-1, order='F')

        J_m_T = np.array([[m_t*((J_arm_v @ self.H[0, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                        [m_t*((J_arm_v @ self.H[1, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                        [m_t*((J_arm_v @ self.H[2, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                        [m_t*((J_arm_v @ self.H[3, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                        [m_t*((J_arm_v @ self.H[4, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
                        [m_t*((J_arm_v @ self.H[5, :, :].T).reshape(-1, order='F')).T @ JJ_inv]])

        J_m_hat = np.vstack((np.zeros((2,1)), J_m_T))
        J_ = np.hstack((J, np.eye(6)))  # J_ 행렬 (예시)

        theta_e = np.array([-self.k_e*np.rad2deg(self.q[2]-self.q[1])])
        epsilon = np.vstack((theta_e, np.zeros((self.n_dof-1,1))))

        C = np.vstack(((J_m_hat+epsilon), np.zeros((6, 1))))
        # Joint limits (예시)
        joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
        joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])        # 상한

        # 현재 조인트 상태와 joint limit 비교
        rho = np.minimum(self.q - joint_limits_lower, joint_limits_upper - self.q)
        A = np.ones(shape=(self.n_dof,self.n_dof+6), dtype=np.float16)  # 예시용 제약조건 행렬
        B = np.array([0., 0., 
                      self.eta * (rho[2]-self.rho_s)/(self.rho_i-self.self.rho_s), 
                      self.eta * (rho[3]-self.self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[4]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[5]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[6]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[7]-self.rho_s)/(self.rho_i-self.rho_s)])  # 예시용 제약조건 벡터

        T = np.linalg.inv(H_current) @ H_desired  # 변환 행렬 (4x4)
        v_desired_star = self.beta * matrix_to_xyzrpy(T)
        v_desired = v_desired_star - self.slack
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
        # 결과를 퍼블리시
        q_dot_msg = Twist()
        q_dot_msg.linear.x = q_dot[0]
        q_dot_msg.linear.y = q_dot[1]
        q_dot_msg.linear.z = q_dot[2]
        q_dot_msg.angular.x = q_dot[3]
        q_dot_msg.angular.y = q_dot[4]
        q_dot_msg.angular.z = q_dot[5]
        print("Optimal base velocity:", v_base)
        print("Optimal joint velocity:", q_dot)