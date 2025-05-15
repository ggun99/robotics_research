import numpy as np
import cvxpy as cp
from jacobian_v1 import Jacobian
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, WrenchStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt

class QP_UR5e(Node):
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
        self.slack = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 슬랙 변수 (예시)
        self.p_desired = np.array([-0.3, -0.126, 0.48, -0.2, 0.97, 0.0])
        # Hessian matrix의 translational component 생성
        self.H = np.zeros((6, 3, 6))  # Hessian 행렬 초기화
        self.q_initialized = False
        self.p_initialized = False
        self.joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
        self.joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])  # 상한
        # self.d1 =  0.163
        # self.a2 = -0.425
        # self.a3 = -0.392
        # self.d4 =  0.133
        # self.d5 =  0.1
        # self.d6 =  0.1
        self.a = np.array([0., -0.425, -0.392, 0., 0., 0.])
        self.d = np.array([0.163,0.,0.,0.133,0.100,0.1])
        self.alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub  # prevent unused variable warning
        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_callback, 10)
        self.q_dot_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.qp_solver)


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
    
    def joint_callback(self, msg):
        # 조인트 상태 업데이트
        q_origin = msg.position
        q_corrected = [q_origin[5], q_origin[0], q_origin[1], q_origin[2], q_origin[3], q_origin[4]]
        self.q = np.array(q_corrected)  # 조인트 각도 (예시)
        self.q = np.vstack(( np.zeros((2,1)), self.q.reshape(-1,1)))  # 베이스 조인트 추가 (예시)
        self.q_initialized = True
        self.q = self.q.ravel() 
        # self.q_dot = msg.velocity

    def tcp_callback(self, msg):
        # TCP 위치 업데이트
        position = msg.pose.position
        orientation = msg.pose.orientation
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler = r.as_euler('xyz')
        self.p = [position.x, position.y, position.z, euler[0], euler[1], euler[2]]  # [x, y, z, roll, pitch, yaw]
        self.p_initialized = True
        # self.p_orientation = msg.pose.orientation
        # self.p_velocity = msg.twist.linear
        # self.p_angular_velocity = msg.twist.angular
    def dh_transform(self, a, alpha, d, theta):
        """Denavit-Hartenberg 변환 행렬 생성 함수"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def ur5e_forward_kinematics(self, joints):
        """
        UR5e 6-DOF 순방향 기구학 계산
        joints: 6개의 조인트 각도(rad] 리스트
        return: 4x4 end-effector 변환 행렬
        """

        # UR5e 공식 DH 파라미터 (단위: m)
        dh_params = [
            (0,        np.pi/2,  0.1625, joints[0]),
            (-0.425,   0,        0,      joints[1]),
            (-0.3922,  0,        0,      joints[2]),
            (0,        np.pi/2,  0.1333, joints[3]),
            (0,       -np.pi/2,  0.0997, joints[4]),
            (0,        0,        0.0996, joints[5])
        ]

        T = np.identity(4)
        for a, alpha, d, theta in dh_params:
            T = np.dot(T, self.dh_transform(a, alpha, d, theta))
  
        return T
    
    def qp_solver(self):
        if not self.q_initialized or not self.p_initialized:
            print("Waiting for joint states and TCP position...")
            return
        T = self.ur5e_forward_kinematics(self.q[2:8])  # UR5e FK 계산
        # print('T:', T)
        x_,y_,z_,roll,pitch,yaw = self.matrix_to_xyzrpy(T)
        print('real_ros  :', self.p)
        print('calculated:',x_,y_,z_,roll,pitch,yaw)
        H_current = self.xyzrpy_to_matrix(self.p[0], self.p[1], self.p[2], self.p[3], self.p[4], self.p[5])  # 현재 위치 변환 행렬 (4x4)
        H_desired = self.xyzrpy_to_matrix(self.p_desired[0], self.p_desired[1], self.p_desired[2], self.p_desired[3], self.p_desired[4], self.p_desired[5])  # 목표 위치 변환 행렬 (4x4)

        # 전체 자코비안 J (6x9)
        J_arm = self.Jacobian.jacobian( self.q[0], self.q[1], self.q[2], self.q[3], self.q[4], self.q[5])  # 예시용 무작위 자코비안 (실제 FK에서 계산해야 함)
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

        # 현재 조인트 상태와 joint limit 비교
        rho = np.minimum(self.q - self.joint_limits_lower, self.joint_limits_upper - self.q)
        A = np.ones(shape=(self.n_dof,self.n_dof+6), dtype=np.float16)  # 예시용 제약조건 행렬
        B = np.array([0., 0., 
                      self.eta * (rho[2]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[3]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[4]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[5]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[6]-self.rho_s)/(self.rho_i-self.rho_s), 
                      self.eta * (rho[7]-self.rho_s)/(self.rho_i-self.rho_s)])  # 예시용 제약조건 벡터

        T = np.linalg.inv(H_current) @ H_desired  # 변환 행렬 (4x4)
        v_desired_star = self.beta * self.matrix_to_xyzrpy(T)
        v_desired = v_desired_star - self.slack
        # 5. 목적함수
        objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + C.T @ x)
        # 예시 제약조건 (속도 제한 등)
        constraints = [
            x >= -1.5,
            x <= 1.5,
            # cp.abs(s) <= 0.1,  # 슬랙이 너무 커지는 걸 방지 (선택 사항)
            A @ x <= B,  # 예시 제약조건 (속도 제한 등)
            J_ @ x == v_desired,  # 엔드이펙터 속도 추종
        ]

        # 풀기
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 결과
        x_opt = x.value
        # print("Problem status:", prob.status)
        # print("A shape:", A.shape)
        # print("B shape:", B.shape)
        # print("J_ shape:", J_.shape)
        # print("v_desired shape:", v_desired.shape)
        # print("QP solution:", x_opt)
        # print("v_desired:", v_desired)
        v_base = x_opt[:2]
        q_dot = x_opt[2:8]
        # 결과를 퍼블리시
        q_dot_msg = Float64MultiArray()
        q_dot_msg.data = q_dot.tolist()
        # self.q_dot_pub.publish(q_dot_msg)
        # print("Optimal base velocity:", v_base)
        # print("Optimal joint velocity:", q_dot)

def main(args=None):
    rclpy.init(args=args)
    listener = QP_UR5e()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown() 



print('Program started')

if __name__ == '__main__':

    main()