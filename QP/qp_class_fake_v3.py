import numpy as np
import cvxpy as cp
from jacobian_v1 import Jacobian
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, WrenchStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
from math import atan2 as atan2
import roboticstoolbox as rtb
from spatialmath import base, SE3
import qpsolvers as qp
import spatialmath as sm


class QP_UR5e(Node):
    def __init__(self):
        super().__init__('QP_UR5e')
        self.ROBOT_IP = '192.168.0.3'
        self.Jacobian = Jacobian()
        self.robot = rtb.models.UR5()
        self.k_a = 0.01
        self.beta = 1
        self.n_dof = 6  # base(2) + arm(6)
        self.k_e = 0.5
        self.rho_i = 0.872665  # influence distance
        self.rho_s = 0.03  # safety factor
        self.eta = 1
        self.q = []
        self.slack = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 슬랙 변수 (예시)
        self.q_desired = np.array([0.5, -1.57, 1.57, 0.0, 1.57, 0.0])  # 목표 조인트 각도 (예시)
        # Hessian matrix의 translational component 생성
        self.H_desired = None
        self.H_trans = np.zeros((6, 3, 6))  # Hessian 행렬 초기화
        self.q_initialized = False
        self.p_initialized = False
        self.joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
        self.joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])  # 상한
        self.qlim = np.vstack([self.joint_limits_lower, self.joint_limits_upper])
        self.a = np.array([0., -0.425, -0.392, 0., 0., 0.])
        self.d = np.array([0.163,0.,0.,0.133,0.100,0.1])
        self.alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub  # prevent unused variable warning
        # self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_callback, 10)
        self.q_dot_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.qp_solver)

    def normalize_rotation(self, R):
        # Orthonormalize R using SVD
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt
    
    def joint_callback(self, msg):
        # 조인트 상태 업데이트
        q_origin = msg.position
        q_corrected = [q_origin[5], q_origin[0], q_origin[1], q_origin[2], q_origin[3], q_origin[4]]
        self.q = np.array(q_corrected)  # 조인트 각도 (예시)
        self.q = np.vstack(( np.zeros((2,1)), self.q.reshape(-1,1)))  # 베이스 조인트 추가 (예시)
        self.q_initialized = True
        self.q = self.q.ravel() 
        self.robot.q = self.q[2:]  # 로봇 모델에 조인트 각도 설정
        # self.q_dot = msg.velocity

    def dh_transform(self, a, alpha, d, theta):
        """Denavit-Hartenberg 변환 행렬 생성 함수"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    # def ur5e_forward_kinematics_SE3(self, joints):
    #     """
    #     UR5e 6-DOF 순방향 기구학 계산 (SE3 사용 버전)
    #     joints: 6개의 조인트 각도 [rad]
    #     return: SE3 객체 (end-effector pose)
    #     """
    #     # UR5e 공식 DH 파라미터 (단위: m)
    #     dh_params = [
    #         (0,        np.pi/2,  0.1625, joints[0]),
    #         (-0.425,   0,        0,      joints[1]),
    #         (-0.3922,  0,        0,      joints[2]),
    #         (0,        np.pi/2,  0.1333, joints[3]),
    #         (0,       -np.pi/2,  0.0997, joints[4]),
    #         (0,        0,        0.0996, joints[5])
    #     ]

    #     T = SE3()
    #     for a, alpha, d, theta in dh_params:
    #         T *= SE3.DH(a=a, alpha=alpha, d=d, theta=theta)

    #     return T  # SE3 객체 리턴
    
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

    def se3_to_vec(self, se3_mat):
        omega = np.array([se3_mat[2,1], se3_mat[0,2], se3_mat[1,0]])
        v = se3_mat[0:3, 3]
        return np.concatenate([v, omega]) 
    
    from typing import (
        List,
        TypeVar,
        Union,
        Dict,
        Tuple,
        overload,
    )
    def joint_velocity_damper(
        self,
        ps: float = 0.05,
        pi: float = 0.1,
        n: int = 8,
        gain: float = 1.0,
    ):
        """
        Compute the joint velocity damper for QP motion control

        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into joint limits. Requires
        the joint limits of the robot to be specified. See examples/mmc.py
        for use case

        Attributes
        ----------
        ps
            The minimum angle (in radians) in which the joint is
            allowed to approach to its limit
        pi
            The influence angle (in radians) in which the velocity
            damper becomes active
        n
            The number of joints to consider. Defaults to all joints
        gain
            The gain for the velocity damper

        Returns
        -------
        Ain
            A (6,) vector inequality contraint for an optisator
        Bin
            b (6,) vector inequality contraint for an optisator

        """

        Ain = np.zeros((n, n))
        Bin = np.zeros(n)

        for i in range(n):
            if self.q[i] - self.qlim[0, i] <= pi:
                Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.q[i] <= pi:
                Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin
    
    def qp_solver(self):
        if not self.q_initialized:
            print("Waiting for joint states and TCP position...")
            return
        # print(self.q)
        # T = self.ur5e_forward_kinematics(self.q[2:8])  # UR5e FK 계산
        T = self.robot.fkine(self.q[2:8])  # UR5e FK 계산
        # print("T:", T)
        H_current = SE3(T) 
        if self.H_desired is None:
            H_desired = SE3(T)
            H_desired.A[:3, :3] = H_current.A[:3, :3]  # 현재 회전 행렬 유지
            H_desired.A[0, -1] += 0.3
            # H_desired.A[2, -1] -= 0.1
            self.H_desired = H_desired
        else:
            H_desired = self.H_desired

        # 전체 자코비안 J (6x9)
        
        # J_arm = self.Jacobian.jacobian( self.q[2], self.q[3], self.q[4], self.q[5], self.q[6], self.q[7])  # 예시용 무작위 자코비안 (실제 FK에서 계산해야 함)
        J_arm = J = self.robot.jacobe(self.q[2:8])  # 베이스 프레임 기준 자코비안 (6x6)
        # J_arm_v = J_arm[:3, :]  # 3x6 자코비안 (선형 속도)
        # J_arm_w = J_arm[3:, :]  # 3x6 자코비안 (각속도)
        # T = self.ur5e_forward_kinematics   # 4x4 변환 행렬
        # J_arm = base.tr2jac(T.T) @ J_arm  # 현재 위치에서의 자코비안 변환
        
        # J_base = np.zeros((6, 2))  # 베이스 자코비안 (예시)
        # J = np.hstack((J_base, J_arm))  # 6x8 자코비안 행렬
        J = J_arm
        # ====== QP 구성 ======
        x = cp.Variable(self.n_dof+6)  # joint velocity (n+6)
        # euqlidian distance
        print("H_current:", H_current.A)
        bTe = self.robot.fkine(self.robot.q, include_base=False).A
        print("bTe:", bTe)
        # print("H_desired:", H_desired.A)
        T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

        et = np.sum(np.abs(T_error[:3, -1])) + np.exp(-16)  # Euclidean distance (예시)
        
        # Gain term (lambda) for control minimisation
        Y = 100.0

        # Quadratic component of objective function
        Q = np.eye(self.n_dof + 6)

        # Joint velocity component of Q
        Q[: self.n_dof, : self.n_dof] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[self.n_dof :, self.n_dof :] = (1.0 / et) * np.eye(6)

        # for i in range(6):
        #     for j in range(6):
        #         a, b = min(i, j), max(i, j)  # 작은 값이 a, 큰 값이 b
        #         self.H_trans[i, :, j] = np.cross(J_arm_w[:, a], J_arm_v[:, b]) # 외적 계산

        # manipulability
        # m = J_arm_v @ J_arm_v.T 
        # m_det = np.linalg.det(m)  
        # m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        # JJ_inv = np.linalg.pinv((J_arm_v @ J_arm_v.T)).reshape(-1, order='F')

        # J_m_T = np.array([[m_t*((J_arm_v @ self.H_trans[0, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
        #                 [m_t*((J_arm_v @ self.H_trans[1, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
        #                 [m_t*((J_arm_v @ self.H_trans[2, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
        #                 [m_t*((J_arm_v @ self.H_trans[3, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
        #                 [m_t*((J_arm_v @ self.H_trans[4, :, :].T).reshape(-1, order='F')).T @ JJ_inv],
        #                 [m_t*((J_arm_v @ self.H_trans[5, :, :].T).reshape(-1, order='F')).T @ JJ_inv]])

        # J_m_hat = np.vstack((np.zeros((2,1)), J_m_T))
        J_ = np.hstack((J, np.eye(6)))  # J_ 행렬 (예시)
        # J_ = base.tr2jac(T.T) @ J_  # 현재 위치에서의 자코비안 변환
        θε = atan2(T.A[1, -1], T.A[0, -1])
        epsilon = np.zeros((self.n_dof,1))
        epsilon[0] = - self.k_e * θε  # 베이스 x 위치 오차
        # C = np.vstack(((J_m_hat+epsilon), np.zeros((6, 1))))
   
        A = np.zeros((self.n_dof + 6, self.n_dof + 6))
        B = np.zeros(self.n_dof + 6)
        A[: self.n_dof, : self.n_dof], B[: self.n_dof] = self.joint_velocity_damper(ps=self.rho_s, pi=self.rho_i, n=self.n_dof, gain=self.eta)  # joint velocity damper
        # print("A:", A)
        # print("B:", B)

        v, _ = rtb.p_servo(H_current, H_desired, 1.0)
        # print("v:", v)
        # 5. 목적함수
        # s = cp.Variable(6)
        objective = cp.Minimize(0.5 * cp.quad_form(x, Q)) # + C.T @ x )#  + 1000 * cp.sum_squares(s)) 
        # 예시 제약조건 (속도 제한 등)
        constraints = [
            x >= -1.5,
            x <= 1.5,
            # cp.abs(s) <= 0.01,  # 슬랙이 너무 커지는 걸 방지 (선택 사항)
            A @ x <= B,  # 예시 제약조건 (속도 제한 등)
            J_ @ x == v,  # 엔드이펙터 속도 추종
        ]

        # lb = np.r_[self.joint_limits_lower, -10 * np.ones(6)]
        # ub = np.r_[self.joint_limits_upper, 10 * np.ones(6)]
        # qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
        # print("qd:", qd)
        # # 풀기
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=True)  # OSQP 솔버 사용

        # 결과
        x_opt = x.value
        if x_opt is None:
            print("QP solution is None")
            x_opt = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])  # 기본값으로 초기화
        v_base = x_opt[:2]
        q_dot = x_opt[2:8]
        # q_dot = qd[2:8]
        # 결과를 퍼블리시
        q_dot_msg = Float64MultiArray()
        q_dot_msg.data = q_dot.tolist()
        self.q_dot_pub.publish(q_dot_msg)
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