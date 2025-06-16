import numpy as np
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
from math import atan2 as atan2
import roboticstoolbox as rtb
from spatialmath import base, SE3
import qpsolvers as qp


class QP_UR5e(Node):
    def __init__(self):
        super().__init__('QP_UR5e')
        self.ROBOT_IP = '192.168.0.3'
        self.robot = rtb.models.UR5()
        # self.k_a = 0.01
        # self.beta = 1
        self.n_dof = 6  # base(2) + arm(6)
        self.k_e = 0.5
        self.rho_i = 0.9 # influence distance
        self.rho_s = 0.1  # safety factor
        self.eta = 1
        self.q = []
        # self.slack = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 슬랙 변수 (예시)
        # self.q_desired = np.array([0.5, -1.57, 1.57, 0.0, 1.57, 0.0])  # 목표 조인트 각도 (예시)
        # Hessian matrix의 translational component 생성
        self.H_desired = None
        # self.H_trans = np.zeros((6, 3, 6))  # Hessian 행렬 초기화
        self.q_initialized = False
        # self.p_initialized = False
        self.qdlim = np.array([0.03]*6)
        # self.joint_limits_lower = np.array([0.0, 0.0, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14])  # 하한
        # self.joint_limits_upper = np.array([0.0, 0.0, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])  # 상한
        self.qlim = np.array([[-3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265],
                               [ 3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265,  3.14159265]])#np.vstack([self.joint_limits_lower, self.joint_limits_upper])
        # self.a = np.array([0., -0.425, -0.392, 0., 0., 0.])
        # self.d = np.array([0.163,0.,0.,0.133,0.100,0.1])
        # self.alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub  # prevent unused variable warning
        # self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_callback, 10)
        self.q_dot_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.qp_solver)
    
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
        T = self.robot.fkine(self.q[2:8])  # UR5e FK 계산
        H_current = SE3(T) 
        if self.H_desired is None:
            H_desired = SE3(T)
            H_desired.A[:3, :3] = T.A[:3, :3]  # 현재 회전 행렬 유지
            H_desired.A[0, -1] -= 0.25
            H_desired.A[2, -1] -= 0.05
            self.H_desired = H_desired
        else:
            H_desired = self.H_desired
            print('desired: ', H_desired)
            print('current: ', T)

        # 전체 자코비안 J (6x9)
        J_arm = J = self.robot.jacobe(self.q[2:8])  # 베이스 프레임 기준 자코비안 (6x6)
        J_arm_v = J_arm[:3, :]  # 3x6 자코비안 (선형 속도)
        J_arm_w = J_arm[3:, :]  # 3x6 자코비안 (각속도)

        
        # bTe = self.robot.fkine(self.robot.q, include_base=False).A  # bTe와 H_current.A는 동일한 값

        T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

        et = np.sum(np.abs(T_error[:3, -1])) #+ np.exp(-16)  # Euclidean distance (예시)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.n_dof + 6)

        # Joint velocity component of Q
        Q[: self.n_dof, : self.n_dof] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[self.n_dof :, self.n_dof :] = (1.0 / et) * np.eye(6)

        H = np.zeros((self.n_dof, 6, self.n_dof))  # same as jacobm

        for j in range(self.n_dof):
            for i in range(j, self.n_dof):
                H[j, :3, i] = np.cross(J_arm_w[:, j], J_arm_v[:, i])
                H[j, 3:, i] = np.cross(J_arm_w[:, j], J_arm_w[:, i])
                if i != j:
                        H[i, :3, j] = H[j, :3, i]
                        H[i, 3:, j] = H[j, 3:, i]
        # manipulability
        m = J_arm @ J_arm.T 
        m_det = np.linalg.det(m)  
        m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        JJ_inv = np.linalg.pinv((J_arm @ J_arm.T)).reshape(-1, order='F')

        # Compute manipulability Jacobian
        J_m_T = np.zeros((self.n_dof, 1))
        for i in range(self.n_dof):
            c = J_arm @ H[i].T  # shape: (6,6)
            J_m_T[i] = m_t * c.flatten("F") @ JJ_inv

        # J_m_hat = np.vstack((np.zeros((2,1)), J_m_T))
        J_ = np.hstack((J, np.eye(6)))  # J_ 행렬 (예시)
        θε = atan2(T.A[1, -1], T.A[0, -1])
        epsilon = np.zeros((self.n_dof,1))
        epsilon[0] = - self.k_e * θε  # 베이스 x 위치 오차

        C = np.vstack(((-J_m_T+epsilon), np.zeros((6, 1))))
        # c = np.concatenate(
        #     (-self.robot.jacobm().reshape((self.n_dof,)), np.zeros(6))
        # )

        # Get base to face end-effector
        # kε = 0.5
        # bTe = self.robot.fkine(self.robot.q, include_base=False).A
        # θε = atan2(bTe[1, -1], bTe[0, -1])
        # ε = kε * θε
        # c[0] = -ε

        A = np.zeros((self.n_dof + 6, self.n_dof + 6))
        B = np.zeros(self.n_dof + 6)
        A[: self.n_dof, : self.n_dof], B[: self.n_dof] = self.joint_velocity_damper(ps=self.rho_s, pi=self.rho_i, n=self.n_dof, gain=self.eta)  # joint velocity damper
 

        eTep = np.linalg.inv(H_current) @ H_desired.A  # 현재 위치에서의 오차 행렬
        e = np.empty(6)

        # Translational error
        e[:3] = eTep[:3, -1]

        # Angular error
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        k = 1.5 * np.eye(6) # gain
        v = k @ e
        v[3:] *= 1.3
        


        lb = -np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.qdlim[: self.n_dof], 10 * np.ones(6)]
        # print('A:', A)
        # print('B:', B)
        qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
        qd = qd[: self.n_dof]
        print("qd:", qd)
        if qd is None:
            print("QP solution is None")
            qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) 

        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4

        if et < 0.02:
            qd *= 0.0 # 목표 위치에 도달했음을 나타냄

        q_dot = qd
        # # 결과를 퍼블리시
        q_dot_msg = Float64MultiArray()
        q_dot_msg.data = q_dot.tolist()
        self.q_dot_pub.publish(q_dot_msg)
        # # print("Optimal base velocity:", v_base)
        # # print("Optimal joint velocity:", q_dot)

def main(args=None):
    rclpy.init(args=args)
    listener = QP_UR5e()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown() 



print('Program started')

if __name__ == '__main__':

    main()