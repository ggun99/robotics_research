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
        self.n_dof = 6  # base(2) + arm(6)
        self.q = []
        self.robot.qdlim = np.array([0.03]*6)
        self.arrived = False 
        self.dt = 0.025
        self.slack = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # 슬랙 변수 (예시)
        self.q_desired = np.array([0.5, -1.57, 1.57, 0.0, 1.57, 0.0])  # 목표 조인트 각도 (예시)
        self.H_desired = None
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub  # prevent unused variable warning
        # self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_callback, 10)
        self.q_dot_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.qp_solver)
        self.q_initialized = False
   
    def normalize_rotation(self, R):
        # Orthonormalize R using SVD
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt
    
    def joint_callback(self, msg):
        # 조인트 상태 업데이트
        q_origin = msg.position
        q_corrected = [q_origin[5], q_origin[0], q_origin[1], q_origin[2], q_origin[3], q_origin[4]]
        self.q = np.array(q_corrected)  # 조인트 각도 (예시)
        self.q_initialized = True
        self.robot.q = self.q  # 로봇 모델에 조인트 각도 설정
 

    
    def qp_solver(self):
        if not self.q_initialized:
            print("Waiting for joint states and TCP position...")
            return
        T = self.robot.fkine(self.robot.q)  # UR5e FK 계산

        if self.H_desired is None:
            H_desired = T
            H_desired.A[:3, :3] = T.A[:3, :3]  # 현재 회전 행렬 유지
            H_desired.A[0, -1] -= 0.25
            H_desired.A[2, -1] -= 0.05
            self.H_desired = H_desired
        else:
            H_desired = self.H_desired
            print('desired: ', H_desired)
            print('current: ', T)

        eTep = np.linalg.inv(T) @ H_desired.A  # 현재 위치에서의 오차 행렬
        et = np.sum(np.abs(eTep[:3, -1]))
        # Gain term (lambda) for control minimisation
        Y = 0.01
        # Quadratic component of objective function
        Q = np.eye(self.n_dof + 6)

        # Joint velocity component of Q
        Q[: self.n_dof, : self.n_dof] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[self.n_dof :, self.n_dof :] = (1.0 / et) * np.eye(6)
        # print(Q)

        v, _ = rtb.p_servo(T, H_desired.A, 1.5)

        v[3:] *= 1.3

        # The equality contraints
        robjac = self.robot.jacobe(self.robot.q)  # UR5e 자코비안 계산
        print("==========================")
        # print(robjac.shape)

        Aeq = np.c_[robjac, np.eye(6)]
        # print('Aeq.shape', Aeq.shape)
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.n_dof + 6, self.n_dof + 6))
        bin = np.zeros(self.n_dof + 6)


        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.1

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[: self.n_dof, : self.n_dof], bin[: self.n_dof] = self.robot.joint_velocity_damper(ps, pi, self.n_dof)

        c = np.concatenate(
            (-self.robot.jacobm().reshape((self.n_dof,)), np.zeros(6))
        )
        # Get base to face end-effector
        kε = 0.5
        bTe = self.robot.fkine(self.robot.q, include_base=False).A
        θε = atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[0] = -ε
        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.robot.qdlim[: self.n_dof], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.n_dof], 10 * np.ones(6)]
        # print('Ain', Ain)
        # print('bin', bin)
        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
        qd = qd[: self.n_dof]
        print("qd:", qd)
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4

        if et < 0.02:
            qd *= 0.0 # 목표 위치에 도달했음을 나타냄
        
        if qd is None:
            print("QP solution is None")
            qd = np.array([0.,0.,0.0,0.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])  # 기본값으로 초기화
        q_dot = qd
        # 결과를 퍼블리시
        q_dot_msg = Float64MultiArray()
        q_dot_msg.data = q_dot.tolist()
        self.q_dot_pub.publish(q_dot_msg)

def main(args=None):
    rclpy.init(args=args)
    listener = QP_UR5e()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown() 



print('Program started')

if __name__ == '__main__':

    main()