import numpy as np
import qpsolvers as qp
from spatialmath import SE3, base
import roboticstoolbox as rtb
from math import atan2
from scipy.spatial.transform import Rotation as R

class QP:
    def __init__(self, DOF, qdlim, q, x, y, z, quat, ur5e_robot, d_safe, d_influence, k_e, obstacles_positions, obstacle_radius, xform_pose):
        self.n_dof = DOF
        self.lb = -np.r_[qdlim[: self.n_dof], 10 * np.ones(6)]
        self.ub = np.r_[qdlim[: self.n_dof], 10 * np.ones(6)]
        self.q = q
        self.x = x
        self.y = y
        self.z = z
        self.quat = quat
        self.ur5e_robot = ur5e_robot
        self.d_safe = d_safe
        self.d_influence = d_influence
        self.k_e = k_e
        self.obstacles_positions = obstacles_positions
        self.obstacle_radius = obstacle_radius
        self.xform_pose = xform_pose

    def solve(self, desired_position, human_position):

        sb_rot = R.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]]).as_matrix()  # 베이스 프레임 회전 행렬
        T_sb = np.eye(4)
        T_sb[0,3] = self.x
        T_sb[1,3] = self.y
        T_sb[2,3] = self.z
        T_sb[:3, :3] = sb_rot  # 베이스 프레임 회전 행렬
        T_b0 = np.eye(4)
        T_b0[0,3] = 0.1315 # 0.1015
        # T_b0[1,3] = 
        T_b0[2,3] = 0.51921  # 0.47921
        T_0e = self.ur5e_robot.fkine(self.q[2:]).A
        print("T: ", T_sb @ T_b0 @ T_0e)
        T = T_b0 @ T_0e  # 베이스 프레임 기준 end-effector 위치
        H_current = SE3(T)  # 현재 end-effector 위치
        T_sd = np.eye(4)
        T_sd[:3, 3] = desired_position

        num_points = 10
        points_between, dist_vec = self.generate_points_between_positions(human_position, desired_position, num_points)
        xform_pose = np.vstack((self.xform_pose, points_between))

        distances = [np.linalg.norm(desired_position - obs)-self.obstacle_radius for obs in self.obstacles_positions]
        min_distance = np.min(distances)
        if min_distance < self.d_safe:
            print("Desired position is too close to an obstacle. Adjusting position.")
            

        T_bd = np.linalg.inv(T_sb) @ T_sd  

        H_desired = SE3(T_bd)  # 목표 end-effector 위치 

        F = np.array([[0.0, 1.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0], 
                        [0.0, 0.0],
                        [1.0, 0.0]])
        
        J_p = base.tr2adjoint(T.T) @ F  # 6x2 자코비안 (선형 속도)
        J_a_e = base.tr2adjoint(T.T) @ self.ur5e_robot.jacob0(self.q[2:])
        J_mb = np.hstack((J_p, J_a_e))  # 6x8 자코비안 (선형 속도 + 각속도)
        J_mb_v = J_mb[:3, :]  # 3x8 자코비안 (선형 속도)
        J_mb_w = J_mb[3:, :]  # 3x8 자코비안 (각속도)
        
        # print(my_robot.body_names)
        # ['base_link', 'front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link', 
        # 'ur5e_shoulder_link', 'ur5e_upper_arm_link', 'ur5e_forearm_link', 'ur5e_wrist_1_link', 'ur5e_wrist_2_link', 'ur5e_wrist_3_link', 
        # 'robotiq_85_left_inner_knuckle_link', 'robotiq_85_left_knuckle_link', 'robotiq_85_right_inner_knuckle_link', 'robotiq_85_right_knuckle_link', 'robotiq_85_left_finger_tip_link', 'robotiq_85_right_finger_tip_link']
        
        T_error = np.linalg.inv(H_current.A) @ H_desired.A  # 4x4

        et = np.sum(np.abs(T_error[:3, -1])) 

        # Gain term (lambda) for control minimisation
        Y = 100

        # Quadratic component of objective function
        Q = np.eye(self.n_dof + 6)

        # Joint velocity component of Q
        Q[:2, :2] *= 1.0 / (et*100)

        # Slack component of Q
        Q[self.n_dof :, self.n_dof :] = (1.0 / et) * np.eye(6)

        H = np.zeros((self.n_dof-2, 6, self.n_dof-2))  # same as jacobm

        for j in range(self.n_dof-2):
            for i in range(j, self.n_dof-2):
                H[j, :3, i] = np.cross(J_mb_w[:, j], J_mb_v[:, i])
                H[j, 3:, i] = np.cross(J_mb_w[:, j], J_mb_w[:, i])
                if i != j:
                        H[i, :3, j] = H[j, :3, i]
                        H[i, 3:, j] = H[j, 3:, i]

        # manipulability only for arm joints
        J_a = self.ur5e_robot.jacob0(self.q[2:])
        m = J_a @ J_a.T 
        m_det = np.linalg.det(m)  
        m_t = np.sqrt(m_det)  # manipulability (sqrt(det(J * J^T)))

        JJ_inv = np.linalg.inv((J_a @ J_a.T))  #.reshape(-1, order='F')

        # Compute manipulability Jacobian only for arm joints
        J_m = np.zeros((self.n_dof-2,1))
        for i in range(self.n_dof-2):
            c = J_a @ np.transpose(H[i, :, :])  # shape: (6,6)
            J_m[i,0] = m_t * np.transpose(c.flatten("F")) @ JJ_inv.flatten("F")

        A = np.zeros((self.n_dof + num_points + 6, self.n_dof + 6))
        B = np.zeros(self.n_dof + num_points + 6)

        J_c = np.zeros(self.n_dof+6)
        w_p_sum = 0.0
        min_dist = 0.0

        for i , pose in enumerate(xform_pose) :
            distance, index, g_vec = self.get_nearest_obstacle_distance(pose, self.obstacles_positions, self.obstacle_radius)
            min_dist = np.min(distance)
            if i < 4:  # mobile base wheels
                jac_mobile = F
                jac_mobile_v = jac_mobile[:3,:]  # 3x2 자코비안 (선형 속도)
                d_dot = g_vec @ jac_mobile_v[:,:2] 
                A[i, :2] = d_dot 
                A[i, 2:] = np.zeros((1, self.n_dof - 2 + 6))  # arm joints
                B[i] = (distance - self.d_safe) / (self.d_influence - self.d_safe) 
                w_p = (self.d_influence-distance)/(self.d_influence-self.d_safe) 
                J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                w_p_sum += w_p
            elif 3 < i < 9:  # UR5e joints
                T_instant = T_b0 @ self.ur5e_robot.fkine(self.q[2:2 + i - 4],end = self.ur5e_robot.links[i - 4]).A
                jac_mobile = base.tr2adjoint(T_instant.T) @ F

                # print(ur5e_robot.links)
                # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                jac_arm = base.tr2adjoint(T_instant.T) @ self.ur5e_robot.jacob0(self.q[2:2 + i - 3], end = self.ur5e_robot.links[i - 2])
                J_mb_instant = np.hstack((jac_mobile, jac_arm))
                J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                d_dot = g_vec @ J_mb_v_instant[:, :] 

                A[i, :i-1] = d_dot
                A[i, i-1:] = np.zeros((1, self.n_dof - (i-1) + 6))  # arm joints
                B[i] = (distance - self.d_safe) / (self.d_influence - self.d_safe)
                w_p = (self.d_influence-distance)/(self.d_influence-self.d_safe) 
                J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                w_p_sum += w_p

            elif i == 9:
                T_instant = T_b0 @ self.ur5e_robot.fkine(self.q[2:2 + i - 4],end = self.ur5e_robot.links[i - 4]).A
                jac_mobile = base.tr2adjoint(T_instant.T) @ F

                # print(ur5e_robot.links)
                # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                jac_arm = base.tr2adjoint(T_instant.T) @ self.ur5e_robot.jacob0(self.q[2:2 + i - 3], end = self.ur5e_robot.links[i - 1])
                J_mb_instant = np.hstack((jac_mobile, jac_arm))
                J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                d_dot = g_vec @ J_mb_v_instant[:, :] 

                A[i, :i-1] = d_dot
                A[i, i-1:] = np.zeros((1, self.n_dof - (i-1) + 6))  # arm joints
                B[i] = (distance - self.d_safe) / (self.d_influence - self.d_safe)
                w_p = (self.d_influence-distance)/(self.d_influence-self.d_safe) 
                J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                w_p_sum += w_p

            else:
                T_topoint = np.eye(4)
                T_topoint[:3, 3] = (i-9) * dist_vec  # 현재 점에서 목표 점까지의 변환 행렬
                T_instant = T_b0 @ self.ur5e_robot.fkine(self.q[2:2 + 9 - 4],end = self.ur5e_robot.links[9 - 4]).A @ T_topoint
                jac_mobile = base.tr2adjoint(T_instant.T) @ F

                # print(ur5e_robot.links)
                # world, base link, shoulder link, upper arm link, forearm link, wrist 1 link, wrist 2 link, wrist 3 link, tool link
                jac_arm = base.tr2adjoint(T_instant.T) @ self.ur5e_robot.jacob0(self.q[2:2 + 9 - 3], end = self.ur5e_robot.links[9 - 1])
                J_mb_instant = np.hstack((jac_mobile, jac_arm))
                J_mb_v_instant = J_mb_instant[:3, :]  # 3x8 자코비안 (선형 속도)

                d_dot = g_vec @ J_mb_v_instant[:, :] 
                # print("d_dot: ", d_dot.shape)  # (8,)
                A[i, :8] = d_dot
                A[i, 8:] = np.zeros((1, 6))  # arm joints
                B[i] = (distance - self.d_safe) / (self.d_influence - self.d_safe)
                w_p = (self.d_influence-distance)/(self.d_influence-self.d_safe) 
                J_c[:6] += A[i, :6] * w_p  # 베이스 조인트 속도에 대한 제약 조건
                w_p_sum += w_p


        C = np.concatenate((np.zeros(2), -J_m.reshape((self.n_dof - 2,)), np.zeros(6)))
        lambda_max = 1.5
        lambda_c = (lambda_max /(self.d_influence - self.d_safe)**2) * (min_dist - self.d_influence)**2
        C += (lambda_c * J_c/w_p_sum)  # 베이스 조인트 속도에 대한 제약 조건 추가

        bTe = self.ur5e_robot.fkine(self.q[2:], include_base=False).A  
        θε = atan2(bTe[1, -1], bTe[0, -1])
        C[0] = - self.k_e * θε  # 베이스 x 위치 오차


        J_ = np.c_[J_mb, np.eye(6)]  # J_ 행렬 (예시)

        eTep = np.linalg.inv(T) @ H_desired.A  # 현재 위치에서의 오차 행렬

        e = np.empty(6)

        # Translational error
        e[:3] = eTep[:3, -1]

        # Angular error
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        
        k = 1.5 * np.eye(6) # gain
        v = k @ e
        v[3:] *= 1.3

        lb = self.lb
        ub = self.ub
        qd = qp.solve_qp(Q,C,A,B,J_,v,lb=lb, ub=ub, solver='quadprog')
        if qd is None:
                print("QP solution is None")
                qd = np.array([0.,0.,0.,0.0,0.,0.,0.,0.]) 
        return qd, eTep
    

    def get_nearest_obstacle_distance(self, position, obstacles, obstacle_radius):
        """
        Calculate the distance to the nearest obstacle from a given position.
        
        Args:
            position (np.ndarray): The position from which to calculate the distance.
            obstacles (list): A list of obstacle positions.
            
        Returns:
            float: The distance to the nearest obstacle.
            index (int): The index of the nearest obstacle.
        """
        for obs in obstacles:
            obs[2] = position[2]  # Set the z-coordinate of the obstacle to the specified value
        distances = [np.linalg.norm(position - obs)-obstacle_radius for obs in obstacles]
        index = np.argmin(distances)
        

        # Create a directional vector
        g_vec = (position - obstacles[index])
        g_vec /= np.linalg.norm(g_vec)
        return min(distances), index, g_vec

    def generate_points_between_positions(self, start_pos, end_pos, num_points=10):
        """
        두 3차원 위치를 이어주는 선에서 일정한 간격으로 점을 생성하는 함수.

        Args:
            start_pos (np.ndarray): 시작 위치 (3차원 좌표).
            end_pos (np.ndarray): 끝 위치 (3차원 좌표).
            num_points (int): 생성할 점의 개수 (기본값: 10).

        Returns:
            np.ndarray: 생성된 점들의 좌표 배열 (shape: num_points x 3).
        """
        # 시작 위치와 끝 위치를 연결하는 선을 따라 일정한 간격으로 점 생성
        points = np.linspace(start_pos, end_pos, num_points)
        dist_vec = (end_pos - start_pos)/num_points
        return points, dist_vec