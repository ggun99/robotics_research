from hmac import new
import time
import sys
import os
import math

import scipy.linalg
from sympy import Max

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
sys.path.append("/home/airlab/robotics_research/mm_controller/coppeliasim")
from cv2 import waitKey
import numpy as np
from zmqRemoteApi import RemoteAPIClient
from kalman import Kalman
from jacobian import Jacobian
from coppel_controller import coppel_controller



class Coppeliasim():
    def __init__(self, sim):
        self.dTol = 0.005
        self.derivative_dist = 0.0
        self.maxForce = 100
        self.integral_dist = 0.0
        self.previous_err_dist = 0.0
        self.integral_theta = 0.0
        self.previous_err_theta = 0.0
        self.state = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        self.U_previous = None
        self.Sigma_previous = None
        self.Vt_previous = None
        self.sim = sim
        # self.dummy = self.sim.getObject('/Dummy')
        self.dummy_move = self.sim.getObjectHandle('/Dummy')
        self.dummy_move_mani = self.sim.getObjectHandle('/Dummy[1]')
        self.f_L = self.sim.getObjectHandle('/base_link_respondable/front_left_wheel')
        self.f_R = self.sim.getObjectHandle('/base_link_respondable/front_right_wheel')
        self.r_L = self.sim.getObjectHandle('/base_link_respondable/rear_left_wheel')
        self.r_R = self.sim.getObjectHandle('/base_link_respondable/rear_right_wheel')
        self.scout = self.sim.getObjectHandle('/base_link_respondable/Dummy')
        # scout_base = sim.getObject('./Dummy')

        # scout = sim.getObjectHandle('/UR5')
        # Ur = sim.getObjectHandle('/base_link_visual')
        # sim.setObjectParent(Ur, scout, True)

        self.Ur5_EE = self.sim.getObject('/base_link_respondable/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint/link/connection')
        self.j1 = self.sim.getObject('/base_link_respondable/UR5/joint')
        self.j2 = self.sim.getObject('/base_link_respondable/UR5/joint/joint')
        self.j3 = self.sim.getObject('/base_link_respondable/UR5/joint/joint/joint')
        self.j4 = self.sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint')
        self.j5 = self.sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint')
        self.j6 = self.sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint/joint')
        self.j1_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint')
        self.j2_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint/joint')
        self.j3_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint')
        self.j4_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint')
        self.j5_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint/joint')
        self.j6_h = self.sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint/joint/joint')
        
        self.wall = self.sim.getObjectHandle('/240cmHighWall50cm/Dummy')
        
        self.new_position_mani = self.sim.getObjectPosition(self.dummy_move, -1)

    def moveToAngle_RL(self, w_R, w_L):
        vel_r = w_R
        vel_l = w_L
        self.sim.setJointTargetVelocity(self.f_L, vel_l)
        self.sim.setJointTargetVelocity(self.f_R, -vel_r)
        self.sim.setJointTargetVelocity(self.r_L, vel_l)
        self.sim.setJointTargetVelocity(self.r_R, -vel_r)

    def set_velocity(self, l_v, a_v, j1_tv, j2_tv, j3_tv, j4_tv, j5_tv, j6_tv, coppel_controller):
        w_R, w_L, wc = coppel_controller.coppel_scout2_controller(l_v, a_v)

        self.sim.setJointTargetVelocity(self.j1_h, j1_tv)  #-wc*angle_con_value)
        self.sim.setJointTargetVelocity(self.j2_h, j2_tv)
        self.sim.setJointTargetVelocity(self.j3_h, j3_tv)
        self.sim.setJointTargetVelocity(self.j4_h, j4_tv)
        self.sim.setJointTargetVelocity(self.j5_h, j5_tv)
        self.sim.setJointTargetVelocity(self.j6_h, j6_tv)
        return w_R, w_L, wc

    def coppeliasim(self, Kalman,Jacobian, dt, X_d_list, coppel_controller):
        covariance, A, B, Q, H, R = Kalman.kalman_init(dt)

        # s = f'Simulation time: {t:.2f} [s]'
        # print(s)
        # joint value (radian)
        j1_r = self.sim.getJointPosition(self.j1)
        j2_r = self.sim.getJointPosition(self.j2)
        j3_r = self.sim.getJointPosition(self.j3)
        j4_r = self.sim.getJointPosition(self.j4)
        j5_r = self.sim.getJointPosition(self.j5)
        j6_r = self.sim.getJointPosition(self.j6)

        # Jacobian matrix
        J = Jacobian.jacobian(j1_r, j2_r, j3_r, j4_r, j5_r, j6_r)
        
        j_J = J @ J.T
        
        # SVD 계산
        try:
            U, Sigma, Vt = scipy.linalg.svd(j_J, full_matrices=True)
            self.U_previous, self.Sigma_previous, self.Vt_previous = U, Sigma, Vt  # 이전 값 업데이트
        except ValueError as e:
            print("SVD computation failed due to invalid input:")
            print(e)
            U, Sigma, Vt = self.U_previous, self.Sigma_previous, self.Vt_previous  # 이전 값 사용

        velocity = [-0.065, 0.0, 0.0]  # Change per simulation step (for each axis)
        X_d = self.sim.getObjectPosition(self.dummy_move, -1)
        X_d = np.array(X_d)
        X_d_mani = self.sim.getObjectPosition(self.dummy_move_mani, -1)
        X_d_mani = np.array(X_d_mani)
        X_c = self.sim.getObjectPosition(self.Ur5_EE, -1)
        X_c = np.array(X_c)
        X_scout = self.sim.getObjectPosition(self.scout, -1)
        X_scout = np.array(X_scout)
        X_d_list.append(X_d)

        wall = self.sim.getObjectPosition(self.wall, -1)
        wall = np.array(wall)
        f_ref = 0.005
        f_res = 0.005
        d_max = 0.8
        
        d_wall = X_d - wall
        d_wall_abs = np.linalg.norm(d_wall)
        d_wall_unit = d_wall/d_wall_abs
        if d_wall_abs < d_max and d_wall_unit[0] > 0:
            X_d = X_d + d_wall_unit * f_ref * np.exp(-np.abs(d_wall_abs)/d_max)
            print(f"==>> d_wall: {d_wall}")
            print(f"==>> d_wall_abs: {d_wall_abs}")
            print(f"==>> d_wall_unit: {d_wall_unit}")
            print(f"==>> 더한값: {d_wall_unit * f_ref * np.exp(-np.abs(d_wall_abs)/d_max)}")
        elif d_wall_abs < d_max and d_wall_unit[0] < 0:
            X_restore = X_d_mani - X_d
            X_restore_unit = X_restore/np.linalg.norm(X_restore)
            X_d = X_d + d_wall_unit * f_ref * np.exp(-np.abs(d_wall_abs)/d_max) + f_res * X_restore_unit
            print('여기야ㅑㅑㅑ')
            # print(f"==>> d_wall_unit: {d_wall_unit}")
            # print(f"==>> X_d: {X_d}")
        d_goal = X_d_mani - X_c
        d_goal_2D = X_d[:2] - X_scout[:2]
        d_goal_unit = d_goal/np.linalg.norm(d_goal)
        print(f"==>> d_goal_unit: {d_goal_unit}")
        print(f"==>> X_d: {X_d}")
        # Update the dummy position over time
        new_position = [X_d[0] + velocity[0]*0.1, X_d[1] + velocity[1]*0.1, X_d[2] + velocity[2]*0.1]
        # self.new_position_mani = X_d_mani
        self.new_position_mani = [self.new_position_mani[0] + velocity[0]*0.1, self.new_position_mani[1] + velocity[1]*0.1, self.new_position_mani[2] + velocity[2]*0.1]
        print(f"==>> self.new_pos: {self.new_position_mani}")
        print(f"==>> new_position: {new_position}")
        # print(f"==>> X_d: {X_d}")
        self.sim.setObjectPosition(self.dummy_move, -1, new_position)
        self.sim.setObjectPosition(self.dummy_move_mani, -1, self.new_position_mani)
        
        # 최대 속도 한계 설정 (예시)
        max_q_dot = 0.1 # 최대 속도 한계를 설정
        # print('distance: ', np.linalg.norm(d_goal))
        if np.linalg.norm(d_goal) < 0.01:
            # self.previous_time = t 
            # 최대 힘 제한을 설정
            self.sim.setJointForce(self.j1_h, self.maxForce)
            self.sim.setJointForce(self.j2_h, self.maxForce)
            self.sim.setJointForce(self.j3_h, self.maxForce)
            self.sim.setJointForce(self.j4_h, self.maxForce)
            self.sim.setJointForce(self.j5_h, self.maxForce)
            self.sim.setJointForce(self.j6_h, self.maxForce)
            l_v = 0.0
            a_v = 0.0
            w_R, w_L, _ = self.set_velocity(l_v, a_v, 0, 0, 0, 0, 0, 0, coppel_controller)

        else:
            # self.previous_time = t 
            l_v, a_v = coppel_controller.kinematic_control(self.integral_dist, self.previous_err_dist,self.integral_theta, self.previous_err_theta, d_goal_2D[0]*10, d_goal_2D[1]*10) #cm

            vectorsize = 0.002
            new_d_goal = d_goal_unit #* vectorsize
            d_goal_v = new_d_goal/(dt)
            epsilon = 1e-6  # 작은 특이값에 대한 임계값
            Sigma_inv = np.diag(1.0 / (Sigma + epsilon))  # 작은 특이값에 임계값을 더하여 안정화
            # pseudo inverse jacobian matrix
            J_pseudo = Vt.T @ Sigma_inv @ U.T
            
            q_dot = J_pseudo @ d_goal_v

            # 속도가 한계를 초과하면 제한
            q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)

            j1_tv = q_dot[0]
            j2_tv = q_dot[1]
            j3_tv = q_dot[2]

            # j4_tv = q_dot[3]
            # j5_tv = q_dot[4]
            # j6_tv = q_dot[5]
            w_R, w_L, wc = self.set_velocity(l_v, a_v, j1_tv, j2_tv, j3_tv, 0, 0, 0, coppel_controller)


        self.moveToAngle_RL(w_R, w_L)
        

print('Program started')
previous_time = 0 
if __name__ == '__main__':
    try:
        waitKey(1)
        Kalman = Kalman()
        Jacobian = Jacobian()
        coppel_simulator = coppel_controller()
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        client.step()
        sim.startSimulation()

        X_d_list = []
        coppel = Coppeliasim(sim)
        while (t := sim.getSimulationTime()) > 0:
            t = sim.getSimulationTime()
            dt = t - previous_time
            if dt <= 0 or np.isnan(dt):
                print("dt가 0이거나 NaN입니다. 기본값으로 설정합니다.")
                dt = 1e-6  # 매우 작은 기본값
            previous_time = t
            
            coppel.coppeliasim(Kalman, Jacobian, dt, X_d_list, coppel_simulator)
            client.step()  # Advance simulation by one step
    except Exception as e:
        print(f"Error during simulation step: {e}")
    finally:
        sim.stopSimulation()
