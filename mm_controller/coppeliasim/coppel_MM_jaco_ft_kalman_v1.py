import time
import sys
import os
import math

import scipy.linalg
from sympy import Max

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
from cv2 import waitKey
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import scipy
from kalman import Kalman
from jacobian import Jacobian
from coppel_controller import coppel_controller
        
dTol = 0.005
derivative_dist = 0.0
maxForce = 100
integral_dist = 0.0
previous_err_dist = 0.0
integral_theta = 0.0
previous_err_theta = 0.0

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

client.step()

sim.startSimulation()
dummy = sim.getObject('/Dummy')
f_L = sim.getObjectHandle('/base_link_respondable/front_left_wheel')
f_R = sim.getObjectHandle('/base_link_respondable/front_right_wheel')
r_L = sim.getObjectHandle('/base_link_respondable/rear_left_wheel')
r_R = sim.getObjectHandle('/base_link_respondable/rear_right_wheel')
# scout_base = sim.getObject('./Dummy')

# scout = sim.getObjectHandle('/UR5')
# Ur = sim.getObjectHandle('/base_link_visual')
# sim.setObjectParent(Ur, scout, True)

Ur5_EE = sim.getObject('/base_link_respondable/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint/link/connection')
j1 = sim.getObject('/base_link_respondable/UR5/joint')
j2 = sim.getObject('/base_link_respondable/UR5/joint/joint')
j3 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint')
j4 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint')
j5 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint')
j6 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint/joint')
j1_h = sim.getObjectHandle('/base_link_respondable/UR5/joint')
j2_h = sim.getObjectHandle('/base_link_respondable/UR5/joint/joint')
j3_h = sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint')
j4_h = sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint')
j5_h = sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint/joint')
j6_h = sim.getObjectHandle('/base_link_respondable/UR5/joint/joint/joint/joint/joint/joint')


def moveToAngle_RL(w_R, w_L):
    vel_r = w_R
    vel_l = w_L
    sim.setJointTargetVelocity(f_L, vel_l)
    sim.setJointTargetVelocity(f_R, -vel_r)
    sim.setJointTargetVelocity(r_L, vel_l)
    sim.setJointTargetVelocity(r_R, -vel_r)

def set_velocity(l_v, a_v, j1_tv, j2_tv, j3_tv, j4_tv, j5_tv, j6_tv):
    w_R, w_L, wc = coppel_controller.coppel_scout2_controller(l_v, a_v)

    sim.setJointTargetVelocity(j1_h, j1_tv)  #-wc*angle_con_value)
    sim.setJointTargetVelocity(j2_h, j2_tv)
    sim.setJointTargetVelocity(j3_h, j3_tv)
    sim.setJointTargetVelocity(j4_h, j4_tv)
    sim.setJointTargetVelocity(j5_h, j5_tv)
    sim.setJointTargetVelocity(j6_h, j6_tv)
    return w_R, w_L, wc


if __name__ == '__main__':
    try:
        waitKey(1)
        Kalman = Kalman()
        Jacobian = Jacobian()
        coppel_controller = coppel_controller()
        current_time = 0
        X_d_list = []
        state = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        t = sim.getSimulationTime()
        while (t := sim.getSimulationTime()) > current_time:
            dt = t - current_time
            covariance, A, B, Q, H, R = Kalman.kalman_init(dt)

            s = f'Simulation time: {t:.2f} [s]'
            print(s)
            # joint value (radian)
            j1_r = sim.getJointPosition(j1)
            j2_r = sim.getJointPosition(j2)
            j3_r = sim.getJointPosition(j3)
            j4_r = sim.getJointPosition(j4)
            j5_r = sim.getJointPosition(j5)
            j6_r = sim.getJointPosition(j6)

            # Jacobian matrix
            J = Jacobian.jacobian(j1_r, j2_r, j3_r, j4_r, j5_r, j6_r)
            
            # J_ = R @ J
            j_J = J @ J.T
            
            U, Sigma, Vt = scipy.linalg.svd(j_J, full_matrices=True)

            
            
            X_d = sim.getObjectPosition(dummy, -1)
            X_d = np.array(X_d)
            X_c = sim.getObjectPosition(Ur5_EE, -1)
            X_c = np.array(X_c)

            
            X_d_list.append(X_d)

            if len(X_d_list) > 9:
                # print(f"==>> len(X_d_list): {len(X_d_list)}")
                for i, z in enumerate(X_d_list):
                    z = np.array(z).reshape(-1, 1)  # Convert to column vector
                    # Prediction step
                    control_input = np.array([0, 0, 0]).reshape(-1, 1)  # No control input
                    predicted_state, predicted_covariance = Kalman.kalman_filter_predict(state, covariance, A, B, control_input, Q)
                    # Update step
                    state, covariance = Kalman.kalman_filter_update(predicted_state, predicted_covariance, H, z, R)
                future_time_step = 3.0  # Time steps to predict into the future
                A_future = A.copy()
                # print(f"==>> A_future: {A_future}")
                A_future[0:3, 3:6] *= future_time_step  # Adjust for future velocity scaling
                predicted_future_state, _ = Kalman.kalman_filter_predict(state, covariance, A_future, B, control_input, Q)
                # print(f"==>> predicted_future_state: {predicted_future_state}")
                X_d = predicted_future_state[0:3].flatten()
                X_d_list.pop(0)

            else:
                
                state = np.array([X_d[0], X_d[1], X_d[2],(X_d[0]-state[3][0])/dt, (X_d[1]-state[4][0])/dt, (X_d[2]-state[5][0])/dt ]).reshape(-1, 1)

            d_goal = X_d - X_c
            d_goal_2D = X_d[:2] - X_c[:2]
            d_goal_unit = d_goal/np.linalg.norm(d_goal)
            d_goal_unit_2D = d_goal_2D/np.linalg.norm(d_goal_2D)
            manipulability_direction = U[:, 0]
            manipulability_direction_array = np.array(manipulability_direction)


            # 시작점 (엔드 이펙터 위치)
            start_point = X_c

            # 벡터 끝점 정의 (스케일 조절 가능)
            scale = 0.2  # 화살표 길이를 조정하는 스케일링 값
            end_point_ = start_point + scale * d_goal_unit
            
            # 화살표 생성 (start_point에서 end_point로)
            line_handle = sim.addDrawingObject(
                    sim.drawing_lines,  # objectType: 선 그리기
                    0.01,               # size: 선 두께
                    0.0,                # duplicateTolerance: 중복 허용 (0으로 비활성화)
                    -1,                 # parentHandle: 월드 좌표계
                    2,                  # maxItemCount: 최대 2개의 점 (시작점과 끝점)
                    [1, 0, 0]           # 색상: 빨간색 (RGB)
                )
            line_handle_ = sim.addDrawingObject(
                    sim.drawing_lines,  # objectType: 선 그리기
                    0.01,               # size: 선 두께
                    0.0,                # duplicateTolerance: 중복 허용 (0으로 비활성화)
                    -1,                 # parentHandle: 월드 좌표계
                    2,                  # maxItemCount: 최대 2개의 점 (시작점과 끝점)
                    [0, 1, 0]           # 색상: 빨간색 (RGB)
                )
            # sim.addDrawingObjectItem(line_handle, list(start_point) + list(end_point))
            sim.addDrawingObjectItem(line_handle_, list(start_point) + list(end_point_))

            time_interval = 0.05
            angle_con_value = 1.0
            # 최대 속도 한계 설정 (예시)
            max_q_dot = 0.1 # 최대 속도 한계를 설정
            print('distance: ', np.linalg.norm(d_goal))
            if np.linalg.norm(d_goal) < 0.01:
                # 최대 힘 제한을 설정
                sim.setJointForce(j1_h, maxForce)
                sim.setJointForce(j2_h, maxForce)
                sim.setJointForce(j3_h, maxForce)
                sim.setJointForce(j4_h, maxForce)
                sim.setJointForce(j5_h, maxForce)
                sim.setJointForce(j6_h, maxForce)
                l_v = 0.0
                a_v = 0.0
                w_R, w_L, _ = set_velocity(l_v, a_v, 0, 0, 0, 0, 0, 0)

            elif np.linalg.norm(d_goal) > 0.25:
                time_interval = t - current_time
                current_time = t 
                l_v, a_v = coppel_controller.kinematic_control(integral_dist, previous_err_dist,integral_theta, previous_err_theta, d_goal_2D[0]*10, d_goal_2D[1]*10) #cm

                vectorsize = 0.002
                new_d_goal = d_goal_unit * vectorsize
                d_goal_v = new_d_goal/(time_interval)
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
                w_R, w_L, wc = set_velocity(l_v, a_v, j1_tv, j2_tv, j3_tv, 0, 0, 0)

                sim.addDrawingObjectItem(line_handle, None)
                sim.addDrawingObjectItem(line_handle_, None)

            else:
                time_interval = t - current_time
                current_time = t 
                l_v, a_v = coppel_controller.kinematic_control(integral_dist, previous_err_dist,integral_theta, previous_err_theta, d_goal_2D[0]*10, d_goal_2D[1]*10) #cm

                new_d_goal = d_goal_unit #* vectorsize
                d_goal_v = new_d_goal/(time_interval)
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
                w_R, w_L, wc = set_velocity(l_v, a_v, j1_tv, j2_tv, j3_tv, 0, 0, 0)

                sim.addDrawingObjectItem(line_handle, None)
                sim.addDrawingObjectItem(line_handle_, None)
            moveToAngle_RL(w_R, w_L)

            client.step()  # Advance simulation by one step
    except Exception as e:
        print(f"Error during simulation step: {e}")
    finally:
        sim.stopSimulation()
