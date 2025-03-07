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

offset = 0.02
dTol = 0.005
integral_dist = 0.0
previous_err_dist = 0.0
derivative_dist = 0.0
maxForce = 100
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
scout_base = sim.getObject('./Dummy')

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

velocity = [0.003, 0.003, 0.005]  # Change per simulation step (for each axis)


def jacobian(j1_r, j2_r, j3_r, j4_r, j5_r, j6_r):
    # UR5e Joint value (radian)
    j = np.array([float(j1_r), float(j2_r), float(j3_r), float(j4_r), float(j5_r), float(j6_r)])

    # UR5e DH parameters
    a = np.array([0., -0.425, -0.3922, 0., 0., 0.])

    d = np.array([0.1625, 0., 0., 0.1333, 0.0997, 0.0996])

    alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
    # Jacobian matrix
  
    J11 = (d[3]*np.cos(j[0]))+np.sin(j[0])*(a[1]*np.sin(j[1])+a[2]*np.sin(j[1]+j[2])+d[4]*np.sin(j[1]+j[2]+j[3]))
    J12 = -(np.cos(j[0])*(a[1]*np.cos(j[1])+a[2]*np.cos(j[1]+j[2])+d[4]*np.cos(j[1]+j[2]+j[3])))
    J13 = -(np.cos(j[0])*(a[2]*np.cos(j[1]+j[2])+d[4]*np.cos(j[1]+j[2]+j[3])))
    J14 = -(d[4]*np.cos(j[0])*np.cos(j[1]+j[2]+j[3]))
    J15 = 0
    J16 = 0
    J21 = (d[3]*np.sin(j[0]))-np.cos(j[0])*(a[1]*np.sin(j[1])+a[2]*np.sin(j[1]+j[2])+d[4]*np.sin(j[1]+j[2]+j[3]))
    J22 = -(np.sin(j[0])*(a[1]*np.cos(j[1])+a[2]*np.cos(j[1]+j[2])+d[4]*np.cos(j[1]+j[2]+j[3])))
    J23 = -(np.sin(j[0])*(a[2]*np.cos(j[1]+j[2])+d[4]*np.cos(j[1]+j[2]+j[3])))
    J24 = -(d[4]*np.sin(j[0])*np.cos(j[1]+j[2]+j[3]))
    J25 = 0
    J26 = 0
    J31 = 0
    J32 = -(a[1]*np.sin(j[1]))-(a[2]*np.sin(j[1]+j[2]))-(d[4]*np.sin(j[1]+j[2]+j[3]))
    J33 = -(a[2]*np.sin(j[1]+j[2]))-(d[4]*np.sin(j[1]+j[2]+j[3]))
    J34 = -(d[4]*np.sin(j[1]+j[2]+j[3]))
    J35 = 0
    J36 = 0
    J = np.array([[J11, J12, J13, J14, J15, J16], [J21, J22, J23, J24, J25, J26], [J31, J32, J33, J34, J35, J36]])

    return J
# def pid_scout(pose_dummy, pose_scout, integral_dist, previous_err_dist, integral_theta, previous_err_theta):
#     # Calculate errors in position
#     distance_x = pose_dummy[0] - pose_scout[0]
    
#     err_dist = distance_x - offset
#     err_theta = pose_dummy[1] - pose_scout[1]

#     Kp_dist = 0.03
#     Ki_dist = 0.1
#     Kd_dist = 0.08
#     Kp_theta = 0.8
#     Ki_theta = 0.1
#     Kd_theta = 0.01
            
#     integral_dist += err_dist
#     # Prevent integral windup
#     integral_dist = min(max(integral_dist, -10), 10)
    
#     derivative_dist = err_dist - previous_err_dist

#     integral_theta += err_theta
#     integral_theta = min(max(integral_theta, -10), 10)
#     derivative_theta = err_theta - previous_err_theta
#     print(f'err_dist: {err_dist}')
#     print(f'err_theta: {err_theta}')
#     if np.abs(err_dist) >= dTol:
#         l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
#         previous_err_dist = err_dist
#     else:
#         print("Scout2.0 stopping - distance within tolerance")
#         l_v = 0.0    
#     # PID control for angular velocity
#     if np.abs(err_theta) >= dTol: #checking whether heading angle error within tolerence
#         a_v = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
#         previous_err_theta = err_theta
        
#     else:
#         print(f"Scout2.0  stopping goal heading within tolerence")
#         a_v = 0.0 
#     vc = float(l_v)
#     wc = -float(a_v)
#     r = 0.165
#     l = 0.582
#     w_R = vc/r + l*wc/(2*r)
#     w_L = vc/r - l*wc/(2*r) # Convert linear velocity to angular velocity
#     return w_R, w_L

def moveToAngle_R(w_R):
    vel = w_R
    # sim.setJointTargetVelocity(f_L, vel)
    sim.setJointTargetVelocity(f_R, -vel)
    # sim.setJointTargetVelocity(r_L, vel)
    sim.setJointTargetVelocity(r_R, -vel)

def moveToAngle_L(w_L):
    vel = w_L
    sim.setJointTargetVelocity(f_L, vel)
    # sim.setJointTargetVelocity(f_R, -vel)
    sim.setJointTargetVelocity(r_L, vel)
    # sim.setJointTargetVelocity(r_R, -vel)


def kinematic_control(e_x, e_y):
    dTol = 0.05

    integral_dist = 0.0
    previous_err_dist = 0.0
    integral_theta = 0.0
    previous_err_theta = 0.0
    
    err_dist = e_x

    err_theta = e_y
    
    
    Kp_dist = 0.0001
    Ki_dist = 0.01
    Kd_dist = 0.08
    Kp_theta = 0.001
    Ki_theta = 0.01
    Kd_theta = 0.01
     
    
    integral_dist += err_dist
    derivative_dist = err_dist - previous_err_dist
    integral_theta += err_theta
    derivative_theta = err_theta - previous_err_theta
    
    # TODO: Add integral and derivative calculations for complete PID

    # PID control for linear velocity
    
    if np.abs(err_dist) >= dTol: #checking whether error distance within tolerence
        l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
        previous_err_dist = err_dist
    else:
        print(f"Scout2.0  stopping goal distance within tolerence")
        l_v = 0.0    

    # PID control for angular velocity
    if np.abs(err_theta) >= dTol: #checking whether heading angle error within tolerence
        a_v = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
        previous_err_theta = err_theta
        
    else:
        print(f"Scout2.0  stopping goal heading within tolerence")
        a_v = 0.0      

    # Send the velocities
    # vmax = 21 , wmax = 5
    vc = float(l_v)
    if vc > 0.2:
        vc = 0.2
    elif vc < -0.2:
        vc = -0.2
    wc = float(a_v)
    if wc > 0.1:
        wc = 0.1
    elif wc < -0.1:
        wc = -0.1
    
    return vc, wc

def coppel_scout2_controller(l_v, a_v):
    vc = float(l_v)
    wc = float(a_v)
    r = 0.165
    l = 0.582
    w_R = vc/r + l*wc/(2*r)
    w_L = vc/r - l*wc/(2*r)
    return w_R, w_L, wc


theta = 0.0
def Rotation_matirx(theta):
    R = np.array([[np.cos(theta),-np.sin(theta), 0],
                 [-np.sin(theta),np.cos(theta), 0],
                 [0, 0 ,1]])
    return R


try:
    waitKey(1)
    total_time = 50
    while True:#(t := sim.getSimulationTime()) < total_time:
        # s = f'Simulation time: {t:.2f} [s]'
        # print(s)
        # joint value (radian)
        j1_r = sim.getJointPosition(j1)
        j2_r = sim.getJointPosition(j2)
        j3_r = sim.getJointPosition(j3)
        j4_r = sim.getJointPosition(j4)
        j5_r = sim.getJointPosition(j5)
        j6_r = sim.getJointPosition(j6)
        R = Rotation_matirx(theta)
        # print(theta)
        # Jacobian matrix
        J = jacobian(j1_r, j2_r, j3_r, j4_r, j5_r, j6_r)
        
        # J_ = R @ J
        j_J = J @ J.T
        
        U, Sigma, Vt = scipy.linalg.svd(j_J, full_matrices=True)

        # Sigma = np.array(Sigma)
        # MaxIndex = np.argmax(Sigma)
        # direction = U[:,MaxIndex]
        
        X_d = sim.getObjectPosition(dummy, -1)
        X_d = np.array(X_d)
        X_c = sim.getObjectPosition(Ur5_EE, -1)
        X_c = np.array(X_c)

        d_goal = X_d - X_c
        d_goal_2D = X_d[:2] - X_c[:2]
        d_goal_unit = d_goal/np.linalg.norm(d_goal)
        d_goal_unit_2D = d_goal_2D/np.linalg.norm(d_goal_2D)
        manipulability_direction = U[:, 0]
        manipulability_direction_array = np.array(manipulability_direction)
        direction = R @ manipulability_direction
        manipulability_2D_direction = direction[:2]
        manipulability_2D_direction_unit = manipulability_2D_direction/np.linalg.norm(manipulability_2D_direction)
        # manipulability_direction_vector = [[direction[0]],direction[1],direction[2]]

        # 시작점 (엔드 이펙터 위치)
        start_point = X_c

        # 벡터 끝점 정의 (스케일 조절 가능)
        scale = 0.2  # 화살표 길이를 조정하는 스케일링 값
        end_point = start_point + scale * direction
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
        sim.addDrawingObjectItem(line_handle, list(start_point) + list(end_point))
        sim.addDrawingObjectItem(line_handle_, list(start_point) + list(end_point_))
        # 내적 (유사도 확인)
        similarity = np.dot(direction, d_goal_unit)
        # 벡터 간 각도 계산 (반시계방향 회전)
        dot_2D = np.dot(manipulability_2D_direction_unit, d_goal_unit_2D)
        cross_2D = np.cross(manipulability_2D_direction_unit, d_goal_unit_2D)
        # angle은 -π에서 +π 사이의 값
        angle_e = np.arctan2(cross_2D, dot_2D)
        # print(f"==>> angle: {angle_e}")
        # print(f"3D 내적 값 (유사도): {similarity}")
        # print(f"2D 내적 값 (유사도): {similarity_2D}")
        time_interval = 0.05
        angle_con_value = 1.0

        print('distance: ', np.linalg.norm(d_goal))
        if np.linalg.norm(d_goal) < 0.024:
            sim.setJointTargetVelocity(j1_h, 0)
            sim.setJointTargetVelocity(j2_h, 0)
            sim.setJointTargetVelocity(j3_h, 0)
            sim.setJointTargetVelocity(j4, 0)
            sim.setJointTargetVelocity(j5, 0)
            sim.setJointTargetVelocity(j6, 0)

            # 최대 힘 제한을 설정
            maxForce = 100  # 힘의 크기 제한
            sim.setJointForce(j1_h, maxForce)
            sim.setJointForce(j2_h, maxForce)
            sim.setJointForce(j3_h, maxForce)
            sim.setJointForce(j4_h, maxForce)
            sim.setJointForce(j5_h, maxForce)
            sim.setJointForce(j6_h, maxForce)
            l_v = 0.0
            a_v = 0.0
            w_R, w_L, _ = coppel_scout2_controller(l_v, a_v)

        else:
            l_v, a_v = kinematic_control(d_goal_2D[0]*10, d_goal_2D[1]*10) #cm
            # print(f"==>> d_goal_2D: {d_goal_2D}")
            
            d_goal_v = d_goal/(total_time)
            # d_goal_unit = d_goal/np.linalg.norm(d_goal)
            epsilon = 1e-6  # 작은 특이값에 대한 임계값
            Sigma_inv = np.diag(1.0 / (Sigma + epsilon))  # 작은 특이값에 임계값을 더하여 안정화
            # pseudo inverse jacobian matrix
            J_pseudo = Vt.T @ Sigma_inv @ U.T
            
            
            q_dot = J_pseudo @ d_goal_v
            # 최대 속도 한계 설정 (예시)
            max_q_dot = 1.0 # 최대 속도 한계를 설정

            # 속도가 한계를 초과하면 제한
            q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)
            # print(q_dot)
            j1_tv = q_dot[0]
            j2_tv = q_dot[1]
            j3_tv = q_dot[2]

            # j4_tv = q_dot[3]
            # j5_tv = q_dot[4]
            # j6_tv = q_dot[5]
            w_R, w_L, wc = coppel_scout2_controller(l_v, a_v)
            print(f"==>> w_L: {w_L}")
            print(f"==>> w_R: {w_R}")
            sim.setJointTargetVelocity(j1_h, j1_tv)#-wc*angle_con_value)
            sim.setJointTargetVelocity(j2_h, j2_tv)
            sim.setJointTargetVelocity(j3_h, j3_tv)
            sim.setJointTargetVelocity(j4, 0)
            sim.setJointTargetVelocity(j5, 0)
            sim.setJointTargetVelocity(j6, 0)

            sim.addDrawingObjectItem(line_handle, None)
            sim.addDrawingObjectItem(line_handle_, None)

            theta += wc*angle_con_value*time_interval
            # print(f"==>> theta: {theta}")
        moveToAngle_R(w_R)
        moveToAngle_L(w_L)

        client.step()  # Advance simulation by one step
except Exception as e:
    print(f"Error during simulation step: {e}")
finally:
    sim.stopSimulation()