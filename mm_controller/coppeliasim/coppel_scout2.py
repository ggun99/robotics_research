import time
import sys
import os
import math

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
from cv2 import waitKey
import numpy as np
from zmqRemoteApi import RemoteAPIClient

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

client.setStepping(True)

sim.startSimulation()
dummy_position = sim.getObjectHandle('/Bill/Bill')
f_L = sim.getObject('/base_link_respondable/front_left_wheel')
f_R = sim.getObject('/base_link_respondable/front_right_wheel')
r_L = sim.getObject('/base_link_respondable/rear_left_wheel')
r_R = sim.getObject('/base_link_respondable/rear_right_wheel')
scout_base = sim.getObject('/base_link_respondable/Dummy')

j1 = sim.getObject('/base_link_respondable/UR5/joint')
j2 = sim.getObject('/base_link_respondable/UR5/joint/joint')
j3 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint')
j4 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint')
j5 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint')
j6 = sim.getObject('/base_link_respondable/UR5/joint/joint/joint/joint/joint/joint')

velocity = [0.003, 0.003, 0.005]  # Change per simulation step (for each axis)

def pid_scout(pose_dummy, pose_scout, integral_dist, previous_err_dist, integral_theta, previous_err_theta):
    # Calculate errors in position
    distance_x = pose_dummy[0] - pose_scout[0]
    
    err_dist = distance_x - offset
    err_theta = pose_dummy[1] - pose_scout[1]

    Kp_dist = 0.03
    Ki_dist = 0.1
    Kd_dist = 0.08
    Kp_theta = 0.8
    Ki_theta = 0.1
    Kd_theta = 0.01
            
    integral_dist += err_dist
    # Prevent integral windup
    integral_dist = min(max(integral_dist, -10), 10)
    
    derivative_dist = err_dist - previous_err_dist

    integral_theta += err_theta
    integral_theta = min(max(integral_theta, -10), 10)
    derivative_theta = err_theta - previous_err_theta
    print(f'err_dist: {err_dist}')
    print(f'err_theta: {err_theta}')
    if np.abs(err_dist) >= dTol:
        l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
        previous_err_dist = err_dist
    else:
        print("Scout2.0 stopping - distance within tolerance")
        l_v = 0.0    
    # PID control for angular velocity
    if np.abs(err_theta) >= dTol: #checking whether heading angle error within tolerence
        a_v = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
        previous_err_theta = err_theta
        
    else:
        print(f"Scout2.0  stopping goal heading within tolerence")
        a_v = 0.0 
    vc = float(l_v)
    wc = -float(a_v)
    r = 0.165
    l = 0.582
    w_R = vc/r + l*wc/(2*r)
    w_L = vc/r - l*wc/(2*r) # Convert linear velocity to angular velocity
    return w_R, w_L

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

try:
    waitKey(1)
    while (t := sim.getSimulationTime()) < 50:
        s = f'Simulation time: {t:.2f} [s]'
        print(s)
        
        # Get current position
        pose_dummy = sim.getObjectPosition(dummy_position, -1)
        print(f'Current pose: {pose_dummy}')
        
        # Update the dummy position over time
        new_position = [pose_dummy[0] + velocity[0], pose_dummy[1] + velocity[1], pose_dummy[2]]
        sim.setObjectPosition(dummy_position, -1, new_position)
        # print(f'New pose: {new_position}')
        
        pose_scout = sim.getObjectPosition(scout_base, -1)
        print(f'Scout pose: {pose_scout}')
    
        w_R, w_L = pid_scout(pose_dummy, pose_scout, integral_dist, previous_err_dist, integral_theta, previous_err_theta)
        moveToAngle_R(w_R)
        moveToAngle_L(w_L)
        
        client.step()  # Advance simulation by one step
except Exception as e:
    print(f"Error during simulation step: {e}")
finally:
    sim.stopSimulation()