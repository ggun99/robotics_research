import time
import sys
import os
import math

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient

offset = 0.02
dTol = 0.005
integral_dist = 0.0
previous_err_dist = 0.0
derivative_dist = 0.0
maxForce = 100

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

sim.startSimulation()
dummy_position = sim.getObjectHandle('/Dummy')
f_L = sim.getObject('/base_link_respondable/front_left_wheel')
f_R = sim.getObject('/base_link_respondable/front_right_wheel')
r_L = sim.getObject('/base_link_respondable/rear_left_wheel')
r_R = sim.getObject('/base_link_respondable/rear_right_wheel')
scout_base = sim.getObject('/base_link_respondable')

velocity = [0.005, 0.005, 0.005]  # Change per simulation step (for each axis)

def pid_scout(pose_dummy, pose_scout, integral_dist, derivative_dist, previous_err_dist):
    # Calculate errors in position
    distance_x = np.abs(pose_dummy[0] - pose_scout[0])
    e_x = distance_x - offset
    err_dist = e_x
       
    Kp_dist = 0.01
    Ki_dist = 0.1
    Kd_dist = 0.08
        
    integral_dist += err_dist
    # Prevent integral windup
    integral_dist = min(max(integral_dist, -10), 10)
    
    derivative_dist = err_dist - previous_err_dist

    if err_dist >= dTol:
        l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
        previous_err_dist = err_dist
    else:
        print("Scout2.0 stopping - distance within tolerance")
        l_v = 0.0    

    vc = float(l_v)
    r = 0.165
    w_R = vc / r  # Convert linear velocity to angular velocity
    return w_R

def moveToAngle(w_R):
    vel = w_R
    sim.setJointTargetVelocity(f_L, vel)
    sim.setJointTargetVelocity(f_R, -vel)
    sim.setJointTargetVelocity(r_L, vel)
    sim.setJointTargetVelocity(r_R, -vel)

try:
    while (t := sim.getSimulationTime()) < 25:
        s = f'Simulation time: {t:.2f} [s]'
        print(s)
        
        # Get current position
        pose_dummy = sim.getObjectPosition(dummy_position, -1)
        print(f'Current pose: {pose_dummy}')
        
        # Update the dummy position over time
        new_position = [pose_dummy[0] + velocity[0], pose_dummy[1], pose_dummy[2]]
        sim.setObjectPosition(dummy_position, -1, new_position)
        print(f'New pose: {new_position}')
        
        pose_scout = sim.getObjectPosition(scout_base, -1)
        print(f'Scout pose: {pose_scout}')
    
        w_R = pid_scout(pose_dummy, pose_scout, integral_dist, derivative_dist, previous_err_dist)
        moveToAngle(w_R)
        
        client.step()  # Advance simulation by one step
except Exception as e:
    print(f"Error during simulation step: {e}")
finally:
    sim.stopSimulation()