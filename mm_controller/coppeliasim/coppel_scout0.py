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
# integral_theta = 0.0
# previous_err_theta = 0.0

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
# Define initial position and velocity
# initial_position = [0, 0, 0]  # Starting at the origin
velocity = [0.005, 0.005, 0.005]  # Change per simulation step (for each axis)

# dummy와 scout의 pose를 받아서 joint velocity 값을 리턴해야함. 
def pid_scout(pose_dummy, pose_scout, integral_dist, derivative_dist, previous_err_dist):
    # Calculate errors in position
        distance_x = np.abs(pose_dummy[0] - pose_scout[0])
        e_x = distance_x - offset
        # ro_distance = y_p
        # e_y = ro_distance
        err_dist = e_x
        # Error in heading
        # err_theta = e_y
       
        Kp_dist = 0.01
        Ki_dist = 0.1
        Kd_dist = 0.08
        # Kp_theta = 0.01
        # Ki_theta = 0.1
        # Kd_theta = 0.01
        
        integral_dist += err_dist
        derivative_dist = err_dist - previous_err_dist
        # integral_theta += err_theta
        # derivative_theta = err_theta - previous_err_theta
        # TODO: Add integral and derivative calculations for complete PID
        # PID control for linear velocity
        if err_dist >= dTol: #checking whether error distance within tolerence
            l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
            previous_err_dist = err_dist
        else:
            print(f"Scout2.0  stopping goal distance within tolerence")
            l_v = 0.0    
        # PID control for angular velocity
        # if err_theta >= self.dTol: #checking whether heading angle error within tolerence
        #     a_v = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
        #     previous_err_theta = err_theta
        # else:
        #     self.get_logger().info(f"Scout2.0  stopping goal heading within tolerence")
        #     a_v = 0.0      

        # Send the velocities
        vc = float(l_v)
        # wc = float(a_v)
        r = 0.165
        l = 0.582
        w_R = vc/r
        return w_R #, wc

def moveToAngle(w_R):
    
    # while abs(jointAngle - targetAngle) > 0.1 * math.pi / 180:
    vel = w_R
    sim.setJointTargetVelocity(f_L, vel)
    sim.setJointTargetVelocity(f_R, vel)
    sim.setJointTargetVelocity(r_L, vel)
    sim.setJointTargetVelocity(r_R, vel)
    sim.setJointTargetForce(f_L, maxForce, False)
    sim.setJointTargetForce(f_R, maxForce, False)
    sim.setJointTargetForce(r_L, maxForce, False)
    sim.setJointTargetForce(r_R, maxForce, False)
    
        # jointAngle = sim.getJointPosition(jointHandle)

while (t := sim.getSimulationTime()) < 25:
    s = f'Simulation time: {t:.2f} [s]'
    print(s)
    
    # Get current position
    pose_dummy = sim.getObjectPosition(dummy_position, -1)
    print(f'Current pose: {pose_dummy}')
    
    # Update the position over time
    new_position = [pose_dummy[0] + velocity[0], pose_dummy[1], pose_dummy[2]]
    # new_position = [pose_dummy[0] + velocity[0], pose_dummy[1] + velocity[1], pose_dummy[2] + velocity[2]]
    sim.setObjectPosition(dummy_position, -1, new_position)
    print(f'New pose: {new_position}')
    
    pose_scout = sim.getObjectPosition(scout_base, -1)
    print(f'scout_pose: {pose_scout}')

    w_R = pid_scout(pose_dummy, pose_scout, integral_dist, derivative_dist, previous_err_dist)
    moveToAngle(w_R)
    client.step()  # Advance simulation by one step

sim.stopSimulation()