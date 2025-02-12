import time
import sys
import os
import math
from scipy.signal import butter, lfilter

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
t_robot = 0
r = 0.165
l = 0.582

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

sim.startSimulation()
dummy_position = sim.getObjectHandle('/Dummy')
# scout_position = sim.getObjectHandle('/Dummy')

f_L = sim.getObject('/base_link_respondable/front_left_wheel')
f_R = sim.getObject('/base_link_respondable/front_right_wheel')
r_L = sim.getObject('/base_link_respondable/rear_left_wheel')
r_R = sim.getObject('/base_link_respondable/rear_right_wheel')
scout_base = sim.getObject('/base_link_respondable')
pose_scout = sim.getObjectPosition(scout_base, -1)
pose_dummy = sim.getObjectPosition(dummy_position, -1)




velocity = [0.005, 0.005, 0.005]  # Change per simulation step (for each axis)
x_cur = 0
y_cur = 0
control_points = np.zeros((3, 2))
value_locked = False  # 값이 고정되었는지 여부


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

def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    # 필터 적용 함수
def butter_lowpass_filter(data, cutoff, fs, order=10):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

def backstepping_control( x, y, x_d, y_d, x_dp, y_dp, x_dpp, y_dpp, theta, K1=1.0, K2=0.1, K3=1.0):
        e_x = x_d - x
        e_y = y_d - y
        e_theta = np.arctan2(y_dp, x_dp) - theta
        # Filter requirements.
        # cutoff = 5.0  # 저역통과 필터의 컷오프 주파수
        # timer_period = 0.05
        # fs = 1/timer_period    # 프레임 속도 (초당 프레임)
        # order = 3     # 필터 차수
        # 좌표 값 버퍼 크기 조정 (필터링할 데이터 크기 유지)
        # if len(e_theta_list) > 10:
        #     e_theta_list.pop(0)
        
        # # 데이터가 충분할 때 필터 적용
        # if len(e_theta_list) > order:
        #     filtered_e_theta = butter_lowpass_filter(e_theta_list, cutoff, fs, order)
        #     e_theta = filtered_e_theta
        # print('e_theta', e_theta)
        T_e = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
       
        mat_q = T_e @ np.array([[e_x],[e_y],[e_theta]])

        v_r = np.sqrt(x_dp**2+y_dp**2)
        if np.abs(x_dp**2 + y_dp**2) < 0.01:
            w_r = 0.0
        else: 
            w_r = (x_dp*y_dpp - y_dp*x_dpp)/(x_dp**2 + y_dp**2)
        print('mat_q[2,0]', mat_q[2,0])
        print('mat_q[0,0]', mat_q[0,0])
        v_c = v_r*np.cos(mat_q[2,0]) + K1*mat_q[0,0]
        # v_c = -v_c
        w_c = w_r + K2*v_r*mat_q[1,0] + K3*np.sin(mat_q[2,0])

        return v_c, w_c

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

def bezier_curve(control_points, t_robot, total_time=10):
    t = t_robot/total_time
    """Calculates a Bezier curve from a set of control points."""
    x_d = control_points[0][0]*(1-t)**2 + control_points[1][0]*2*t*(1-t) + control_points[2][0]*t**2
    y_d = control_points[0][1]*(1-t)**2 + control_points[1][1]*2*t*(1-t) + control_points[2][1]*t**2
    print(x_d)
    print(y_d)
    timer_period = 0.05
    # x_dp = (x_d-x_cur)/timer_period
    # y_dp = (y_d-y_cur)/timer_period
    x_dp = -2*control_points[0][0] + control_points[0][0]*2*t + 2*control_points[1][0] - 4*control_points[1][0]*t + 2*control_points[2][0]*t
    y_dp = -2*control_points[0][1] + control_points[0][1]*2*t + 2*control_points[1][1] - 4*control_points[1][1]*t + 2*control_points[2][1]*t

    x_dpp = 2*control_points[0][0] - 4*control_points[1][0] + 2*control_points[2][0]
    y_dpp = 2*control_points[0][1] - 4*control_points[1][1] + 2*control_points[2][1]

    print(x_dp)
    print(y_dp)
    print(t)
    return x_d, y_d, x_dp, y_dp, x_dpp, y_dpp


try:
    while (t := sim.getSimulationTime()) < 20:
        s = f'Simulation time: {t:.2f} [s]'
        print(s)
        x_d, y_d, x_dp, y_dp, x_dpp, y_dpp = bezier_curve(control_points,t_robot)
        # time_step = timer_period
        start_time = time.time()
        x_cur = x_d
        y_cur = y_d
        vc, wc = backstepping_control(x_cur, y_cur, x_desired, y_desired, x_dp, y_dp, x_dpp, y_dpp, y_e, K1=12, K2=3, K3=3)

        vc = float(vc)
        if vc > 0.2:
            vc = 0.2
        elif vc < -0.2:
            vc = -0.2
        wc = float(wc)
        if wc > 0.1:
            wc = 0.1
        elif wc < -0.1:
            wc = -0.1
        x_desired = pose_dummy[0]
        y_desired = pose_dummy[1]
        control_points[1][0] = x_desired*3/5
        control_points[1][1] = y_desired*4/5
        control_points[2][0] = x_desired
        control_points[2][1] = y_desired
        
        print(f'Scout pose: {pose_scout}')
        y_e = np.arctan((pose_scout[1] - pose_dummy[1])/(pose_scout[0] - pose_dummy[0]))
        vc, wc = scout_control(y_e, t, x_desired, y_desired)
        
        w_R = vc/r + l*wc/(2*r)
        w_L = vc/r - l*wc/(2*r) # Convert linear velocity to angular velocity
        moveToAngle_R(w_R)
        moveToAngle_L(w_L)
        
        client.step()  # Advance simulation by one step

except Exception as e:
    print(f"Error during simulation step: {e}")
finally:
    sim.stopSimulation()