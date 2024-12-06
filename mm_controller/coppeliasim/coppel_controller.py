import numpy as np

class coppel_controller():
    def __init__(self):
        pass
    def kinematic_control(self, integral_dist, previous_err_dist,integral_theta, previous_err_theta, e_x, e_y):
        dTol = 0.05
        
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
        
        # PID control for linear velocity
        
        if np.abs(err_dist) >= dTol: #checking whether error distance within tolerence
            l_v = Kp_dist * abs(err_dist) + Ki_dist * integral_dist + Kd_dist * derivative_dist
            previous_err_dist = err_dist
        else:
            #print(f"Scout2.0  stopping goal distance within tolerence")
            l_v = 0.0    

        # PID control for angular velocity
        if np.abs(err_theta) >= dTol: #checking whether heading angle error within tolerence
            a_v = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
            previous_err_theta = err_theta
            
        else:
            #print(f"Scout2.0  stopping goal heading within tolerence")
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

    def coppel_scout2_controller(self, l_v, a_v):
        vc = float(l_v)
        wc = float(a_v)
        r = 0.165
        l = 0.582
        w_R = vc/r + l*wc/(2*r)
        w_L = vc/r - l*wc/(2*r)
        return w_R, w_L, wc