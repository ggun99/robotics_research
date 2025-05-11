import time
import sys
import os
from turtle import distance
import scipy.linalg

import rclpy.time
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from cv2 import waitKey
import numpy as np
import rtde_control
import rtde_receive
from jacobian_v1 import Jacobian
from admittance_controller_v1 import AdmittanceController
import rclpy
from rclpy.node import Node 
from scipy.signal import butter, lfilter
import threading
from geometry_msgs.msg import Twist, WrenchStamped
import math
import socket

class UR5e_controller(Node):
    def __init__(self):
        super().__init__('UR5e_node')
        self.ROBOT_IP = '192.168.0.3'  # 로봇의 IP 주소로 바꿔주세요  212
        # RTDE 수신 객체 생성
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
        # RTDE Control Interface 초기화
        self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
        self.rtde_c.zeroFtSensor()
        # """Dashboard Server에 명령 전송"""
        # try:
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         s.connect((self.ROBOT_IP, 29999))
        #         s.sendall(("zero_ftsensor()" + "\n").encode())
        #         response = s.recv(1024).decode()
        #     print(f"Dashboard Response: {response}")
        # except Exception as e:
        #     print(f"Error: {e}")
        # self.dTol = 0.005
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
        self.previous_time = time.time() 
        self.Jacobian = Jacobian()
        M = np.diag([10.0, 10.0, 10.0, 0.3, 0.3, 0.3])  # Mass matrix
        B = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])  # Damping matrix
        K = np.diag([350.0, 350.0, 350.0, 0.0, 0.0, 0.0])  # Stiffness matrix
        self.admit = AdmittanceController(M, B, K)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.admittance)
         # ft data list 저장
        self.ft_x = []
        self.ft_y = []
        self.ft_z = []
        self.emergency_stop = False
        # Start a thread to listen for emergency stop input
        self.emergency_stop_thread = threading.Thread(target=self.emergency_stop_listener)
        self.emergency_stop_thread.daemon = True
        self.emergency_stop_thread.start()
        # mobile robot
        self.dTol = 0.01 # distance Tolerance? 
        self.controlpublisher = self.create_publisher(Twist,'/cmd_vel', 10)
        self.ftpublisher = self.create_publisher(WrenchStamped,'/ur_ftsensor', 10)
        self.K1 = 0.2
        self.K2 = 0.2
        self.integral_dist = 0
        self.previous_err_x = 0
        self.previous_err_y = 0
        self.ref_force = 15.0

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians  

    def kinematic_control(self, e_x, e_y):
        if e_x > 0:
            Kp_x = 2.5
        else:
            Kp_x = 0.5
        # Kp_x = 1.0
        Ki_x = 0.01
        Kd_x = 0.8
        Kp_y = 0.5
        Ki_y = 0.01
        Kd_y = 0.8
        
        self.integral_dist += e_x
        # Prevent integral windup
        self.integral_dist = min(max(self.integral_dist, -0.05), 0.05)
        
        derivative_dist = e_x - self.previous_err_x
        derivative_theta = e_y - self.previous_err_y
        vc = Kp_x * e_x + Ki_x * self.integral_dist + Kd_x * derivative_dist
        # print('vc', vc)
        if -self.dTol < vc < self.dTol:
            vc = 0.0
            print("Scout2.0 stopping - distance within tolerance")
            self.previous_err_x = 0.0
        elif vc > 0.1 :
            vc = 0.1
            self.previous_err_x = e_x
        elif vc < -0.1 :
            vc = -0.1
            self.previous_err_x = e_x
        else:
            vc = vc

        wc = Kp_y * e_y + Ki_y * self.integral_dist + Kd_y * derivative_theta
        # print('wc', wc)
        if -self.dTol < wc < self.dTol:
            # wc = Kp_y * abs(e_y) + Ki_y * self.integral_dist + Kd_y * derivative_dist
            wc = 0.0
            print("Scout2.0 stopping - distance within tolerance")
            self.previous_err_y = 0.0
        elif wc > 0.1 :
            wc = 0.1
            self.previous_err_y = e_y
        elif wc < -0.1 :
            wc = -0.1
            self.previous_err_y = e_y
        else:
            wc = wc 

       
        return vc, wc #np.array([[vc], [wc]])

    def emergency_stop_listener(self):
        #  """Listen for Enter key to activate emergency stop."""
        print("Press Enter to activate emergency stop.")
        while True:
            input()  # Wait for Enter key press
            self.emergency_stop = True
            self.get_logger().warn("Emergency stop activated!")

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    # 필터 적용 함수
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def admittance(self):
        # self.rtde_c.set_gravity([0, 0, -9.81])
        t_start = self.rtde_c.initPeriod()
        wrench = self.rtde_r.getActualTCPForce()   # 중력 혹은 다른 힘들이 보정이 된 TCP 에서 측정된 힘
        # TCPpose = self.rtde_r.getActualTCPPose()
        # TCPpose = np.array(TCPpose)
        # raw_wrench = self.rtde_r.getFtRawWrench()   # 중력 혹은 다른 힘들이 일체 보정이 되지 않은 raw 데이터
        # 6D 벡터: [Fx, Fy, Fz, Tx, Ty, Tz]
        # print("Force/Torque values:", wrench)
        # print("Raw Force/Torque values:", raw_wrench)
        ft = wrench
        wrenchstamp = WrenchStamped()
        wrenchstamp.wrench.force.x = wrench[0]
        wrenchstamp.wrench.force.y = wrench[1]
        wrenchstamp.wrench.force.z = wrench[2]
        self.ftpublisher.publish(wrenchstamp)
        

def main(args=None):
    rclpy.init(args=args)
    listener = UR5e_controller()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown() 
    listener.rtde_c.servoStop()
    listener.rtde_c.stopScript()


print('Program started')

if __name__ == '__main__':
    waitKey(100)
    import time
    time.sleep(1)
    main()