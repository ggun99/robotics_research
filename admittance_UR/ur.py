import time
import sys
import os
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from cv2 import waitKey
import numpy as np
from jacobian_v1 import Jacobian
from admittance_controller_v1 import AdmittanceController
import rclpy
from rclpy.node import Node 
from scipy.signal import butter, lfilter
import threading
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class UR5e_controller(Node):
    def __init__(self):
        super().__init__('UR5e_node')


        self.maxForce = 100
        self.Jacobian = Jacobian()
        M = np.diag([10.0, 10.0, 10.0, 0.3, 0.3, 0.3])  # Mass matrix
        B = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])  # Damping matrix
        K = np.diag([350.0, 350.0, 350.0, 0.0, 0.0, 0.0])  # Stiffness matrix
        self.admit = AdmittanceController(M, B, K)
        self.dt = 0.05
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.wrench_sub = self.create_subscription(WrenchStamped,'/force_torque_sensor_broadcaster/wrench', self.listener_callback, 10)
        self.emergency_stop = False
        # Start a thread to listen for emergency stop input
        self.emergency_stop_thread = threading.Thread(target=self.emergency_stop_listener)
        self.emergency_stop_thread.daemon = True
        self.emergency_stop_thread.start()
        self.dTol = 0.01 # distance Tolerance? 
        self.ur_control = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
  
        self.ref_force = 15.0
        self.wrench = np.array([])
        # ft data list 저장
        self.ft_x = []
        self.ft_y = []
        self.ft_z = []
        self.joint_q = np.array([])

    def joint_callback(self, msg):
        self.joint_q = np.array([msg.position[5], msg.position[0], msg.position[1], msg.position[2], msg.position[3], msg.position[4]])
        # self.joint_q = 
        print('joint_q', self.joint_q)

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
    

    def listener_callback(self, msg):
        # self.wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, 0., 0., 0.])
        self.wrench = np.array([-30.0, 0.0, 0.0, 0., 0., 0.])

        ft = self.wrench
        
        if ft is not None:
            # Filter requirements.
            cutoff = 30.0  # 저역통과 필터의 컷오프 주파수
            fs = 100.0     # 프레임 속도 (초당 프레임)
            order = 3     # 필터 차수
            self.ft_x.append(ft[0])
            self.ft_y.append(ft[1])
            self.ft_z.append(ft[2])
            print('ft_nfilt', ft)
            if len(self.ft_x) > 5 :
                self.ft_x.pop(0)
                self.ft_y.pop(0)
                self.ft_z.pop(0)
            # 데이터가 충분할 때 필터 적용
            if len(self.ft_x) > order:
                filtered_ft_x = self.butter_lowpass_filter(self.ft_x, cutoff, fs, order)
                filtered_ft_y = self.butter_lowpass_filter(self.ft_y, cutoff, fs, order)
                filtered_ft_z = self.butter_lowpass_filter(self.ft_z, cutoff, fs, order)
                print(filtered_ft_x)
                ft[0] = filtered_ft_x[-1]
                ft[1] = filtered_ft_y[-1]
                ft[2] = filtered_ft_z[-1]
                print('ft_filtered', ft)

        
        delta_position, _ = self.admit.update(ft, self.dt)
        joint_positions = self.joint_q
       
    
        # new jac
        J = self.Jacobian.jacobian(joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3], joint_positions[4], joint_positions[5])
        j_J = J #@ J.T
        
        # SVD 계산
        try:
            U, Sigma, Vt = np.linalg.svd(j_J, full_matrices=False, hermitian=False)
            self.U_previous, self.Sigma_previous, self.Vt_previous = U, Sigma, Vt  # 이전 값 업데이트
        except ValueError as e:
            print("SVD computation failed due to invalid input:")
            print(e)
            U, Sigma, Vt = self.U_previous, self.Sigma_previous, self.Vt_previous  # 이전 값 사용
        
        max_q_dot = 0.1 # 최대 속도 한계를 설정

        # pseudo inverse jacobian matrix
        J_pseudo_inv = np.linalg.pinv(J)
        # print(f"==>> J_pseudo_inv: {J_pseudo_inv}")

        d_goal_v = delta_position
        print(f"==>> delta_position: {delta_position}")
        # print(f"==>> d_goal_v: {d_goal_v}")
        q_dot = J_pseudo_inv @ d_goal_v

        # cal_check = J @ q_dot
        # print('checking:',cal_check)

        # 속도가 한계를 초과하면 제한
        q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)
        q_dot = q_dot.flatten().tolist()
        q_dot = [0.0,0.,0.,0.,0.,0.]
        # print('q_dot1', q_dot)
        # acceleration = 0.2

        if abs(delta_position[0])<0.01:
            q_dot = [0.0,0.0,0.0,0.0,0.0,0.0]

        if self.emergency_stop:
            q_dot = [0.0,0.0,0.0,0.0,0.0,0.0]
        
        q_topic = Float64MultiArray()
        q_topic.data = q_dot
        print('q_dot2', q_dot)
        self.ur_control.publish(q_topic)

    def emergency_stop_listener(self):
        #  """Listen for Enter key to activate emergency stop."""
        print("Press Enter to activate emergency stop.")
        while True:
            input()  # Wait for Enter key press
            self.emergency_stop = True
            self.get_logger().warn("Emergency stop activated!")


def main(args=None):
    rclpy.init(args=args)
    listener = UR5e_controller()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown() 



print('Program started')

if __name__ == '__main__':
    # waitKey(3000)
    # import time
    # time.sleep(5)
    main()