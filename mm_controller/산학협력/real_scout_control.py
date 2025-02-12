import time
from scipy.signal import butter, lfilter
from importlib import simple
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
import numpy as np
from zmqRemoteApi import RemoteAPIClient
from cubic_bezier_ import Bezier
from backstepping import Backstepping



class RealScoutControl():
    def __init__(self):
        super().__init__('current_odom_subscriber')
        self.subscription = self.create_subscription(
            Pose,
            '/aruco_pose',
            self.listener_callback,
            10)
        self.subscription 
        self.controlpublisher = self.create_publisher(Twist,'/cmd_vel', 10)
        self.dTol = 0.05 # distance Tolerance? 
        self.state = False
        self.timer_period = 0.05
        self.x_cur = 0.0
        self.y_cur = 0.0
        self.offset = 1.0
        self.x_desired = 0.0
        self.y_desired = 0.0
        self.y_e = 0.0
        self.timer = self.create_timer(self.timer_period, self.scout_control)
        self.start_time = self.get_clock().now().to_msg().sec
        self.start_timet = time.time()
        self.control_points = np.zeros((3, 2))
        self.value_locked = False  # 값이 고정되었는지 여부
        self.x_d = 0
        self.y_d = 0 
        self.x_dp = 0
        self.y_dp = 0 
        self.t_robot = 0
        self.r = 0.165
        self.l = 0.582
        self.Bezier = Bezier()
        self.Backstepping = Backstepping()
        value_locked = False  # 값이 고정되었는지 여부

    # def velocity_control(self, y_e, t_robot, x_desired, y_desired):
    #     print(self.control_points)
    #     self.x_d, self.y_d, x_dp, y_dp, x_dpp, y_dpp = self.Bezier.bezier_curve(control_points=self.control_points, t_robot=t_robot)

    #     vc, wc = self.Backstepping.backstepping_control(self.x_d, self.y_d, x_desired, y_desired, x_dp, y_dp, x_dpp, y_dpp, y_e, K1=12, K2=3, K3=3)

    #     vc = float(vc)
    #     if vc > 0.22:
    #         vc = 0.22
    #     elif vc < -0.22:
    #         vc = -0.22
    #     wc = float(wc)
    #     if wc > 0.2:
    #         wc = 0.2
    #     elif wc < -0.2:
    #         wc = -0.2
    #     print(vc,wc)
    #     return vc, wc

    def butter_lowpass(self, cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        # 필터 적용 함수
    def butter_lowpass_filter(self, data, cutoff, fs, order=10):
            b, a = self.butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

    def listener_callback(self, msg):
        # position
        x_p = msg.position.x
        y_p = msg.position.y
        z_p = msg.position.z
        # quarternion
        x_q = msg.orientation.x
        y_q = msg.orientation.y
        z_q = msg.orientation.z
        w_q = msg.orientation.w
        # quarternion to euler
        x, y, z = self.euler_from_quaternion(x_q, y_q, z_q, w_q)
        # x_start, offset, y_desired, total_time
        # print(f'1: {self.x_start}')
        
        # print(f'2: {self.x_start}')
        if not self.value_locked:  # 처음 한 번만 값 저장
            self.x_desired = x_p
            self.y_desired = y_p
            self.control_points[1][0] = self.x_desired*3/5
            self.control_points[1][1] = self.y_desired*4/5
            self.control_points[2][0] = self.x_desired
            self.control_points[2][1] = self.y_desired
            self.value_locked = True
        self.get_logger().info(f"🔒 값 저장: {self.x_desired, self.y_desired}")
        
        # euler
        # self.x_e = x
        self.y_e = y
        # self.z_e = z

    def scout_control(self):
        self.t_robot = time.time()-self.start_timet
        self.x_d, self.y_d, x_dp, y_dp, x_dpp, y_dpp = self.Bezier.bezier_curve(control_points=self.control_points, t_robot=self.t_robot)

        vc, wc = self.Backstepping.backstepping_control(self.x_d, self.y_d, self.x_desired, self.y_desired, x_dp, y_dp, x_dpp, y_dpp, self.y_e, K1=12, K2=3, K3=3)

        vc = float(vc)
        if vc > 0.22:
            vc = 0.22
        elif vc < -0.22:
            vc = -0.22
        wc = float(wc)
        if wc > 0.2:
            wc = 0.2
        elif wc < -0.2:
            wc = -0.2
        print(vc,wc)

        twist = Twist()
        twist.linear.x = vc
        twist.angular.z = wc
        self.controlpublisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    simple_controller = RealScoutControl()
    rclpy.spin(simple_controller)
    simple_controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()    
