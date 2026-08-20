import rclpy
from rclpy.node import Node
import time
import rbpodo as rb

class TeleoperationRB3(Node):
    def __init__(self):
        super().__init__('teleoperation_rb3')
        self.get_logger().info('TeleoperationRB3 node has been started.')

        #About Robot
        ROBOT_IP_FOLLOWER = "192.168.0.74"
        ROBOT_IP_LEADER = "192.168.0.201"
        self.leader_robot = rb.Cobot(ROBOT_IP_LEADER)
        self.follower_robot = rb.Cobot(ROBOT_IP_FOLLOWER)

        self.rc = rb.ResponseCollector()
        self.data_channel_leader = rb.CobotData(ROBOT_IP_LEADER)
        self.data_channel_follower = rb.CobotData(ROBOT_IP_FOLLOWER)

        #Timers
        self.timers_get_leader = self.create_timer(0.01, self.loop_get_leader_joints)
        self.timers_get_follower = self.create_timer(0.01, self.loop_get_follower_joints)
        self.timers_set_follower = self.create_timer(0.01, self.loop_set_follower_joints)

        self.leader_robot_joints = None
        self.follower_robot_joints = None

        #Publishers and Subscribers can be defined here
        # self.pub_leader_joints = self.create_publisher()
        # self.pub_follower_joints = self.create_publisher()

    def loop_get_leader_joints(self):
        self.leader_robot_joints = self.data_channel_leader.request_data().sdata.jnt_ref
        print(f"==>> self.leader_robot_joints: {self.leader_robot_joints}")

    def loop_get_follower_joints(self):
        self.follower_robot_joints = self.data_channel_follower.request_data().sdata.jnt_ref
        print(f"==>> self.follower_robot_joints: {self.follower_robot_joints}")


    def loop_set_follower_joints(self):
        if self.leader_robot_joints is not None:
            self.follower_robot.move_j(self.rc, self.leader_robot_joints, 60, 80)
            print("움직임")

def main(args=None):
    rclpy.init(args=args)
    node = TeleoperationRB3()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()