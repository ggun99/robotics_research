import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys
import termios
import tty
import threading

class KeyboardMarkerControl(Node):
    def __init__(self):
        super().__init__('keyboard_marker_control')
        self.pose_pub = self.create_publisher(PoseStamped, "marker_pose", 10)

        # 마커 초기 위치
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0.0
        self.pose.pose.position.y = 0.0
        self.pose.pose.position.z = 0.0
        self.pose.header.frame_id = "map"

        # 키보드 입력을 별도 쓰레드에서 처리
        thread = threading.Thread(target=self.keyboard_loop)
        thread.daemon = True
        thread.start()

    def publish_pose(self):
        self.pose_pub.publish(self.pose)
        self.get_logger().info(
            f"x={self.pose.pose.position.x:.2f}, "
            f"y={self.pose.pose.position.y:.2f}, "
            f"z={self.pose.pose.position.z:.2f}"
        )

    def keyboard_loop(self):
        print("Use WASD to move X/Y, QE to move Z. Ctrl+C to quit.")
        print("W: +Y, S: -Y, A: -X, D: +X, Q: +Z, E: -Z")
        settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                key = sys.stdin.read(1)
                if key == 'w':
                    self.pose.pose.position.y += 0.1
                elif key == 's':
                    self.pose.pose.position.y -= 0.1
                elif key == 'a':
                    self.pose.pose.position.x -= 0.1
                elif key == 'd':
                    self.pose.pose.position.x += 0.1
                elif key == 'q':
                    self.pose.pose.position.z += 0.1
                elif key == 'e':
                    self.pose.pose.position.z -= 0.1
                elif key == '\x03':  # Ctrl+C
                    break
                else:
                    continue

                self.publish_pose()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardMarkerControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
