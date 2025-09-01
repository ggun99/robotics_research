import rclpy
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


class InteractiveMarkerDemo(Node):
    def __init__(self):
        super().__init__('interactive_marker_demo')

        # Interactive Marker 서버 생성
        self.server = InteractiveMarkerServer(self, "simple_marker")

        # 퍼블리셔: 마커 좌표 퍼블리시용
        self.pose_pub = self.create_publisher(PoseStamped, "marker_pose", 10)

        # 마커 생성
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = "my_marker"
        int_marker.description = "Drag me!"
        int_marker.scale = 1.0
        int_marker.pose.position.x = 0.0
        int_marker.pose.position.y = 0.0
        int_marker.pose.position.z = 0.0

        # 실제 보여지는 구체 마커
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

        # 이동 가능하게 만들기 (XY plane)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_xy"
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        int_marker.controls.append(move_control)

        # 마커 서버에 삽입
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        pose = PoseStamped()
        pose.header = feedback.header
        pose.pose = feedback.pose
        self.pose_pub.publish(pose)
        self.get_logger().info(f"Marker moved to: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}, z={pose.pose.position.z:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveMarkerDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
