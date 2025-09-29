import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

class MultipleCylinders(Node):
    def __init__(self):
        super().__init__('multiple_cylinders')
        self.publisher = self.create_publisher(MarkerArray, 'cylinder_markers', 10)

        # 타이머 1초마다 퍼블리시
        self.timer = self.create_timer(1.0, self.publish_markers)

    def publish_markers(self):
        marker_array = MarkerArray()
        positions = [
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (1.0, 1.0, 0.5)
        ]

        for i, (x, y, z) in enumerate(positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cylinders"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} cylinders")

def main(args=None):
    rclpy.init(args=args)
    node = MultipleCylinders()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
