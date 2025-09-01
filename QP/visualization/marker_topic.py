import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)  # 1초마다 퍼블리시
        self.get_logger().info("Marker Publisher Node Started")

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"  # RViz에서 맞는 TF frame으로 변경
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "example"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # scale: POINTS는 x,y 값이 width/height를 의미
        marker.scale.x = 0.1  # 점의 가로 크기
        marker.scale.y = 0.1  # 점의 세로 크기

        # 색상 (전체 공통 색 지정 가능)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 빨간색

        # Points 추가
        points = []
        for i in range(10):
            p = Point()
            p.x = float(i) * 0.1
            p.y = float(i) * 0.1
            p.z = 0.0
            points.append(p)

        marker.points = points

        self.publisher.publish(marker)
        self.get_logger().info("Published POINTS Marker")


def main(args=None):
    rclpy.init(args=args)
    node = MarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()