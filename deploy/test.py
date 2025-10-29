#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class AprilTagSubscriber(Node):
    def __init__(self):
        super().__init__('apriltag_subscriber')
        # Create subscriber to /apriltag/info topic
        self.subscription = self.create_subscription(
            Point,
            '/apriltag/info',
            self.listener_callback,
            10  # QoS queue depth
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Subscribed to /apriltag/info')

    def listener_callback(self, msg: Point):
        # Print every message received
        self.get_logger().info(f"Received: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
