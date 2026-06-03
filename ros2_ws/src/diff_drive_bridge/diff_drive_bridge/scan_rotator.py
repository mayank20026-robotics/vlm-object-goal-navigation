import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ScanRotator(Node):
    def __init__(self):
        super().__init__('scan_rotator')
        self.declare_parameter('angle_offset', -1.5708)  # -90° clockwise
        self.offset = self.get_parameter('angle_offset').value

        self.sub = self.create_subscription(
            LaserScan, '/scan_raw', self.callback, 10)
        self.pub = self.create_publisher(
            LaserScan, '/scan', 10)

        self.get_logger().info(f'Scan rotator started with offset: {self.offset:.4f} rad')

    def callback(self, msg):
        rotated = LaserScan()
        rotated.header       = msg.header
        rotated.angle_min    = msg.angle_min    + self.offset
        rotated.angle_max    = msg.angle_max    + self.offset
        rotated.angle_increment = msg.angle_increment
        rotated.time_increment  = msg.time_increment
        rotated.scan_time    = msg.scan_time
        rotated.range_min    = msg.range_min
        rotated.range_max    = msg.range_max
        rotated.ranges       = msg.ranges
        rotated.intensities  = msg.intensities
        self.pub.publish(rotated)

def main(args=None):
    rclpy.init(args=args)
    node = ScanRotator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
