#! usr/bin/env/ python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import pi
import serial
import time

class cmd_vel_listener(Node):
    def __init__(self):
        super().__init__('cmd_vel_listener')
        self.serial_port = "/dev/ttyUSB0"
        self.baudrate = 9600

        try:
            self.ser = serial.Serial(self.serial_port,self.baudrate,timeout= 1)
            time.sleep(2)
            self.get_logger().info(f"Connected to arduino through serial port: {self.serial_port}")            
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.ser = None

        self.subscription = self.create_subscription(Twist,'/cmd_vel',self.cmd_vel_callback, 10)
        self.get_logger().info("cmd_vel_serial_bridge node has started: ")
                

    def cmd_vel_callback(self,msg: Twist):
        v = msg.linear.x
        omega = msg.angular.z
        L = 0.38
        R = 0.055

        v_left = v - (omega*(L/2))
        v_right = v + (omega*(L/2))

        N_left = (v_left*60)/(2*pi*R)
        N_right = (v_right*60)/(2*pi*R)

        N_max = 100
        N_left = max(min(N_left,N_max),-N_max)
        N_right = max(min(N_right,N_max),-N_max)

        self.get_logger().info(f"N_left: {N_left} and N_right: {N_right}")

        try:
            data = f"{N_left:.3f},{N_right:.3f}\n"
            self.ser.write(data.encode("utf-8"))
        except Exception as e:
            self.get_logger().warn(f'Serial Write Failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = cmd_vel_listener()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()