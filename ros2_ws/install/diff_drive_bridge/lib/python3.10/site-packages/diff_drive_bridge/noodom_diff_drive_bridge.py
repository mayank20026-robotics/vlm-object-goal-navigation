#!/usr/bin/env python3
"""
diff_drive_bridge.py  —  ROS 2 Humble
Write-only bridge — sends velocity commands to Arduino.
rf2o_laser_odometry handles odometry from LiDAR.
No encoder reading, no serial lock conflicts.

Serial protocol:
  Jetson → Arduino: "V <left_rad_s> <right_rad_s>\n"
"""

import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

# ── Robot parameters ─────────────────────────────────────────
WHEEL_RADIUS  = 0.055   # metres
WHEEL_BASE    = 0.38    # metres
MAX_WHEEL_VEL = 5    # rad/s

# ── Serial ───────────────────────────────────────────────────
# Make sure your udev rules map your Mega to this, or use /dev/ttyACM0
SERIAL_PORT = "/dev/arduino" 
BAUD_RATE   = 115200

class DiffDriveBridge(Node):
    def __init__(self):
        super().__init__("diff_drive_bridge")

        self.current_vl = 0.0
        self.current_vr = 0.0

        # Subscribe to Nav2 / teleop cmd_vel
        self.cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_cb, 10)

        # Initialize Serial
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
            time.sleep(2.0)  # Wait for Arduino auto-reset on connect
            self.get_logger().info(f"Successfully opened {SERIAL_PORT}")
        except serial.SerialException as e:
            self.get_logger().fatal(f"Cannot open {SERIAL_PORT}: {e}")
            raise

        # 10Hz timer — pushes commands and keeps Arduino watchdog alive
        self.timer = self.create_timer(0.1, self._publish_cmd)
        self.get_logger().info("Bridge ready. Waiting for /cmd_vel...")

    def cmd_vel_cb(self, msg: Twist):
        # 1. Cap raw input velocities to sane hardware limits
        v = max(-0.5, min(0.5, msg.linear.x))
        w = max(-1, min(1, msg.angular.z)) 

        # 2. Kinematic math
        if abs(v) < 0.001 and abs(w) < 0.001:
            self.current_vl = 0.0
            self.current_vr = 0.0
        else:
            vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
            vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
            
            # 3. Cap final wheel speeds
            self.current_vl = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, vl))
            self.current_vr = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, vr))

    def _publish_cmd(self):
        line = f"V {self.current_vl:.4f} {self.current_vr:.4f}\n"
        try:
            self.ser.write(line.encode('ascii'))
            self.ser.flush()
        except serial.SerialException as e:
            self.get_logger().warn(f"Serial write error: {e}")

    def stop_robot(self):
        """Hard stop signal sent on node shutdown"""
        try:
            self.ser.write(b"V 0.0000 0.0000\n")
            self.ser.flush()
            self.get_logger().info("Sent hardware kill signal. Motors stopped.")
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveBridge()
    
    try:
        # Single-threaded execution prevents race conditions on velocity floats
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
