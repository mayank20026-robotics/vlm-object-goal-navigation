#!/usr/bin/env python3
"""
diff_drive_bridge.py  —  ROS 2 Humble
Reads encoder ticks from Arduino Mega over USB serial,
publishes /odom and the odom→base_link tf transform.
Subscribes to /cmd_vel and sends wheel velocities to Arduino.

Serial protocol:
  Arduino → Jetson:  "O <left_ticks> <right_ticks> <millis>\\n"
  Jetson  → Arduino: "V <left_rad_s> <right_rad_s>\\n"

Usage:
  python3 diff_drive_bridge.py
"""

import math
import threading
import serial
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist
from tf2_ros import TransformBroadcaster

# ── Robot parameters ─────────────────────────────────────────
WHEEL_RADIUS  = 0.055
WHEEL_BASE    = 0.38
ENCODER_CPR   = 58000.0
DIST_PER_TICK = (2.0 * math.pi * WHEEL_RADIUS) / ENCODER_CPR
MAX_WHEEL_VEL = 6.5  # rad/s — above this motors can't stop cleanly

# ── Serial ───────────────────────────────────────────────────
SERIAL_PORT = "/dev/arduino"
BAUD_RATE   = 115200


class DiffDriveBridge(Node):
    def __init__(self):
        super().__init__("diff_drive_bridge")

        # ── Publishers / subscribers / tf ─────────────────────
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_bcast = TransformBroadcaster(self)
        self.cmd_sub  = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_cb, 10)

        self.current_vl = 0.0
        self.current_vr = 0.0
        self._serial_lock = threading.Lock()

        # ── Odometry state ────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.prev_enc_l = None
        self.prev_enc_r = None

        # ── Serial ────────────────────────────────────────────
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
            self.get_logger().info(f"Opened serial port {SERIAL_PORT}")
        except serial.SerialException as e:
            self.get_logger().fatal(f"Cannot open {SERIAL_PORT}: {e}")
            raise

        self.create_timer(0.1, self._publish_cmd)

        self._lock = threading.Lock()
        self._read_thread = threading.Thread(
            target=self._serial_read_loop, daemon=True)
        self._read_thread.start()

    # ── cmd_vel callback ──────────────────────────────────────
    def cmd_vel_cb(self, msg: Twist):
        v = max(-0.38, min(0.38,msg.linear.x))
        w = max(-0.5, min(0.5,msg.angular.z))

        if abs(v) < 0.001 and abs(w) < 0.001:
            self.current_vl = 0.0
            self.current_vr = 0.0
        else:
            vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
            vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
            # Clamp to safe range
            self.current_vl = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, vl))
            self.current_vr = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, vr))

        # self.get_logger().info(
        #     f"cmd_vel: v={v:.2f} w={w:.2f} → vl={self.current_vl:.2f} vr={self.current_vr:.2f}")

    # ── 10Hz timer ───────────────────────────────────────────
    def _publish_cmd(self):
        line = f"V {self.current_vl:.4f} {self.current_vr:.4f}\n"
        # print(f"Writing to serial: {line.strip()}", flush=True)
        with self._serial_lock:
            try:
                self.ser.write(line.encode())
                self.ser.flush()
            except serial.SerialException as e:
                self.get_logger().warn(f"Serial write error: {e}")

    # ── Background serial read loop ───────────────────────────
    def _serial_read_loop(self):
        while rclpy.ok():
            with self._serial_lock:
                try:
                    raw = self.ser.readline()
                except serial.SerialException as e:
                    self.get_logger().warn(f"Serial read error: {e}")
                    break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("O"):
                self._process_odom(line)

    # ── Parse odom ────────────────────────────────────────────
    def _process_odom(self, line: str):
        parts = line.split()
        if len(parts) != 4:
            return
        try:
            enc_l  = int(parts[1])
            enc_r  = int(parts[2])
            # now_ms = int(parts[3])
        except ValueError:
            return
        
        # self.get_logger().info(f"ticks L={enc_l} R={enc_r} | dl={enc_l - (self.prev_enc_l or enc_l)} dr={enc_r - (self.prev_enc_r or enc_r)}")

        if self.prev_enc_l is None:
            self.prev_enc_l = enc_l
            self.prev_enc_r = enc_r
            return

        dl = enc_l - self.prev_enc_l
        dr = enc_r - self.prev_enc_r
        self.prev_enc_l = enc_l
        self.prev_enc_r = enc_r

        dist_l = dl * DIST_PER_TICK
        dist_r = dr * DIST_PER_TICK
        dist_c = (dist_l + dist_r) / 2.0
        d_yaw  = (dist_r - dist_l) / WHEEL_BASE

        with self._lock:
            self.yaw += d_yaw
            self.x   += dist_c * math.cos(self.yaw)
            self.y   += dist_c * math.sin(self.yaw)
            x, y, yaw = self.x, self.y, self.yaw

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        odom.pose.covariance[0]   = 0.001
        odom.pose.covariance[7]   = 0.001
        odom.pose.covariance[35]  = 0.005
        odom.twist.covariance[0]  = 0.001
        odom.twist.covariance[35] = 0.005
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id  = "base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(yaw / 2.0)
        tf.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_bcast.sendTransform(tf)


def main():
    rclpy.init()
    node = DiffDriveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
