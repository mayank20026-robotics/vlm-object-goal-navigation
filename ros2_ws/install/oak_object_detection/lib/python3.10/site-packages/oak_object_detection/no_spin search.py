#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
import time


class SimpleYoloApproach(Node):
    def __init__(self, target_object):
        super().__init__('simple_yolo_approach')

        self.target_object = target_object.lower().strip()
        self.bridge = CvBridge()

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.rgb_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/oak/stereo/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.vision_callback)

        self.get_logger().info("📥 Loading YOLO model...")
        self.yolo = YOLO('yolov8n.pt')

        self.conf_threshold = 0.35
        self.standoff_m = 0.5
        self.max_goal_step_m = 1.5
        self.goal_cooldown = 3.0
        self.last_goal_time = 0.0
        self.target_found = False

        self.get_logger().info(f"🎯 Target object: {self.target_object}")
        self.get_logger().info(f"🎚️ Confidence threshold: {self.conf_threshold}")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def vision_callback(self, rgb_msg, depth_msg):
        if self.target_found:
            return

        if time.time() - self.last_goal_time < self.goal_cooldown:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        results = self.yolo(cv_image, verbose=False, conf=self.conf_threshold)

        best_box = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                object_name = self.yolo.names[cls_id].lower()

                if object_name != self.target_object:
                    continue

                if conf < self.conf_threshold:
                    continue

                if (x2 - x1) < 40 or (y2 - y1) < 40:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

        if best_box is None:
            return

        self.get_logger().info(
            f"✅ Detected [{self.target_object}] with confidence {best_conf:.2f}"
        )

        x1, y1, x2, y2 = best_box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        rgb_h, rgb_w = cv_image.shape[:2]
        depth_h, depth_w = cv_depth.shape[:2]

        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h

        depth_cx = int(center_x * scale_x)
        depth_cy = int(center_y * scale_y)

        depth_cx = max(0, min(depth_cx, depth_w - 1))
        depth_cy = max(0, min(depth_cy, depth_h - 1))

        depth_mm = int(cv_depth[depth_cy, depth_cx])

        if depth_mm <= 0:
            self.get_logger().warn("⚠️ Invalid depth at target center.")
            return

        z_forward = depth_mm / 1000.0
        hfov_deg = 73.0
        fx = depth_w / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        cx = depth_w / 2.0
        x_lateral = (depth_cx - cx) * z_forward / fx

        raw_x = z_forward
        raw_y = -x_lateral

        distance = math.sqrt(raw_x**2 + raw_y**2)

        if distance <= self.standoff_m + 0.1:
            self.get_logger().info("🎉 Target reached.")
            self.stop_robot()
            self.target_found = True
            return

        drive_distance = max(distance - self.standoff_m, 0.0)
        drive_distance = min(drive_distance, self.max_goal_step_m)

        if drive_distance <= 0.1:
            self.get_logger().info("Already close enough, not sending new goal.")
            return

        scale = drive_distance / distance if distance > 1e-3 else 0.0
        target_x = raw_x * scale
        target_y = raw_y * scale

        yaw = math.atan2(target_y, target_x)
        q = self.get_quaternion_from_euler(0.0, 0.0, yaw)

        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(target_x)
        goal.pose.position.y = float(target_y)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        self.goal_pub.publish(goal)
        self.last_goal_time = time.time()

        self.get_logger().info(
            f"🚀 Sent goal toward [{self.target_object}] at "
            f"({target_x:.2f}, {target_y:.2f}), raw distance={distance:.2f}m"
        )


def main(args=None):
    print("\n" + "=" * 50)
    print("SIMPLE YOLO APPROACH")
    print("=" * 50)
    target_object = input("Enter target object (e.g. chair, bottle, laptop): ").strip().lower()

    rclpy.init(args=args)
    node = SimpleYoloApproach(target_object)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()