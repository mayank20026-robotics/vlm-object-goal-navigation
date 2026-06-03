#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
import time
import re

MAC_IP = "10.120.66.46"

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

SYNONYM_MAP = {
    "sofa": "couch",
    "table": "dining table",
    "desk": "dining table",
    "phone": "cell phone",
    "mobile": "cell phone",
    "tv monitor": "tv",
    "television": "tv",
    "plant": "potted plant",
    "computer": "laptop",
    "notebook": "laptop"
}

class YoloSearchCommander(Node):
    def __init__(self, user_query):
        super().__init__('yolo_search_commander')

        self.user_query = user_query.strip().lower()
        self.target_class, self.target_attributes = self.parse_query(self.user_query)

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.current_yaw = 0.0
        self.target_yaw = 0.0

        self.rgb_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/oak/stereo/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.15
        )
        self.ts.registerCallback(self.vision_callback)

        self.get_logger().info("📥 Loading YOLO model...")
        self.yolo = YOLO('yolov8n.pt')

        self.target_found = False
        self.state = "STATIONARY_SCAN"
        self.scan_start_time = time.time()
        self.scan_duration = 5.0  
        
        self.turn_count = 0
        self.max_turns_before_move = 8

        self.angular_speed = 0.5 
        self.turn_angle_deg = 45.0

        self.turning = False
        self.paused_until = 0.0
        self.goal_wait_until = 0.0

        self.cooldown_until = 0.0
        self.conf_threshold = 0.35
        self.max_global_radius = 3.0  
        self.max_goal_step = 2.0      

        # ===== Trial metrics =====
        self.trial_start_time = time.time()
        self.first_detection_time = None
        self.reach_time = None
        self.total_turns_taken = 0
        self.total_vantage_moves = 0
        self.metrics_printed = False
        # =========================

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🎯 User query: {self.user_query}")
        self.get_logger().info(f"🎯 Parsed YOLO target class: {self.target_class}")
        self.get_logger().info("=" * 60)

        if self.target_class is None:
            self.get_logger().error("❌ Could not map query to a YOLO class.")
        else:
            self.get_logger().info("✅ Search node ready.")

        self.timer = self.create_timer(0.1, self.control_loop)

    def parse_query(self, query):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
        tokens = [t for t in text.split() if t]

        stopwords = {"find", "a", "an", "the", "please", "look", "for", "me", "to", "go", "towards", "toward", "search", "locate", "detect"}
        filtered = [t for t in tokens if t not in stopwords]

        target_class = None
        for label in sorted(COCO_LABELS, key=lambda x: len(x), reverse=True):
            if label in text:
                target_class = label
                break

        if target_class is None:
            for token in filtered:
                if token in SYNONYM_MAP:
                    mapped = SYNONYM_MAP[token]
                    if mapped in COCO_LABELS:
                        target_class = mapped
                        break
                if token in COCO_LABELS:
                    target_class = token
                    break

        attributes = []
        if target_class:
            label_tokens = set(target_class.split())
            for token in filtered:
                if token not in label_tokens and SYNONYM_MAP.get(token, token) != target_class:
                    attributes.append(token)

        return target_class, attributes

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def vision_callback(self, rgb_msg, depth_msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            pass

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def start_turn_45(self):
        self.turning = True
        self.total_turns_taken += 1
        
        raw_target = self.current_yaw + math.radians(self.turn_angle_deg)
        self.target_yaw = math.atan2(math.sin(raw_target), math.cos(raw_target))
        
        self.get_logger().info(f"↪️ Turning 45 degrees (Sector turn {self.turn_count}/{self.max_turns_before_move}, Total: {self.total_turns_taken})...")

    def move_to_new_vantage_point(self):
        self.total_vantage_moves += 1
        self.get_logger().info(f"🚶 Full 360 sweep complete. Moving to a new vantage point... (Count: {self.total_vantage_moves})")
        
        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = 1.0
        goal.pose.position.y = 0.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.goal_wait_until = time.time() + 8.0
        self.turn_count = 0
        self.state = "WAIT_FOR_MOVE"

    def print_trial_metrics(self, status="SUCCESS"):
        if self.metrics_printed:
            return

        total_elapsed = time.time() - self.trial_start_time
        detection_time_str = f"{self.first_detection_time:.2f}s" if self.first_detection_time is not None else "N/A"
        reach_time_str = f"{self.reach_time:.2f}s" if self.reach_time is not None else "N/A"

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📊 TRIAL METRICS [{status}]")
        self.get_logger().info(f"🎯 Target class: {self.target_class}")
        self.get_logger().info(f"⏱️ Detection time: {detection_time_str}")
        self.get_logger().info(f"🏁 Reach time: {reach_time_str}")
        self.get_logger().info(f"↪️ Total turns taken: {self.total_turns_taken}")
        self.get_logger().info(f"🚶 Vantage point moves: {self.total_vantage_moves}")
        self.get_logger().info(f"🕒 Total trial elapsed: {total_elapsed:.2f}s")
        self.get_logger().info("=" * 60)

        self.metrics_printed = True

    def navigate_to_target(self, box):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        rgb_h, rgb_w = self.latest_rgb.shape[:2]
        depth_h, depth_w = self.latest_depth.shape[:2]

        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h

        depth_cx = int(center_x * scale_x)
        depth_cy = int(center_y * scale_y)

        depth_cx = max(0, min(depth_cx, depth_w - 1))
        depth_cy = max(0, min(depth_cy, depth_h - 1))

        depth_mm = int(self.latest_depth[depth_cy, depth_cx])
        if depth_mm <= 0:
            return

        z_forward = depth_mm / 1000.0
        fx = depth_w / (2.0 * math.tan(math.radians(73.0 / 2.0)))
        cx = depth_w / 2.0
        x_lateral = (depth_cx - cx) * z_forward / fx

        raw_x = z_forward
        raw_y = -x_lateral

        distance = math.sqrt(raw_x**2 + raw_y**2)
        standoff = 0.5

        # Log first detection time
        if self.first_detection_time is None:
            self.first_detection_time = time.time() - self.trial_start_time
            self.get_logger().info(f"⏱️ First target detection at {self.first_detection_time:.2f}s")

        # Log reach time and print metrics when done
        if distance <= standoff + 0.1:
            self.reach_time = time.time() - self.trial_start_time
            self.get_logger().info("🎉 Target reached.")
            self.stop_robot()
            self.target_found = True
            self.print_trial_metrics(status="SUCCESS")
            return

        drive_distance = max(distance - standoff, 0.0)
        drive_distance = min(drive_distance, self.max_goal_step)

        scale = drive_distance / distance if distance > 1e-3 else 0.0
        target_x = raw_x * scale
        target_y = raw_y * scale

        yaw = math.atan2(target_y, target_x)
        
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw/2)
        qw = math.cos(yaw/2)

        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(target_x)
        goal.pose.position.y = float(target_y)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)
        self.get_logger().info(f"🚀 Nav2 goal sent step={drive_distance:.2f}m")
        self.target_found = True

    def detect_target_with_yolo(self):
        if self.latest_rgb is None or self.target_class is None:
            return None, None

        image = self.latest_rgb.copy()
        results = self.yolo(image, verbose=False, conf=self.conf_threshold)

        best_box = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo.names[cls_id].lower()

                if label == self.target_class and conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                        best_conf = conf
                        best_box = (x1, y1, x2, y2)

        return best_box, best_conf

    def control_loop(self):
        if self.target_found or self.target_class is None:
            self.stop_robot()
            return

        now = time.time()

        if now < self.paused_until or now < self.cooldown_until:
            return

        if self.state == "WAIT_FOR_MOVE":
            if now >= self.goal_wait_until:
                self.get_logger().info("📍 Reached new vantage region. Resuming scans.")
                self.state = "STATIONARY_SCAN"
                self.scan_start_time = time.time()
            return

        if self.turning:
            error = self.target_yaw - self.current_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            if abs(error) > 0.05: 
                speed = max(0.15, min(self.angular_speed, abs(error) * 1.5))
                msg = Twist()
                msg.angular.z = speed if error > 0 else -speed
                self.cmd_pub.publish(msg)
                return
            else:
                self.turning = False
                self.stop_robot()
                self.paused_until = now + 0.5
                self.get_logger().info("✅ Turn complete. Scanning this sector...")
                self.state = "STATIONARY_SCAN"
                self.scan_start_time = time.time()
                return

        best_box, best_conf = self.detect_target_with_yolo()

        if best_box is not None:
            self.stop_robot()
            self.get_logger().info(f"✅ YOLO found [{self.target_class}]! Navigating...")
            self.navigate_to_target(best_box)
            return

        if self.state == "STATIONARY_SCAN":
            self.stop_robot() 
            elapsed = now - self.scan_start_time
            
            if elapsed >= self.scan_duration:
                self.get_logger().info("⏳ Sector clear. Initiating turn.")
                self.turn_count += 1

                if self.turn_count >= self.max_turns_before_move:
                    self.move_to_new_vantage_point()
                else:
                    self.start_turn_45()

    def destroy_node(self):
        # Print metrics if the node is killed before reaching the goal
        if not self.metrics_printed and self.target_class is not None:
            self.print_trial_metrics(status="STOPPED")
        super().destroy_node()

def main(args=None):
    print("\n" + "=" * 60)
    print("YOLO SEARCH COMMANDER (ODOMETRY CLOSED-LOOP)")
    print("=" * 60)
    user_query = input("What object should the robot search for?: ")

    rclpy.init(args=args)
    node = YoloSearchCommander(user_query=user_query)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
