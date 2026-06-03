#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import depthai as dai
import time
import math
import re

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

class OakYoloSearchCommander(Node):
    def __init__(self, user_query):
        super().__init__('oak_yolo_search_commander')

        self.user_query = user_query.strip().lower()
        self.target_class, self.target_attributes = self.parse_query(self.user_query)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.current_yaw = 0.0
        self.target_yaw = 0.0

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
        self.max_goal_step = 2.0
        self.standoff_m = 0.5

        # ===== Trial metrics =====
        self.trial_start_time = time.time()
        self.first_detection_time = None
        self.reach_time = None
        self.total_turns_taken = 0
        self.total_vantage_moves = 0
        self.goal_sent_time = None
        self.last_detection_conf = None
        self.last_detection_distance = None
        self.metrics_printed = False
        # =========================

        self.get_logger().info("=" * 60)
        self.get_logger().info("📷 Booting OAK-D pipeline with ONBOARD Hardware YOLO...")
        self.get_logger().info(f"🎯 User query: {self.user_query}")
        self.get_logger().info(f"🎯 Parsed target class: {self.target_class}")
        self.get_logger().info("=" * 60)

        if self.target_class is None:
            self.get_logger().error("❌ Could not map query to a supported YOLO class.")
            return

        self.setup_depthai()
        self.timer = self.create_timer(0.1, self.control_loop)

    def parse_query(self, query):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
        tokens = [t for t in text.split() if t]

        stopwords = {
            "find", "a", "an", "the", "please", "look", "for", "me", "to",
            "go", "towards", "toward", "search", "locate", "detect"
        }
        filtered = [t for t in tokens if t not in stopwords]

        target_class = None
        for label in sorted(COCO_LABELS, key=lambda x: len(x), reverse=True):
            if label in text:
                target_class = label
                break

        if target_class is None:
            for i, token in enumerate(filtered):
                if i < len(filtered) - 1:
                    two_word = filtered[i] + " " + filtered[i + 1]
                    if two_word in SYNONYM_MAP:
                        mapped = SYNONYM_MAP[two_word]
                        if mapped in COCO_LABELS:
                            target_class = mapped
                            break

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
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def setup_depthai(self):
        self.pipeline = dai.Pipeline()

        # 1. Hardware Nodes
        rgbcamera = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=20
        )
        monoleft = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=20
        )
        monoright = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=20
        )

        stereo_depth = self.pipeline.create(dai.node.StereoDepth)
        stereo_depth.setExtendedDisparity(True)
        monoleft.requestOutput((640, 400)).link(stereo_depth.left)
        monoright.requestOutput((640, 400)).link(stereo_depth.right)

        # 2. VPU Neural Network & Spatial Calculator
        modelDescription = dai.NNModelDescription("yolov6-nano")
        spatial_detection = self.pipeline.create(dai.node.SpatialDetectionNetwork).build(
            rgbcamera, stereo_depth, modelDescription
        )

        spatial_detection.spatialLocationCalculator.initialConfig.setSegmentationPassthrough(False)
        spatial_detection.setConfidenceThreshold(self.conf_threshold)
        spatial_detection.input.setBlocking(False)
        spatial_detection.setDepthLowerThreshold(100)
        spatial_detection.setDepthUpperThreshold(5000)

        # 3. ONLY pull the NN meta-data queue. Zero images transferred over USB!
        self.q_nn = spatial_detection.out.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()
        self.get_logger().info("✅ OAK-D Spatial Hardware Pipeline Running. USB Video Feed is OFF.")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def start_turn_45(self):
        self.turning = True
        self.total_turns_taken += 1
        raw_target = self.current_yaw + math.radians(self.turn_angle_deg)
        self.target_yaw = math.atan2(math.sin(raw_target), math.cos(raw_target))
        self.get_logger().info(
            f"↪️ Turning 45 degrees (sector turn {self.turn_count}/{self.max_turns_before_move}, total turns={self.total_turns_taken})..."
        )

    def move_to_new_vantage_point(self):
        self.total_vantage_moves += 1
        self.get_logger().info(
            f"🚶 Full 360 sweep complete. Moving to a new vantage point... (count={self.total_vantage_moves})"
        )

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

    def get_best_detection(self):
        # Only read the lightweight metadata queue
        in_nn = self.q_nn.tryGet()

        if in_nn is None:
            return None, None

        detections = in_nn.detections
        best_detection = None
        best_conf = 0.0

        for detection in detections:
            try:
                object_name = COCO_LABELS[detection.label]
            except Exception:
                object_name = "unknown"

            if object_name != self.target_class:
                continue

            # Filtering out noise using normalized percentages instead of pixels
            # 0.06 is ~40px wide on a 640px image. 0.10 is ~40px high on a 400px image.
            bbox_width = detection.xmax - detection.xmin
            bbox_height = detection.ymax - detection.ymin

            if bbox_width < 0.06 or bbox_height < 0.10:
                continue

            conf = float(detection.confidence)
            if conf > best_conf:
                best_conf = conf
                best_detection = detection

        return best_detection, best_conf

    def print_trial_metrics(self, status="SUCCESS"):
        if self.metrics_printed:
            return

        total_elapsed = time.time() - self.trial_start_time
        detection_time_str = f"{self.first_detection_time:.2f}s" if self.first_detection_time is not None else "N/A"
        reach_time_str = f"{self.reach_time:.2f}s" if self.reach_time is not None else "N/A"
        goal_sent_delay_str = f"{self.goal_sent_time:.2f}s" if self.goal_sent_time is not None else "N/A"
        last_conf_str = f"{self.last_detection_conf:.2f}" if self.last_detection_conf is not None else "N/A"
        last_dist_str = f"{self.last_detection_distance:.2f}m" if self.last_detection_distance is not None else "N/A"

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📊 TRIAL METRICS [{status}]")
        self.get_logger().info(f"🎯 Target class: {self.target_class}")
        self.get_logger().info(f"⏱️ Detection time: {detection_time_str}")
        self.get_logger().info(f"🏁 Reach time: {reach_time_str}")
        self.get_logger().info(f"📨 Goal sent time: {goal_sent_delay_str}")
        self.get_logger().info(f"↪️ Total turns taken: {self.total_turns_taken}")
        self.get_logger().info(f"🚶 Vantage point moves: {self.total_vantage_moves}")
        self.get_logger().info(f"📦 Detection confidence: {last_conf_str}")
        self.get_logger().info(f"📏 Detection distance: {last_dist_str}")
        self.get_logger().info(f"🕒 Total trial elapsed: {total_elapsed:.2f}s")
        self.get_logger().info("=" * 60)

        self.metrics_printed = True

    def navigate_to_detection(self, detection, best_conf=None):
        # We no longer need to calculate this from the depth map!
        # The OAK-D Spatial Calculator provides the absolute XYZ directly.
        raw_x = detection.spatialCoordinates.z / 1000.0  # OAK-D Z is ROS Forward X
        raw_y = -(detection.spatialCoordinates.x / 1000.0) # OAK-D X is ROS Right (-Y)

        distance = math.sqrt(raw_x**2 + raw_y**2)
        self.last_detection_distance = distance
        self.last_detection_conf = best_conf

        if self.first_detection_time is None:
            self.first_detection_time = time.time() - self.trial_start_time
            self.get_logger().info(
                f"⏱️ First target detection at {self.first_detection_time:.2f}s"
            )

        if distance <= self.standoff_m + 0.1:
            self.reach_time = time.time() - self.trial_start_time
            self.get_logger().info("🎉 Target reached.")
            self.stop_robot()
            self.target_found = True
            self.print_trial_metrics(status="SUCCESS")
            return

        drive_distance = max(distance - self.standoff_m, 0.0)
        drive_distance = min(drive_distance, self.max_goal_step)

        scale = drive_distance / distance if distance > 1e-3 else 0.0
        target_x = raw_x * scale
        target_y = raw_y * scale

        yaw = math.atan2(target_y, target_x)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(target_x)
        goal.pose.position.y = float(target_y)
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)

        if self.goal_sent_time is None:
            self.goal_sent_time = time.time() - self.trial_start_time

        self.reach_time = time.time() - self.trial_start_time

        self.get_logger().info(f"🚀 Nav2 goal sent for [{self.target_class}] step={drive_distance:.2f}m")
        self.target_found = True
        self.print_trial_metrics(status="SUCCESS")

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

        best_detection, best_conf = self.get_best_detection()

        if best_detection is not None:
            self.stop_robot()
            self.get_logger().info(
                f"✅ YOLO found [{self.target_class}] with confidence {best_conf:.2f}. Navigating..."
            )
            self.navigate_to_detection(best_detection, best_conf=best_conf)
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
        if not self.metrics_printed and self.target_class is not None:
            self.print_trial_metrics(status="STOPPED")

        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    print("\n" + "=" * 60)
    print("OAK HARDWARE VPU SEARCH COMMANDER (HEADLESS)")
    print("=" * 60)
    user_query = input("What object should the robot search for?: ")

    rclpy.init(args=args)
    node = OakYoloSearchCommander(user_query=user_query)

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