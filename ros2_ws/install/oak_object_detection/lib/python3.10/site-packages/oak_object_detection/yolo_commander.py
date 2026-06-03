#!/usr/bin/env python3
import math
import time
from collections import deque

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class PureYoloCommander(Node):
    def __init__(self, target_object: str):
        super().__init__('yolo_commander_node')

        self.target_object = target_object.strip().lower()
        self.target_found = False
        self.target_locked = False

        self.robot_state = 'STEP_SCAN_ROTATE'
        self.last_seen_time = 0.0
        self.last_goal_time = 0.0
        self.last_goal_xy = None
        self.last_range_m = None

        self.current_yaw = None
        self.scan_start_yaw = None
        self.scan_target_delta = math.radians(45.0)

        self.spin_speed = 0.10
        self.step_pause_sec = 1.2
        self.max_scan_steps = 8
        self.current_scan_step = 0
        self.state_deadline = time.time()

        self.patrol_distance = 1.2
        self.patrol_wait_time = 6.0

        self.near_hold_distance_m = 0.9

        self.goal_resend_period = 1.0
        self.goal_position_epsilon = 0.15

        self.conf_threshold = 0.50
        self.min_box_area = 0.01
        self.required_detection_streak = 3
        self.max_history = 5
        self.box_center_jump_limit = 0.12
        self.box_area_ratio_limit = 0.45

        self.depth_patch_radius = 2
        self.depth_min_mm = 250
        self.depth_max_mm = 5000

        self.standoff_m = 0.6
        self.stop_tolerance_m = 0.1
        self.max_mini_goal_m = 1.0

        self.recent_centers = deque(maxlen=self.max_history)
        self.recent_areas = deque(maxlen=self.max_history)
        self.detection_streak = 0

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.bridge = CvBridge()
        self.get_logger().info('Loading YOLO model...')
        self.yolo = YOLO('yolov8n.pt')
        self.get_logger().info(f'Target object: {self.target_object}')

        self.rgb_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/oak/stereo/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.vision_callback)

        self.timer = self.create_timer(0.1, self.state_machine_logic)

    # ------------------------------------------------------------ math helpers

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def angle_diff(self, target, current):
        return self.normalize_angle(target - current)

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = math.sin(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) - math.cos(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)
        qy = math.cos(roll/2)*math.sin(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.cos(pitch/2)*math.sin(yaw/2)
        qz = math.cos(roll/2)*math.cos(pitch/2)*math.sin(yaw/2) - math.sin(roll/2)*math.sin(pitch/2)*math.cos(yaw/2)
        qw = math.cos(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)
        return qx, qy, qz, qw

    # ------------------------------------------------------------ callbacks

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------ robot helpers

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def publish_spin(self, angular_z):
        msg = Twist()
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def publish_local_goal(self, x: float, y: float):
        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)

        yaw = math.atan2(y, x)
        qx, qy, qz, qw = self.get_quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)

    def reset_detection_history(self):
        self.detection_streak = 0
        self.recent_centers.clear()
        self.recent_areas.clear()

    # ------------------------------------------------------------ state changes

    def enter_step_scan_rotate(self):
        if self.current_yaw is None:
            return
        self.robot_state = 'STEP_SCAN_ROTATE'
        self.scan_start_yaw = self.current_yaw
        self.get_logger().info(
            f'Scan step {self.current_scan_step + 1}/{self.max_scan_steps}: rotating 45 deg using odom yaw'
        )

    def enter_step_scan_pause(self):
        self.robot_state = 'STEP_SCAN_PAUSE'
        self.state_deadline = time.time() + self.step_pause_sec
        self.stop_robot()
        self.get_logger().info('Pausing to observe...')

    def enter_patrol(self):
        self.robot_state = 'PATROL'
        self.stop_robot()
        self.publish_local_goal(self.patrol_distance, 0.0)
        self.state_deadline = time.time() + self.patrol_wait_time
        self.get_logger().info('360 scan complete. Moving to new vantage point...')

    def enter_hold_position(self):
        self.robot_state = 'HOLD_POSITION'
        self.stop_robot()
        self.get_logger().info(
            'Object was close but lost sight. Holding position and waiting to reacquire...'
        )

    # ------------------------------------------------------------ perception

    def choose_target_box(self, results, image_shape):
        h, w = image_shape[:2]
        candidates = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.yolo.names[cls_id].lower()

                if label != self.target_object or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                area_norm = ((x2 - x1) * (y2 - y1)) / float(w * h)
                if area_norm < self.min_box_area:
                    continue

                candidates.append((conf * area_norm, conf, x1, y1, x2, y2, area_norm))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, conf, x1, y1, x2, y2, area_norm = candidates[0]

        return {
            'conf': conf,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'area_norm': area_norm,
        }

    def is_box_stable(self, center_xy, area_norm):
        self.recent_centers.append(center_xy)
        self.recent_areas.append(area_norm)
        self.detection_streak += 1

        if self.detection_streak < self.required_detection_streak:
            return False

        centers = np.array(self.recent_centers, dtype=np.float32)
        areas = np.array(self.recent_areas, dtype=np.float32)

        center_mean = centers.mean(axis=0)
        center_dev = np.linalg.norm(centers - center_mean, axis=1)
        area_dev = np.abs(areas - areas.mean())

        if np.max(center_dev) > self.box_center_jump_limit:
            return False
        if np.max(area_dev) > self.box_area_ratio_limit:
            return False

        return True

    def compute_depth_from_patch(self, cv_depth, depth_cx, depth_cy):
        r = self.depth_patch_radius
        y1 = max(0, depth_cy - r)
        y2 = min(cv_depth.shape[0], depth_cy + r + 1)
        x1 = max(0, depth_cx - r)
        x2 = min(cv_depth.shape[1], depth_cx + r + 1)

        patch = cv_depth[y1:y2, x1:x2]
        valid = patch[(patch >= self.depth_min_mm) & (patch <= self.depth_max_mm)]

        if valid.size < 5:
            return None

        return float(np.median(valid))

    def should_send_goal(self, target_x, target_y):
        now = time.time()

        if (now - self.last_goal_time) < self.goal_resend_period:
            return False

        if self.last_goal_xy is None:
            return True

        dx = target_x - self.last_goal_xy[0]
        dy = target_y - self.last_goal_xy[1]

        return math.hypot(dx, dy) >= self.goal_position_epsilon

    # ------------------------------------------------------------ FSM

    def state_machine_logic(self):
        now = time.time()

        if self.target_found:
            return

        if self.current_yaw is None:
            return

        if self.robot_state == 'STEP_SCAN_ROTATE':
            if self.scan_start_yaw is None:
                self.scan_start_yaw = self.current_yaw

            rotated = abs(self.angle_diff(self.current_yaw, self.scan_start_yaw))

            target_with_margin = self.scan_target_delta - math.radians(2.5)

            if rotated < target_with_margin:
                remaining = self.scan_target_delta - rotated
                cmd = max(0.08, min(self.spin_speed, 1.5 * remaining))
                self.publish_spin(cmd)
            else:
                self.enter_step_scan_pause()
            return

        if self.robot_state == 'STEP_SCAN_PAUSE':
            if now >= self.state_deadline:
                self.current_scan_step += 1
                if self.current_scan_step >= self.max_scan_steps:
                    self.current_scan_step = 0
                    self.enter_patrol()
                else:
                    self.enter_step_scan_rotate()
            return

        if self.robot_state == 'PATROL':
            if now >= self.state_deadline:
                self.current_scan_step = 0
                self.enter_step_scan_rotate()
            return

        if self.robot_state == 'PURSUIT':
            time_since_seen = now - self.last_seen_time
            if time_since_seen > 4.0:
                if self.last_range_m is not None and self.last_range_m <= self.near_hold_distance_m:
                    self.enter_hold_position()
                else:
                    self.get_logger().warn('Target lost while far. Returning to scan.')
                    self.target_locked = False
                    self.current_scan_step = 0
                    self.reset_detection_history()
                    self.enter_step_scan_rotate()
            return

        if self.robot_state == 'HOLD_POSITION':
            return

    # ------------------------------------------------------------ vision

    def vision_callback(self, rgb_msg, depth_msg):
        if self.target_found:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        results = self.yolo(cv_image, verbose=False)
        best_box = self.choose_target_box(results, cv_image.shape)

        if best_box is None:
            if self.robot_state in ['PURSUIT', 'HOLD_POSITION']:
                return
            self.reset_detection_history()
            return

        rgb_h, rgb_w = cv_image.shape[:2]
        depth_h, depth_w = cv_depth.shape[:2]

        center_x = int((best_box['x1'] + best_box['x2']) / 2)
        sample_y = int(best_box['y1'] + 0.75 * (best_box['y2'] - best_box['y1']))
        center_norm = (center_x / float(rgb_w), sample_y / float(rgb_h))

        if not self.is_box_stable(center_norm, best_box['area_norm']):
            return

        if not self.target_locked:
            self.target_locked = True
            self.get_logger().info(
                f'Target LOCKED: {self.target_object.upper()} — will not switch targets.'
            )

        self.last_seen_time = time.time()

        if self.robot_state != 'PURSUIT':
            self.get_logger().info('Target reacquired. Entering PURSUIT.')
            self.robot_state = 'PURSUIT'
            self.stop_robot()

        scale_x = depth_w / float(rgb_w)
        scale_y = depth_h / float(rgb_h)
        depth_cx = int(center_x * scale_x)
        depth_cy = int(sample_y * scale_y)

        depth_cx = max(0, min(depth_cx, depth_w - 1))
        depth_cy = max(0, min(depth_cy, depth_h - 1))

        depth_mm = self.compute_depth_from_patch(cv_depth, depth_cx, depth_cy)
        if depth_mm is None:
            self.get_logger().warn('Depth patch invalid. Skipping frame.')
            return

        z_forward = depth_mm / 1000.0
        fx = depth_w / (2.0 * math.tan(math.radians(73.0 / 2.0)))
        cx_img = depth_w / 2.0
        x_lateral = (depth_cx - cx_img) * z_forward / fx

        raw_x = z_forward
        raw_y = -x_lateral
        distance = math.hypot(raw_x, raw_y)
        self.last_range_m = distance

        if distance <= (self.standoff_m + self.stop_tolerance_m):
            self.get_logger().info('Standoff distance reached. Target acquired.')
            self.stop_robot()
            self.target_found = True
            return

        drive_distance = min(distance - self.standoff_m, self.max_mini_goal_m)
        if drive_distance <= 0.0:
            self.stop_robot()
            return

        scale = drive_distance / max(distance, 1e-6)
        target_x = raw_x * scale
        target_y = raw_y * scale

        if not self.should_send_goal(target_x, target_y):
            return

        self.publish_local_goal(target_x, target_y)
        self.last_goal_time = time.time()
        self.last_goal_xy = (target_x, target_y)
        self.get_logger().info(
            f"Goal | conf={best_box['conf']:.2f} dist={distance:.2f}m goal=({target_x:.2f},{target_y:.2f})"
        )


def main(args=None):
    print('\n' + '=' * 50)
    print('PURE YOLO NAV2 COMMANDER')
    print('=' * 50)
    print('Examples: person, cup, bottle, chair, laptop, cell phone')

    user_target = input('What object should the robot hunt for?: ').strip()
    print(f"\n[SYSTEM] Target: '{user_target}'. Initializing ROS 2...\n")

    rclpy.init(args=args)
    node = PureYoloCommander(target_object=user_target)

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