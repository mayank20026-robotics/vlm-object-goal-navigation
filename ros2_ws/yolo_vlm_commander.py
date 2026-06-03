#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import depthai as dai
import cv2
import base64
import json
import time
import math
from openai import OpenAI


MAC_IP = "172.20.10.7"

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


class YoloVlmCommander(Node):
    def __init__(self, target_object):
        super().__init__('yolo_vlm_commander')

        self.target_object = target_object
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.target_found = False
        self.vlm_cooldown = 0.0
        self.last_vlm_result = None

        self.vlm_client = OpenAI(
            base_url=f"http://{MAC_IP}:11434/v1",
            api_key="sk-no-key-required",
            timeout=20.0
        )

        self.get_logger().info("📷 Booting Hardware-Accelerated OAK-D Pipeline...")
        self.get_logger().info(f"🎯 Primary Target Description: [{self.target_object}]")
        self.setup_depthai()
        self.timer = self.create_timer(0.1, self.vision_loop)
        self.trial_start_time = time.monotonic()
        self.detection_time_sec = None

    def setup_depthai(self):
        self.pipeline = dai.Pipeline()

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

        modelDescription = dai.NNModelDescription(model="yolov6-nano", platform="RVC2")
        spatial_detection = self.pipeline.create(dai.node.SpatialDetectionNetwork).build(
            rgbcamera, stereo_depth, modelDescription
        )

        spatial_detection.setConfidenceThreshold(0.5)
        spatial_detection.input.setBlocking(False)
        spatial_detection.setDepthLowerThreshold(100)
        spatial_detection.setDepthUpperThreshold(5000)

        self.q_rgb = rgbcamera.requestOutput((640, 400)).createOutputQueue(maxSize=4, blocking=False)
        self.q_nn = spatial_detection.out.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()
        self.get_logger().info("✅ OAK-D Pipeline Running! Commencing Search...")

    def extract_json(self, text):
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        return json.loads(text)

    def query_vlm_match(self, roi_image):
        _, buffer = cv2.imencode('.jpg', roi_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
You are helping a mobile robot find an object from a natural language description.

Target description: "{self.target_object}"

Look only at the provided image crop and decide whether it matches the description.

Respond ONLY in valid JSON using exactly this schema:
{{
  "match": true,
  "score": 0.0,
  "reason": "short phrase",
  "object_type": "short noun phrase"
}}

Rules:
- "match" must be true only if the crop likely contains the described target.
- "score" must be between 0.0 and 1.0.
- Use a higher score only when the match is visually convincing.
- Consider object category, color, shape, and visible context.
- If unsure, set "match": false and give a low score.
"""

        response = self.vlm_client.chat.completions.create(
            model="qwen2.5vl:3b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=80,
            temperature=0.0
        )

        return self.extract_json(response.choices[0].message.content)

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def spin_robot_to_search(self):
        msg = Twist()
        msg.angular.z = 0.5
        self.cmd_pub.publish(msg)

    def navigate_to_detection(self, detection):
        if self.detection_time_sec is None:
            self.detection_time_sec = time.monotonic() - self.trial_start_time
            self.get_logger().info(f'⏱️ DETECTION TIME: {self.detection_time_sec:.2f} s')
            
        raw_x = detection.spatialCoordinates.z / 1000.0
        raw_y = -(detection.spatialCoordinates.x / 1000.0)

        distance = math.sqrt(raw_x**2 + raw_y**2)
        standoff_m = 0.5

        if distance > standoff_m and distance > 1e-3:
            scale = (distance - standoff_m) / distance
            target_x_forward = raw_x * scale
            target_y_lateral = raw_y * scale
        else:
            target_x_forward = 0.0
            target_y_lateral = 0.0

        self.get_logger().info(f"📍 RAW TARGET: [{raw_x:.2f}m, {raw_y:.2f}m]")
        self.get_logger().info(f"🛑 NAV2 GOAL (0.5m Standoff): [{target_x_forward:.2f}m, {target_y_lateral:.2f}m]")

        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(target_x_forward)
        goal.pose.position.y = float(target_y_lateral)

        yaw = math.atan2(target_y_lateral, target_x_forward)
        q = self.get_quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        self.goal_pub.publish(goal)
        self.get_logger().info("🚀 SAFE GOAL SENT TO NAV2! Halting vision node.")

        self.pipeline.stop()
        rclpy.shutdown()

    def vision_loop(self):
        if self.target_found:
            return

        in_rgb = self.q_rgb.tryGet()
        in_nn = self.q_nn.tryGet()

        if in_rgb is None or in_nn is None:
            if time.time() > self.vlm_cooldown:
                self.spin_robot_to_search()
            return

        cv_image = in_rgb.getCvFrame()
        detections = in_nn.detections

        interesting_object_in_view = False
        best_detection = None
        best_score = 0.0
        best_reason = ""
        best_name = "unknown"

        for detection in detections:
            try:
                object_name = COCO_LABELS[detection.label]
            except Exception:
                object_name = "unknown"

            xmin = int(detection.xmin * cv_image.shape[1])
            ymin = int(detection.ymin * cv_image.shape[0])
            xmax = int(detection.xmax * cv_image.shape[1])
            ymax = int(detection.ymax * cv_image.shape[0])

            if (xmax - xmin) <= 50 or (ymax - ymin) <= 50:
                continue

            interesting_object_in_view = True
            # cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # cv2.putText(cv_image, object_name, (xmin, max(20, ymin - 5)),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if time.time() <= self.vlm_cooldown:
                continue

            cropped_roi = cv_image[
                max(0, ymin-20):min(cv_image.shape[0], ymax+20),
                max(0, xmin-20):min(cv_image.shape[1], xmax+20)
            ]

            self.stop_robot()
            self.get_logger().info(f"🔍 YOLO proposed [{object_name}]. Asking VLM to match description...")

            try:
                result = self.query_vlm_match(cropped_roi)
                score = float(result.get("score", 0.0))
                match = bool(result.get("match", False))
                reason = result.get("reason", "")
                object_type = result.get("object_type", "unknown")

                self.get_logger().info(
                    f"🧠 VLM: match={match}, score={score:.2f}, object_type={object_type}, reason={reason}"
                )

                if match and score > best_score:
                    best_score = score
                    best_detection = detection
                    best_reason = reason
                    best_name = object_name

            except Exception as e:
                self.get_logger().warn(f"⚠️ VLM Error: {e}")

        if best_detection is not None and best_score >= 0.70:
            self.target_found = True
            self.get_logger().info(
                f"✅ VLM CONFIRMED TARGET: {self.target_object} | YOLO proposal={best_name} | score={best_score:.2f} | reason={best_reason}"
            )
            self.navigate_to_detection(best_detection)
            return

        if interesting_object_in_view and time.time() > self.vlm_cooldown:
            self.get_logger().info("❌ No crop matched strongly enough. Continuing search...")
            self.vlm_cooldown = time.time() + 2.0

        if not interesting_object_in_view and time.time() > self.vlm_cooldown:
            self.spin_robot_to_search()

        # cv2.imshow("Sumo Supervisor View", cv_image)
        # cv2.waitKey(1)


def main(args=None):
    print("\n" + "="*50)
    print("🧠 YOLO + VLM HYBRID SUPERVISOR NODE")
    print("="*50)
    user_target = input("What object should Sumo hunt for?: ")
    print(f"\n[SYSTEM] Target designated: '{user_target}'. Initializing ROS 2...\n")

    rclpy.init(args=args)
    node = YoloVlmCommander(target_object=user_target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        if hasattr(node, 'pipeline'):
            node.pipeline.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()