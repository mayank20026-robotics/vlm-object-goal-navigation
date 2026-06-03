#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import math
import os
import sys
import json
import numpy as np
import cv2
import time
import base64
from openai import OpenAI  # <--- EDGE SERVER CLIENT
import depthai as dai


class VLMHybridCommander(Node):
    def __init__(self,target_object):
        super().__init__('vlm_hybrid_commander')
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.count_publisher(Twist,'/cmd_vel',10)

        self.is_processing = False
        self.timer = self.create_timer(1.0, self.execute_vlm_command)
        self.target_object = target_object


def main(args=None):
    rclpy.init(args=args)
    print("\n" + "="*50)
    print("🧠 ACTIVE SEARCH")
    print("="*50)
    target = input("Enter your semantic target (e.g., 'the red chair', 'the door'): ")
    print(f"Target locked: {target}. Booting ROS 2 node...\n")

    node = VLMHybridCommander(target_object=target)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node terminated by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
