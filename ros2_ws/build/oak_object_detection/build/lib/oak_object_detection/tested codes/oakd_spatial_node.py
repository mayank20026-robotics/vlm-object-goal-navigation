#!/usr/bin/env python3

import depthai as dai
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


fps = 20
size = (640, 400) # Added: Stereo math needs a strict resolution
modelDescription = dai.NNModelDescription("yolov6-nano")

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

class TargetReceiver(dai.node.HostNode):
    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        self.ros_callback = None
        
    def build(self, detection_out: dai.Node.Output):
        self.link_args(detection_out)

    def process(self, in_data):
        if in_data and self.ros_callback is not None:
            self.ros_callback(in_data)

class OakDNode(Node):

    def __init__(self):
        super().__init__('oakd_spatial_node')

        # Mouth of the system
        self.publisher = self.create_publisher(Point, "target_coordinate", 10)
        self.get_logger().info("Initializing OAK D lite VPU...")
        self.pipeline = dai.Pipeline()

        rgbcamera = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps) # Fixed: sensorFps
        monoleft = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)  # Fixed: Camera & CAM_B
        monoright = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps) # Fixed: Camera & CAM_C

        stereo_depth = self.pipeline.create(dai.node.StereoDepth)
        stereo_depth.setExtendedDisparity(True)

        # Fixed: requestOutput before linking
        monoleft.requestOutput(size).link(stereo_depth.left)
        monoright.requestOutput(size).link(stereo_depth.right)

        spatial_detection = self.pipeline.create(dai.node.SpatialDetectionNetwork).build(rgbcamera, stereo_depth, modelDescription)

        spatial_detection.spatialLocationCalculator.initialConfig.setSegmentationPassthrough(False)
        spatial_detection.input.setBlocking(False)
        spatial_detection.setDepthLowerThreshold(100)
        spatial_detection.setDepthUpperThreshold(5000)
        
        # Attaching the custom reciever
        self.reciever = self.pipeline.create(TargetReceiver)
        self.reciever.build(spatial_detection.out)

        # The bridge between ros_callback and publish to ros.
        self.reciever.ros_callback = self.publish_to_ros

        # Ignite the camera
        self.pipeline.run()
        self.device.startPipeline(self.pipeline)
        self.get_logger().info("Camera Pipeline Started broadcasting to target_coordinates")

    def publish_to_ros(self,indata):
        if indata is not None:
            for detection in indata.detections:
                try:
                    object_name = COCO_LABELS[detection.label]
                except:
                    object_name = f"Unknown object ID: {detection.label}"

                msg = Point()
                msg.x = detection.spatialCoordinates.x / 1000.0
                msg.y = detection.spatialCoordinates.y / 1000.0
                msg.z = detection.spatialCoordinates.z / 1000.0

                self.publisher.publish(msg)
                self.get_logger().info(f"Target Acquired:[{object_name}] | X: {msg.x:.2f}m | Z: {msg.z:.2f}m")

    def destroy_node(self):
        super().destroy_node()

def main(agrs=None):
    rclpy.init(args=agrs)
    node = OakDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()





