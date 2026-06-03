#!/usr/bin/env python3
import depthai as dai
import cv2
import time

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

print("====================================================")
print("OAK-D VPU VISION DEBUGGER")
print("====================================================")
print("Booting pipeline...")

# 1. Create pipeline
pipeline = dai.Pipeline()

# 2. Define sources and outputs
camRgb = pipeline.create(dai.node.Camera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.CameraSensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setFps(20)

monoLeft = pipeline.create(dai.node.Camera)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.CameraSensorResolution.THE_400_P)

monoRight = pipeline.create(dai.node.Camera)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.CameraSensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of the RGB camera
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(camRgb.getPreviewWidth(), camRgb.getPreviewHeight())

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# 3. Spatial Detection Network
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
spatialDetectionNetwork.setBlobPath(dai.Path("yolov6-nano.blob")) # Make sure you have this blob!
spatialDetectionNetwork.setConfidenceThreshold(0.35)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# YOLO specific parameters for yolov6-nano
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([]) # yolov6 is anchor free
spatialDetectionNetwork.setAnchorMasks({})
spatialDetectionNetwork.setIouThreshold(0.5)

camRgb.preview.link(spatialDetectionNetwork.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# 4. Create Output Queues
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
spatialDetectionNetwork.passthrough.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
spatialDetectionNetwork.out.link(xoutNN.input)

# 5. Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print("✅ Pipeline started. Press 'q' to quit.")
    
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            
            if inDet is not None:
                detections = inDet.detections
                
                for detection in detections:
                    # Map coordinates to the frame
                    x1 = int(detection.xmin * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    x2 = int(detection.xmax * frame.shape[1])
                    y2 = int(detection.ymax * frame.shape[0])
                    
                    try:
                        label = COCO_LABELS[detection.label]
                    except:
                        label = str(detection.label)
                        
                    conf = f"{detection.confidence * 100:.1f}%"
                    
                    # Get 3D Coordinates
                    z_dist = detection.spatialCoordinates.z / 1000.0
                    
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw Text Background
                    cv2.rectangle(frame, (x1, y1 - 40), (x1 + 150, y1), (0, 0, 0), -1)
                    
                    # Draw Text
                    cv2.putText(frame, f"{label} {conf}", (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Z: {z_dist:.2f}m", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("OAK-D Hardware YOLO Debugger", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
