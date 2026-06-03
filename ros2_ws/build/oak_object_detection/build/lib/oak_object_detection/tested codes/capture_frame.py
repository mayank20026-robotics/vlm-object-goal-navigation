import cv2
import depthai as dai
import time
import numpy as np 
import os

def capture_rgb_and_depth():
    pipeline = dai.Pipeline()

    # 1. The Color Camera 
    camRGB = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbout = camRGB.requestOutput((1280, 720))

    # 2. Mono Cameras for Stereo Depth
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoLeftOut = monoLeft.requestOutput((640, 400))

    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    monoRightOut = monoRight.requestOutput((640, 400))

    # 3. Stereo Depth Node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    
    # --- THE FIX ---
    # Force the depth engine to output exactly 1280x720 so the width is divisible by 16
    stereo.setOutputSize(1280, 720)
    # ---------------

    # Align depth to the RGB camera lens
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # 4. Linking
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    print("Booting V3 pipeline.....")

    with pipeline:
        # Create queues
        qrgb = rgbout.createOutputQueue(maxSize=1, blocking=False)
        qdepth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
        
        pipeline.start()

        # Fetch the device to read calibration data
        device = pipeline.getDefaultDevice()
        calibdata = device.readCalibration()
        intrinsics = calibdata.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)

        print('\n--- CAMERA INTRINSICS ---')
        print(intrinsics)
        print("-------------------------\n")

        print("Warming up sensors......")
        time.sleep(5.0)

        print('Snapping Picture and Depth Map...')
        inrgb = qrgb.get()
        indepth = qdepth.get()

        cv_frame = inrgb.getCvFrame()
        depth_frame = indepth.getFrame()

        base_path = os.path.expanduser("~/ros2_ws/src/oak_object_detection/oak_object_detection/")

        cv2.imwrite(os.path.join(base_path, 'vlm_test_frame.jpg'), cv_frame)
        np.save(os.path.join(base_path, 'depth_data.npy'), depth_frame)

        print(f"Success! Saved RGB and Depth Matrix to: {base_path}")

if __name__ == '__main__':
    capture_rgb_and_depth()
