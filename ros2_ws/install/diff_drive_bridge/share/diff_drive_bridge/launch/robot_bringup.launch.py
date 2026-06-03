from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
import xacro
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction
 
def generate_launch_description():
 
    share_dir   = get_package_share_directory('Urdf_model_description')
    xacro_file  = os.path.join(share_dir, 'urdf', 'Urdf_model.xacro')
    rviz_config = os.path.join(share_dir, 'config', 'display.rviz')
 
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()
 
    # ── Launch arguments ──────────────────────────────────────
    gui_arg = DeclareLaunchArgument(
        name='gui', default_value='False')
    show_gui = LaunchConfiguration('gui')
 
    rviz_arg = DeclareLaunchArgument(
        name='use_rviz', default_value='False', description='Launch RViz?')
    use_rviz = LaunchConfiguration('use_rviz')
 
    # ── Robot state publisher ─────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_urdf}]
    )
 
    # ── Joint state publisher (GUI) ───────────────────────────
    joint_state_publisher = Node(
        condition=UnlessCondition(show_gui),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )
    joint_state_publisher_gui = Node(
        condition=IfCondition(show_gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
 
    # ── RPLidar A1 ────────────────────────────────────────────
    rplidar = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[{
            'serial_port':      '/dev/lidar',
            'serial_baudrate':  115200,
            'frame_id':         'laser_frame',
            'angle_compensate': True,
        }]
    )
 
    # ── OAK-D LITE DEPTH CAMERA ───────────────────────────────
    depthai_prefix = get_package_share_directory("depthai_ros_driver")
    robot_prefix   = get_package_share_directory("Urdf_model_description")
 
    oak_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(depthai_prefix, "launch", "camera.launch.py")
        ),
        launch_arguments={
            "name":             "oak",
            "params_file":      os.path.join(robot_prefix, "config", "oak_pcl.yaml"),
            "parent_frame":     "oak_data_frame",
            "imu.i_enable_imu": "false",
            "use_rviz":         "False",
        }.items()
    )
 
    # ── PointCloud processor inside camera container ──────────
    pointcloud_processor = LoadComposableNodes(
        target_container="oak_container",
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyziNode",
                name="point_cloud_xyzi",
                parameters=[{'output_frame': 'base_link'}],
                remappings=[
                    ("depth/image_rect",    "oak/stereo/image_raw"),
                    ("intensity/image_rect","oak/right/image_rect"),
                    ("intensity/camera_info","oak/stereo/camera_info"),
                    ("points",              "oak/points"),
                ],
            ),
        ],
    )
 
    # ── RF2O Laser Odometry ───────────────────────────────────
    rf2o_odometry = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic':       '/odom',
            'publish_tf':       True,
            'base_frame_id':    'base_footprint',
            'odom_frame_id':    'odom',
            'init_pose_from_topic': '',
            'freq':             10.0
        }]
    )
 
    # ── Diff drive bridge ─────────────────────────────────────
    diff_drive_bridge = Node(
        package='diff_drive_bridge',
        executable='noodom_diff_drive_bridge',
        name='noodom_diff_drive_bridge',
        output='screen'
    )
 
    # ── RViz ──────────────────────────────────────────────────
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
 
    # ── base_footprint static TF ──────────────────────────────
    base_footprint_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_footprint',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link']
    )
 
    # ── PointCloud → LaserScan (depth camera obstacles) ───────
    # Filters out floor and ceiling — only marks obstacles
    # between 5cm and 60cm height (below LiDAR at 63cm)
    pcl_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/oak/points'),
            ('scan',     '/oak/scan')   # separate topic from RPLidar /scan
        ],
        parameters=[{
            'target_frame':        'base_link',
            'transform_tolerance': 0.1,
            'min_height':          0.10,   # ignore floor (below 5cm)
            'max_height':          0.62,   # ignore ceiling (above 60cm = below LiDAR)
            'angle_min':          -0.87,   # ~50 degrees left FOV
            'angle_max':           0.87,   # ~50 degrees right FOV
            'angle_increment':     0.0087,
            'scan_time':           0.05,
            'range_min':           0.3,    # ignore camera self-reflections
            'range_max':           3.0,    # ignore distant noise
            'use_inf':             True,
            'inf_epsilon':         1.0
        }]
    )
 
    # ── Visual Odometry (disabled — rf2o used instead) ────────
    visual_odometry_node = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='stereo_odometry',
        output='screen',
        parameters=[{
            'frame_id':              'base_footprint',
            'odom_frame_id':         'odom',
            'publish_tf':            True,
            'approx_sync':           True,
            'qos_image':             2,
            'qos_camera_info':       2,
            'publish_null_when_lost': False,
            'Odom/ResetCountdown':   1
        }],
        remappings=[
            ('left/image_rect',   '/oak/left/image_rect'),
            ('right/image_rect',  '/oak/right/image_rect'),
            ('left/camera_info',  '/oak/left/camera_info'),
            ('right/camera_info', '/oak/right/camera_info')
        ]
    )
 
    return LaunchDescription([
        gui_arg,
        rviz_arg,
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rplidar,
        rf2o_odometry,
        diff_drive_bridge,
        rviz,
        base_footprint_tf,
        oak_camera,
        pointcloud_processor,   # generates /oak/points
        pcl_to_laserscan,       # filters height → publishes /oak/scan
        # visual_odometry_node  # disabled: rf2o used instead
    ])

