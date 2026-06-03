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
    ])

