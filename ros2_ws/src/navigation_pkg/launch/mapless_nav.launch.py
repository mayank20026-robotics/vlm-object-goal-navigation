# mapless_nav.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_params = os.path.join(
        get_package_share_directory('your_pkg'), 'config', 'nav2_params.yaml')

    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            parameters=[nav2_params],
            output='screen'),
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='local_costmap',
            parameters=[nav2_params],
            output='screen'),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            parameters=[nav2_params],
            output='screen'),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            parameters=[nav2_params],
            output='screen'),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            parameters=[nav2_params],
            output='screen'),
    ])