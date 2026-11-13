"""Launch GPU controller on car; provide odom_topic for installed localization."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic', description='Car localization odometry topic'),
        DeclareLaunchArgument(
            'track', default_value='levine', description='Waypoint track name'),
        Node(
            package='mppi',
            executable='mppi_node.py',
            output='screen',
            parameters=[{
                'track': LaunchConfiguration('track'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'use_cpu': False,
                'visualize': False,
            }],
        ),
    ])
