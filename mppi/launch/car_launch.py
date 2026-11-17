"""Launch GPU controller on car; provide odom_topic for installed localization."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic', description='Car localization odometry topic'),
        DeclareLaunchArgument(
            'track', default_value='levine', description='Waypoint track name'),
        DeclareLaunchArgument(
            'tracker', default_value='none', description='none or acados'),
        DeclareLaunchArgument('use_cpu', default_value='false'),
        DeclareLaunchArgument('horizon', default_value='40'),
        DeclareLaunchArgument('max_speed', default_value='6.0'),
        Node(
            package='mppi',
            executable='mppi_node.py',
            output='screen',
            parameters=[{
                'track': LaunchConfiguration('track'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'use_cpu': ParameterValue(LaunchConfiguration('use_cpu'), value_type=bool),
                'horizon': ParameterValue(LaunchConfiguration('horizon'), value_type=int),
                'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                'visualize': False,
                'tracker': LaunchConfiguration('tracker'),
            }],
        ),
    ])
