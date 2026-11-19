"""Launch GPU controller on car; provide odom_topic for installed localization."""
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import os
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_controller(context):
    """Apply explicit launch overrides after parameters loaded from car config."""
    overrides = {'odom_topic': LaunchConfiguration('odom_topic')}
    parameter_types = {
        'track': str, 'tracker': str, 'dynamics': str,
        'publish_selected_paths': bool, 'publish_top_paths': bool, 'use_cpu': bool,
        'horizon': int, 'max_speed': float,
    }
    for name, value_type in parameter_types.items():
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = ParameterValue(value, value_type=value_type)
    return [Node(
        package='mppi',
        executable='mppi_node.py',
        output='screen',
        parameters=[LaunchConfiguration('config_file'), overrides],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic', description='Car localization odometry topic'),
        DeclareLaunchArgument(
            'config_file', default_value=os.path.join(
                get_package_share_directory('mppi'), 'config', 'car_config.yaml'),
            description='Car ROS parameter file'),
        DeclareLaunchArgument(
            'track', default_value='', description='Override waypoint track name'),
        DeclareLaunchArgument(
            'tracker', default_value='', description='Override tracker: none or acados'),
        DeclareLaunchArgument(
            'dynamics', default_value='', description='Override dynamics: kinematic or dynamic'),
        DeclareLaunchArgument('publish_selected_paths', default_value=''),
        DeclareLaunchArgument('publish_top_paths', default_value=''),
        DeclareLaunchArgument('use_cpu', default_value=''),
        DeclareLaunchArgument('horizon', default_value=''),
        DeclareLaunchArgument('max_speed', default_value=''),
        OpaqueFunction(function=launch_controller),
    ])
