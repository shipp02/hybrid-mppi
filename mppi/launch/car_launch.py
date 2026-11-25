"""Launch GPU controller on car; provide odom_topic for installed localization."""
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import os
import shlex
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def interpreter_prefix(context):
    """Run the node with the configured Python; empty keeps the script shebang."""
    python = LaunchConfiguration('python_executable').perform(context)
    return shlex.quote(os.path.expanduser(python)) if python else None


def launch_controller(context):
    """Apply explicit launch overrides after parameters loaded from car config."""
    overrides = {'odom_topic': LaunchConfiguration('odom_topic')}
    parameter_types = {
        'track': str, 'tracker': str, 'dynamics': str,
        'publish_selected_paths': bool, 'publish_top_paths': bool, 'use_cpu': bool,
        'horizon': int, 'max_speed': float, 'steering_delay_s': float,
    }
    for name, value_type in parameter_types.items():
        argument = LaunchConfiguration(name)
        if argument.perform(context):
            overrides[name] = ParameterValue(argument, value_type=value_type)
    return [Node(
        package='hybrid_mppi',
        executable='mppi_node.py',
        output='screen',
        prefix=interpreter_prefix(context),
        # Jetson CPU and GPU share memory: allocate on demand instead of 75% upfront.
        additional_env={'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
        parameters=[LaunchConfiguration('config_file'), overrides],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'python_executable', default_value='~/car_jax_setup/.venv-jax/bin/python',
            description='Python that runs the node (needs JAX); empty uses the script shebang'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/pf/pose/odom',
            description='Override localization odometry topic (map frame, e.g. /pf/pose/odom)'),
        DeclareLaunchArgument(
            'config_file', default_value=os.path.join(
                get_package_share_directory('hybrid_mppi'), 'config', 'car_config.yaml'),
            description='Car ROS parameter file'),
        DeclareLaunchArgument(
            'track', default_value='levine_waypoints', description='Override waypoint track name'),
        DeclareLaunchArgument(
            'tracker', default_value='none', description='Override tracker: none or acados'),
        DeclareLaunchArgument(
            'dynamics', default_value='', description='Override dynamics: kinematic or dynamic'),
        DeclareLaunchArgument('publish_selected_paths', default_value=''),
        DeclareLaunchArgument('publish_top_paths', default_value=''),
        DeclareLaunchArgument('use_cpu', default_value=''),
        DeclareLaunchArgument('horizon', default_value=''),
        DeclareLaunchArgument('max_speed', default_value=''),
        DeclareLaunchArgument(
            'steering_delay_s', default_value='',
            description='Steering dead time in seconds (direct kinematic MPPI only)'),
        OpaqueFunction(function=launch_controller),
    ])
