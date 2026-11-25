"""Launch the MPPI controller on the car.

Controller settings live in config/car_config.yaml (edit, then restart; no rebuild).
The arguments below are only for quick changes at the track:

    ros2 launch hybrid_mppi car_launch.py
    ros2 launch hybrid_mppi car_launch.py max_speed:=1.5 track:=levine_waypoints.csv
    ros2 launch hybrid_mppi car_launch.py config_file:=$HOME/tune/aggressive.yaml

Any other parameter can be overridden ad hoc with ros2 run:

    ros2 run --prefix ~/car_jax_setup/.venv-jax/bin/python hybrid_mppi mppi_node.py \
      --ros-args --params-file <config> -p lambda:=0.2
"""
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import os
import shlex
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

OVERRIDE_TYPES = {'track': str, 'max_speed': float, 'use_cpu': bool}


def interpreter_prefix(context):
    """Run the node with the configured Python; empty keeps the script shebang."""
    python = LaunchConfiguration('python_executable').perform(context)
    return shlex.quote(os.path.expanduser(python)) if python else None


def command_line_overrides(context):
    """Only arguments given on the command line override the config file."""
    overrides = {}
    for name, value_type in OVERRIDE_TYPES.items():
        argument = LaunchConfiguration(name)
        if argument.perform(context):
            overrides[name] = ParameterValue(argument, value_type=value_type)
    return overrides


def launch_controller(context):
    return [Node(
        package='hybrid_mppi',
        executable='mppi_node.py',
        output='screen',
        prefix=interpreter_prefix(context),
        # Jetson CPU and GPU share memory: allocate on demand instead of 75% upfront.
        additional_env={'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
        parameters=[LaunchConfiguration('config_file'), command_line_overrides(context)],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'python_executable', default_value='~/car_jax_setup/.venv-jax/bin/python',
            description='Python that runs the node (needs JAX); empty uses the script shebang'),
        DeclareLaunchArgument(
            'config_file', default_value=os.path.join(
                get_package_share_directory('hybrid_mppi'), 'config', 'car_config.yaml'),
            description='Controller parameter file'),
        DeclareLaunchArgument(
            'track', default_value='', description='Waypoint CSV (default: config file)'),
        DeclareLaunchArgument(
            'max_speed', default_value='', description='Speed cap [m/s] (default: config file)'),
        DeclareLaunchArgument(
            'use_cpu', default_value='', description='Run JAX on CPU (default: config file)'),
        OpaqueFunction(function=launch_controller),
    ])
