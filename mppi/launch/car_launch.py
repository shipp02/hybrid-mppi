"""Launch the MPPI controller on the car, recording the run to a bag.

Controller settings live in config/car_config.yaml (edit, then restart; no rebuild).
The arguments below are only for quick changes at the track:

    ros2 launch hybrid_mppi car_launch.py
    ros2 launch hybrid_mppi car_launch.py max_speed:=1.5 run_label:=levine_1p5
    ros2 launch hybrid_mppi car_launch.py config_file:=$HOME/tune/aggressive.yaml
    ros2 launch hybrid_mppi car_launch.py record:=false

Each run is recorded until Ctrl-C (or until mppi_node exits) into
<run_logs>/<date>_<run_label>/: bag/ (MCAP), car_config.yaml, launch_args.yaml and
version.txt. On the laptop, tools/fetch_car_run.sh copies finished runs and removes
them from the car.

Any other parameter can be overridden ad hoc with ros2 run:

    ros2 run --prefix ~/car_jax_setup/.venv-jax/bin/python hybrid_mppi mppi_node.py \
      --ros-args --params-file <config> -p lambda:=0.2
"""
from datetime import datetime
import os
import shlex
import shutil
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

OVERRIDE_TYPES = {'track': str, 'max_speed': float, 'use_cpu': bool}
METRICS = ('route_speed', 'commanded_speed', 'measured_speed',
           'compute_ms', 'period_ms', 'odom_age_ms', 'scan_age_ms',
           'selected_mode_nominal', 'selected_mode_left', 'selected_mode_right',
           'fallback_used')
RECORDED_TOPICS = [
    '/tf', '/tf_static', '/map', '/scan',
    '/odom', '/pf/pose/odom', '/pf/viz/inferred_pose', '/pf/viz/particles',
    '/drive', '/teleop', '/ackermann_cmd', '/joy', '/sensors/core',
    '/mppi/waypoints', '/mppi/selected_paths', '/mppi/top_paths', '/mppi/best_path_costs',
    '/mppi/occupancy_grid', '/mppi/map_obstacles', '/mppi/corridor',
    *(f'/mppi/metrics/{name}' for name in METRICS),
    '/rosout',
]


def interpreter_prefix(context):
    """Run the node with the configured Python; 'shebang' keeps the script's own."""
    python = LaunchConfiguration('python_executable').perform(context)
    return None if python == 'shebang' else shlex.quote(os.path.expanduser(python))


def command_line_values(context):
    """Launch arguments given on the command line, as plain strings."""
    return {name: LaunchConfiguration(name).perform(context) for name in OVERRIDE_TYPES
            if LaunchConfiguration(name).perform(context)}


def command_line_overrides(context):
    """Only arguments given on the command line override the config file."""
    return {name: ParameterValue(LaunchConfiguration(name), value_type=OVERRIDE_TYPES[name])
            for name in command_line_values(context)}


def repository_version():
    """Commit and local changes of the checkout this launch file is linked from."""
    repository = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    result = subprocess.run(
        ['sh', '-c', 'git log -1 --oneline && git status --short'],
        cwd=repository, capture_output=True, text=True)
    return result.stdout or 'unknown (not a git checkout)\n'


def create_run_directory(context):
    """Make the run folder and store what this run was configured with."""
    root = os.path.expanduser(LaunchConfiguration('run_logs').perform(context))
    label = LaunchConfiguration('run_label').perform(context)
    run = os.path.join(root, f'{datetime.now():%Y%m%d_%H%M%S}_{label}')
    os.makedirs(run)
    shutil.copy(LaunchConfiguration('config_file').perform(context),
                os.path.join(run, 'car_config.yaml'))
    with open(os.path.join(run, 'launch_args.yaml'), 'w') as stream:
        stream.writelines(f'{name}: {value}\n'
                          for name, value in command_line_values(context).items())
    with open(os.path.join(run, 'version.txt'), 'w') as stream:
        stream.write(repository_version())
    return run


def bag_recorder(run):
    """rosbag2 gets SIGINT on launch shutdown, which flushes and closes the MCAP file."""
    print(f'[car_launch] recording to {run}')
    return ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-o', os.path.join(run, 'bag'),
             '--topics', *RECORDED_TOPICS],
        name='bag_record', output='log', sigterm_timeout='20')


def shutdown_when_exits(node):
    """End the whole launch, and so the recording, when the controller exits."""
    return RegisterEventHandler(OnProcessExit(
        target_action=node,
        on_exit=[EmitEvent(event=Shutdown(reason='mppi_node exited'))]))


def launch_controller(context):
    node = Node(
        package='hybrid_mppi',
        executable='mppi_node.py',
        output='screen',
        prefix=interpreter_prefix(context),
        # Jetson CPU and GPU share memory: allocate on demand instead of 75% upfront.
        additional_env={'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
        parameters=[LaunchConfiguration('config_file'), command_line_overrides(context)],
    )
    if LaunchConfiguration('record').perform(context).lower() != 'true':
        return [node]
    return [bag_recorder(create_run_directory(context)), node, shutdown_when_exits(node)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'python_executable', default_value='~/car_jax_setup/.venv-jax/bin/python',
            description="Python that runs the node (needs JAX); 'shebang' uses the script's own"),
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
        DeclareLaunchArgument(
            'record', default_value='true', description='Record this run to a bag'),
        DeclareLaunchArgument(
            'run_label', default_value='run', description='Suffix of the run folder name'),
        DeclareLaunchArgument(
            'run_logs', default_value='~/f1tenth_ws/run_logs/hybrid_mppi',
            description='Folder that collects recorded runs'),
        OpaqueFunction(function=launch_controller),
    ])
