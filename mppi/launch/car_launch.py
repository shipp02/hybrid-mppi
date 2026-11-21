"""Launch GPU controller on car; odom_topic comes from car_config.yaml unless overridden.

    ros2 launch mppi car_launch.py [name:=value ...]

Parameters are read from config_file (car_config.yaml), then any launch
argument given here overrides it. [arg] = settable as name:=value on this
launch; every other parameter can go in config_file. Defaults below are the
node's; car_config.yaml sets track, max_speed and odom_topic for the car.

mppi_node ROS parameters (default, meaning)

Topics, track and device
  odom_topic        ''        [arg] map-frame localization odometry, e.g.
                              /pf/pose/odom (not /odom: wheel odometry drifts)
  track             levine    [arg] waypoint CSV in mppi/waypoints/ (.csv optional)
  use_cpu           false     [arg] run JAX on CPU; false needs a working GPU
  path_frame        map       frame of published paths and markers

Speed, horizon and command output
  max_speed         2.0       [arg] speed cap [m/s]; also caps waypoint speeds
  horizon           40        [arg] planning steps of 0.05 s (40 = 2 s)
  speed_lead_time   0.05      speed setpoint = v + accel * lead, accelerating [s]
  brake_lead_time   0.05      same lead when braking [s]

Planner selection
  planner           ''        [arg] mppi, mpc or hybrid; '' = mpc if tracker
                              is acados, else mppi
  low_speed_planner mpc       hybrid: planner from rest / below the switch
  high_speed_planner mppi     hybrid: planner above the switch
  high_enter_speed  4.0       [arg] hybrid: speed that starts the high planner [m/s]
  high_exit_speed   3.0       [arg] hybrid: speed that hands back to the low
                              planner [m/s]; must be below high_enter_speed
  switch_min_dwell  0.5       hybrid: minimum time between switches [s]

MPC tracker (acados)
  tracker           none      [arg] none or acados (needs -DMPPI_BUILD_ACADOS=ON)
  tracker_reference spline    [arg] MPC target: spline (quintic reference) or
                              nominal (the MPPI plan)
  tracker_prebuilt_dir ''     solver folder; '' = installed lib/mppi/acados_tracker
  tracker_max_steer_rate 3.14 steering slew limit [rad/s]; must match the solver

MPPI sampling and cost
  lambda            0.1       temperature; lower = favour the best samples more
  u_std             [1.5, 0.15]  sampling std of [accel, steer]
  u_std_min         [0.3, 0.03]  floor of that std under the covariance update
  cov_update        true      adapt the sampling std each step
  sampling          uniform   clip, truncated (normal) or uniform
  integrator        rk4       rollout integration: euler or rk4
  reference         quintic   reference path: quintic spline or linear
  obstacle_weight   10.0      scale of the blocked-cell (LiDAR grid) penalty

LiDAR occupancy grid
  lidar_offset      0.1016    LiDAR ahead of the odometry point [m] (car value;
                              the sim uses 0.275)
  grid_forward      6.0       grid reach ahead of the car [m]

Vehicle model
  dynamics          kinematic [arg] kinematic or dynamic (dynamic: no tracker)
  wb                0.33      wheelbase [m]
  max_steer_rate    3.14      dynamic: steering command slew [rad/s]
  Dynamic-only estimates, not identified car values: mass (3.5 kg),
  yaw_inertia (0.05), front_length (0.16 m), tire_mu (0.8),
  front_stiffness / rear_stiffness (20 N/rad), front_drive_fraction (0.5),
  speed_response_time (0.25 s), dynamic_substeps (10), blend_low_speed /
  blend_high_speed (0.5 / 1.5 m/s), lateral_response_time (0.1 s),
  max_sideslip (0.6 rad), max_yaw_rate (5.0 rad/s), envelope_weight (100)

Debug output (published from a side process)
  publish_selected_paths false [arg] recent chosen paths on /mppi/selected_paths
  publish_top_paths false     [arg] best sampled rollouts on /mppi/top_paths
  publish_occupancy_grid false  local grid on /mppi/occupancy_grid
"""
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import os
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_controller(context):
    """Apply explicit launch overrides after parameters loaded from car config."""
    overrides = {}
    parameter_types = {
        'odom_topic': str, 'track': str, 'tracker': str, 'tracker_reference': str, 'dynamics': str,
        'publish_selected_paths': bool, 'publish_top_paths': bool, 'use_cpu': bool,
        'horizon': int, 'max_speed': float,
        'planner': str, 'high_enter_speed': float, 'high_exit_speed': float,
    }
    for name, value_type in parameter_types.items():
        # empty = keep the config file value; pass the substitution itself so
        # launch_ros converts '1.0' / 'true' to value_type (a plain str is rejected)
        if LaunchConfiguration(name).perform(context):
            overrides[name] = ParameterValue(LaunchConfiguration(name), value_type=value_type)
    return [Node(
        package='mppi',
        executable='mppi_node.py',
        output='screen',
        parameters=[LaunchConfiguration('config_file'), overrides],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic', default_value='',
            description='Override localization odometry topic (map frame, e.g. /pf/pose/odom)'),
        DeclareLaunchArgument(
            'config_file', default_value=os.path.join(
                get_package_share_directory('mppi'), 'config', 'car_config.yaml'),
            description='Car ROS parameter file'),
        DeclareLaunchArgument(
            'track', default_value='', description='Override waypoint track name'),
        DeclareLaunchArgument(
            'tracker', default_value='', description='Override tracker: none or acados'),
        DeclareLaunchArgument(
            'tracker_reference', default_value='',
            description='Override MPC target: spline or nominal'),
        DeclareLaunchArgument(
            'dynamics', default_value='', description='Override dynamics: kinematic or dynamic'),
        DeclareLaunchArgument('publish_selected_paths', default_value=''),
        DeclareLaunchArgument('publish_top_paths', default_value=''),
        DeclareLaunchArgument('use_cpu', default_value=''),
        DeclareLaunchArgument('horizon', default_value=''),
        DeclareLaunchArgument('max_speed', default_value=''),
        DeclareLaunchArgument(
            'planner', default_value='', description='Override planner: mppi, mpc or hybrid'),
        DeclareLaunchArgument(
            'high_enter_speed', default_value='',
            description='Hybrid: speed that starts the high-speed planner [m/s]'),
        DeclareLaunchArgument(
            'high_exit_speed', default_value='',
            description='Hybrid: speed that hands back to the low-speed planner [m/s]'),
        OpaqueFunction(function=launch_controller),
    ])
