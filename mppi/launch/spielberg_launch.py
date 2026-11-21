"""What the autograder runs on Spielberg (one lap):

    ros2 launch mppi spielberg_launch.py

Everything the graded run needs goes here: the nodes to start, the waypoints
to follow and the parameter values that suit Spielberg (levine_launch.py is
Levine's). Start your own nodes only: the simulator is already running.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

# scripts/mppi_node.py
EXECUTABLE = 'mppi_node.py'

# `track` tells the skeleton which waypoints to load; add this track's values
# for the other parameters your node declares, e.g. 'max_speed': 6.0, or give
# it a full .yaml config file
PARAMETERS = {
    'track': 'spielberg',
    'use_cpu': True,
    'odom_topic': '/ego_racecar/odom',
    # acados MPC tracks the quintic spline reference (needs -DMPPI_BUILD_ACADOS=ON)
    'tracker': 'acados',
    'tracker_reference': 'spline',
    # MPC from rest, MPPI from 4 m/s, back to MPC below 3 m/s
    'planner': 'hybrid',
    'max_speed': 6.0,
    # gym speed loop is P with gain ~4.75 accelerating, ~19 braking
    'speed_lead_time': 0.2,
    'brake_lead_time': 0.05,
    # sim odom is base_link (rear axle); sim.yaml mounts the LiDAR 0.275 m ahead
    'lidar_offset': 0.275,
    # less steering noise and a greedier average: noisy steering penalized fast
    # samples, so MPPI averaged toward braking and cruised at ~3 m/s
    'u_std': [1.5, 0.05],
    'u_std_min': [0.3, 0.02],
    'lambda': 0.03,
}


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mppi',
            executable=EXECUTABLE,
            output='screen',
            parameters=[PARAMETERS],
        ),
    ])
