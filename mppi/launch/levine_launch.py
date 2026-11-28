"""What the autograder runs on Levine (levine_blocked, three laps, and the pose checks):

    ros2 launch hybrid_mppi levine_launch.py

Everything the graded run needs goes here: the nodes to start, the waypoints
to follow and the parameter values that suit Levine (spielberg_launch.py is
Spielberg's). Start your own nodes only: the simulator is already running.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

# scripts/mppi_node.py
EXECUTABLE = 'mppi_node.py'

# `track` tells the skeleton which waypoints to load; add this track's values
# for the other parameters your node declares, e.g. 'max_speed': 6.0, or give
# it a full .yaml config file
PARAMETERS = {
    'track': 'levine',
    'use_cpu': True,
    'odom_topic' : '/ego_racecar/odom',
    'publish_selected_paths': False,
    # Hybrid MPPI-MPC (arXiv 2208.02439): MPPI plans coarsely, the acados MPC tracks
    # the race line inside a collision-free corridor around that plan, and its
    # smoothed controls warm-start the next MPPI step (needs -DMPPI_BUILD_ACADOS=ON)
    'tracker': 'acados',
    'tracker_reference': 'corridor',
    'max_speed': 6.0,
    # plan from where the car will be when the command lands (~15 ms compute plus
    # sim stepping); gentler MPC steering corrections are robust to that delay
    'delay_compensation': 0.03,
    'corridor_steer_rate_weight': 200.0,
    # gym speed loop is P with gain ~4.75 accelerating, ~19 braking
    'speed_lead_time': 0.2,
    'brake_lead_time': 0.05,
    # sim odom is base_link (rear axle); sim.yaml mounts the LiDAR 0.275 m ahead
    'lidar_forward_offset': 0.275,
}


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hybrid_mppi',
            executable=EXECUTABLE,
            output='screen',
            parameters=[PARAMETERS],
        ),
    ])
