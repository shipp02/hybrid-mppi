"""What the autograder runs on Spielberg (one lap):

    ros2 launch hybrid_mppi spielberg_launch.py

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
