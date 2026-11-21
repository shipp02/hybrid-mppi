"""Car launch overrides, no ROS graph needed."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'car_launch.py'


def load_car_launch():
    """Execute car_launch.py against minimal launch stubs; return its node factory."""
    class Configuration:
        def __init__(self, name):
            self.name = name

        def perform(self, context):
            return context.get(self.name, '')

    launch = types.ModuleType('launch')
    launch.LaunchDescription = list
    actions = types.ModuleType('launch.actions')
    actions.DeclareLaunchArgument = lambda name, **kwargs: (name, kwargs)
    actions.OpaqueFunction = lambda function: function
    substitutions = types.ModuleType('launch.substitutions')
    substitutions.LaunchConfiguration = Configuration
    ros = types.ModuleType('launch_ros.actions')
    ros.Node = lambda **kwargs: types.SimpleNamespace(**kwargs)
    parameters = types.ModuleType('launch_ros.parameter_descriptions')
    parameters.ParameterValue = lambda value, value_type: (value, value_type)
    index = types.ModuleType('ament_index_python.packages')
    index.get_package_share_directory = lambda package: '/share/' + package
    namespace = {}
    with patch.dict(sys.modules, {
            'launch': launch, 'launch.actions': actions, 'launch.substitutions': substitutions,
            'launch_ros': types.ModuleType('launch_ros'), 'launch_ros.actions': ros,
            'launch_ros.parameter_descriptions': parameters,
            'ament_index_python': types.ModuleType('ament_index_python'),
            'ament_index_python.packages': index}):
        exec(compile(LAUNCH.read_text(), str(LAUNCH), 'exec'), namespace)
        description = namespace['generate_launch_description']()
    arguments = {item[0]: item[1] for item in description if isinstance(item, tuple)}
    return arguments, description[-1]


class CarLaunchTest(unittest.TestCase):
    def test_runs_without_arguments_using_config_odometry(self):
        arguments, launch_controller = load_car_launch()
        self.assertEqual(arguments['odom_topic']['default_value'], '')
        node, = launch_controller({})
        config, overrides = node.parameters
        self.assertNotIn('odom_topic', overrides)  # car_config.yaml supplies it
        config_text = (LAUNCH.parents[2] / 'config' / 'car_config.yaml').read_text()
        self.assertIn('odom_topic: /pf/pose/odom', config_text)

    def test_odom_topic_argument_overrides_config(self):
        _, launch_controller = load_car_launch()
        node, = launch_controller({'odom_topic': '/other/odom'})
        value, value_type = node.parameters[1]['odom_topic']
        self.assertEqual((value.name, value_type), ('odom_topic', str))

    def test_planner_and_thresholds_are_overridable(self):
        arguments, launch_controller = load_car_launch()
        for name in ('planner', 'high_enter_speed', 'high_exit_speed'):
            self.assertEqual(arguments[name]['default_value'], '')
        node, = launch_controller({'planner': 'hybrid', 'high_enter_speed': '1.5',
                                   'high_exit_speed': '1.0'})
        overrides = {name: (value.name, value_type)
                     for name, (value, value_type) in node.parameters[1].items()}
        # the substitution, not its text: launch_ros converts it to value_type
        self.assertEqual(overrides['planner'], ('planner', str))
        self.assertEqual(overrides['high_enter_speed'], ('high_enter_speed', float))
        self.assertEqual(overrides['high_exit_speed'], ('high_exit_speed', float))

    def test_unset_arguments_leave_config_file_values(self):
        _, launch_controller = load_car_launch()
        node, = launch_controller({})
        self.assertNotIn('planner', node.parameters[1])
        self.assertNotIn('high_enter_speed', node.parameters[1])


if __name__ == '__main__':
    unittest.main()
