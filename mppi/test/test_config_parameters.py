"""ROS parameter mapping for MPPIConfig, runnable without ROS."""
import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller  # noqa: E402


class ConfigParameterTest(unittest.TestCase):
    def test_every_tunable_field_has_a_unique_parameter(self):
        names = [controller.config_parameter_name(f.name)
                 for f in controller.tunable_config_fields()]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn('horizon', names)
        self.assertIn('num_samples', names)
        self.assertIn('q', names)
        self.assertNotIn('nx', names)
        self.assertEqual(len(names), len(fields(controller.MPPIConfig)) - 2)

    def test_defaults_round_trip(self):
        cfg = controller.MPPIConfig()
        values = {}
        for field in controller.tunable_config_fields():
            name = controller.config_parameter_name(field.name)
            ros_value = controller.ros_parameter_default(field.default)
            values[field.name] = controller.config_value(name, field.default, ros_value)
        self.assertEqual(controller.replace(cfg, **values), cfg)

    def test_yaml_style_values_are_cast(self):
        self.assertEqual(controller.config_value('q', (1., 2.), [3, 4]), (3., 4.))
        self.assertEqual(controller.config_value('max_speed', 2.0, 3), 3.0)
        self.assertIsInstance(controller.config_value('max_speed', 2.0, 3), float)
        self.assertEqual(controller.config_value('horizon', 40, 20), 20)
        self.assertIs(controller.config_value('cov_update', True, False), False)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            controller.config_value('q', (1., 2., 3., 4.), [1., 2.])
        with self.assertRaises(ValueError):
            controller.config_value('cov_update', True, 'false')
        with self.assertRaises(ValueError):
            controller.MPPIConfig(H=0)


if __name__ == '__main__':
    unittest.main()
