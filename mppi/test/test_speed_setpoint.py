"""Kinematic speed setpoint: starting from rest, lead limit and bounds."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller  # noqa: E402


class SpeedSetpointTest(unittest.TestCase):
    def setUp(self):
        self.cfg = controller.MPPIConfig(MAX_SPEED=3.0)  # DT 0.05, lead 1.0

    def setpoint(self, previous, measured, accel):
        return controller.kinematic_speed_setpoint(previous, measured, accel, self.cfg)

    def test_rises_from_rest_while_the_car_has_not_moved(self):
        # Recorded stall: measured speed stays 0 and the planner asks for ~2 m/s^2.
        # The old rule (measured + a*dt) published 0.10 m/s forever.
        setpoint = 0.0
        for _ in range(10):  # 0.5 s
            setpoint = self.setpoint(setpoint, 0.0, 2.0)
        self.assertAlmostEqual(setpoint, 1.0)  # 10 * 0.1, at the lead limit

    def test_lead_limit_caps_the_setpoint_above_measured_speed(self):
        self.assertAlmostEqual(self.setpoint(2.5, 1.0, 3.0), 2.0)

    def test_braking_lowers_the_last_setpoint(self):
        self.assertAlmostEqual(self.setpoint(2.0, 2.0, -3.0), 1.85)

    def test_bounds(self):
        self.assertEqual(self.setpoint(0.05, 0.0, -3.0), 0.0)
        self.assertEqual(self.setpoint(2.95, 2.9, 3.0), 3.0)

    def test_lead_must_be_positive(self):
        with self.assertRaises(ValueError):
            controller.MPPIConfig(MAX_SPEED_LEAD=0.0)


if __name__ == '__main__':
    unittest.main()
