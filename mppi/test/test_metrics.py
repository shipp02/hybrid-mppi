"""Speed and latency metric helpers, runnable without ROS."""
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_route_speed_uses_nearest_waypoint(self):
        waypoints = np.array([[0., 0., 0., 1.0], [1., 0., 0., 2.0], [2., 0., 0., 3.0]])
        self.assertEqual(controller.route_speed_at(waypoints, 1.2, 0.3), 2.0)
        self.assertEqual(controller.route_speed_at(waypoints, -5., 0.), 1.0)

    def test_stamp_age(self):
        stamp = types.SimpleNamespace(sec=10, nanosec=500_000_000)
        self.assertAlmostEqual(controller.stamp_age_ms(10_520_000_000, stamp), 20.0)
        self.assertTrue(math.isnan(controller.stamp_age_ms(1, None)))


if __name__ == '__main__':
    unittest.main()
