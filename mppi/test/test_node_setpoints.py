"""Speed setpoint lead and scan-derived grid geometry, no ROS needed."""
import unittest

import numpy as np

from test_control_step import controller as c


class SpeedSetpointTest(unittest.TestCase):
    def test_lead_depends_on_accelerating_or_braking(self):
        cfg = c.MPPIConfig(MAX_SPEED=6., SPEED_LEAD_TIME=0.2, BRAKE_LEAD_TIME=0.05)
        x0 = np.array([0., 0., 0., 2.])
        self.assertAlmostEqual(c.speed_setpoint(x0, [2., 0.], cfg), 2.4)
        self.assertAlmostEqual(c.speed_setpoint(x0, [-2., 0.], cfg), 1.9)
        self.assertAlmostEqual(c.speed_setpoint(x0, [3., 0.], c.replace(cfg, MAX_SPEED=2.2)), 2.2)

    def test_default_lead_is_one_control_step(self):
        cfg = c.MPPIConfig()
        x0 = np.array([0., 0., 0., 1.])
        self.assertAlmostEqual(c.speed_setpoint(x0, [2., 0.], cfg), 1. + 2. * cfg.DT)
        self.assertAlmostEqual(c.speed_setpoint(x0, [-2., 0.], cfg), 1. - 2. * cfg.DT)

    def test_dynamic_mode_keeps_command_memory_step(self):
        cfg = c.MPPIConfig(DYNAMICS='dynamic', MAX_SPEED=4., SPEED_LEAD_TIME=0.5)
        x0 = np.array([0., 0., 0., 2., 0., 0., 0., 1.5])
        self.assertAlmostEqual(c.speed_setpoint(x0, [2., 0.], cfg), 1.5 + 2. * cfg.DT)


class ScanGeometryTest(unittest.TestCase):
    def test_grid_follows_scan_layout(self):
        scan = (np.zeros(819), -2.356, 4.712 / 818, 0.05, 25., 'laser', None)
        geometry = c.MPPI._scan_geometry(scan)
        self.assertEqual(geometry['num_beams'], 819)
        self.assertAlmostEqual(geometry['angle_min'], -2.356)
        self.assertAlmostEqual(geometry['angle_max'], 2.356)
        grid = c.og.OccupancyGrid.create(width=3., height=7., middle=(1.5, 1.), resolution=0.05,
                                         inflation_radius=0.15, distance_cap=0.25, **geometry)
        self.assertEqual(grid.from_scan(np.full(819, 5., np.float32)).cells.shape, grid.spec.shape)


if __name__ == '__main__':
    unittest.main()
