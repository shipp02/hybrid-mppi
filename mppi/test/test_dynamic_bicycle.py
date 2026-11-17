"""Dynamic model invariants, runnable with JAX CPU and without ROS."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from controller_config import MPPIConfig
import dynamic_bicycle as model


class DynamicBicycleTest(unittest.TestCase):
    def setUp(self):
        self.cfg = MPPIConfig(DYNAMICS='dynamic')
        self.x = jnp.array([0., 0., 0., 3., 0., 0., 0., 3.])

    def test_config_dimensions_and_validation(self):
        self.assertEqual(self.cfg.NX, 8)
        self.assertEqual(MPPIConfig().NX, 4)
        for field, value in [('MASS', 0.), ('TIRE_MU', float('nan')),
                             ('DYNAMIC_SUBSTEPS', 0), ('FRONT_LENGTH', 1.),
                             ('FRONT_DRIVE_FRACTION', 2.), ('BLEND_HIGH_SPEED', .1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.cfg, **{field: value})

    def test_straight_equilibrium_and_command_memory(self):
        step = jax.jit(model.step, static_argnames=('cfg',))
        actual = step(self.x, jnp.array([0., 0.]), self.cfg)
        expected = np.asarray(self.x).copy()
        expected[0] = 3 * self.cfg.DT
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        actual = step(self.x, jnp.array([2., .1]), self.cfg)
        self.assertAlmostEqual(float(actual[7]), 3.1, places=6)
        self.assertAlmostEqual(float(actual[6]), .1, places=6)
        self.assertGreater(float(actual[4]), 0.)
        self.assertGreater(float(actual[5]), 0.)
        # Speed servo response is not instantaneous imposed acceleration.
        self.assertGreater(float(actual[3]), 3.)
        self.assertLess(float(actual[3]), 3.1)

    def test_axles_obey_combined_friction_circles(self):
        lf, lr = self.cfg.FRONT_LENGTH, self.cfg.WB - self.cfg.FRONT_LENGTH
        capacity = self.cfg.TIRE_MU * self.cfg.MASS * 9.81 * np.array([lr, lf]) / self.cfg.WB
        for speed, vy, rate, steer, command in [(5., 2., -3., .4, 6.),
                                               (3., -3., 5., -.4, 0.)]:
            x = self.x.at[3:6].set(jnp.array([speed, vy, rate]))
            fx, fy = model.tire_forces(x, command, steer, self.cfg)
            self.assertTrue(np.all(np.hypot(fx, fy) <= capacity + 1e-5))

    def test_left_right_symmetry(self):
        left = model.step(self.x, jnp.array([0., .2]), self.cfg)
        right = model.step(self.x, jnp.array([0., -.2]), self.cfg)
        np.testing.assert_allclose(left, right * jnp.array([1, -1, -1, 1, -1, -1, -1, 1]), atol=1e-6)

    def test_finite_at_rest_and_blend_boundaries(self):
        step = jax.jit(model.step, static_argnames=('cfg',))
        for speed in (0., .01, self.cfg.BLEND_LOW_SPEED, self.cfg.BLEND_HIGH_SPEED):
            x = self.x.at[3].set(speed).at[7].set(speed)
            for _ in range(20):
                x = step(x, jnp.array([.5, .2]), self.cfg)
            self.assertTrue(np.isfinite(x).all())

    def test_projection_enforces_first_and_later_slew(self):
        cfg = replace(self.cfg, MAX_STEER_RATE=1.)
        U = jnp.array([[10., .4], [10., -.4], [-10., .4]])
        x = self.x.at[7].set(5.99)
        commands = model.project_controls(U, x, cfg)
        steering = np.r_[float(x[6]), np.asarray(commands[:, 1])]
        self.assertTrue(np.all(np.abs(np.diff(steering)) <= cfg.MAX_STEER_RATE * cfg.DT + 1e-6))
        targets = float(x[7]) + np.cumsum(np.asarray(commands[:, 0])) * cfg.DT
        self.assertTrue(np.all((targets >= 0) & (targets <= cfg.MAX_SPEED + 1e-6)))
        self.assertTrue(np.all(commands[:, 0] <= cfg.MAX_ACCEL + 1e-5))
        np.testing.assert_allclose(model.project_controls(commands, x, cfg), commands, atol=1e-5)

    def test_cost_projection_allows_heading_course_difference(self):
        beta = .3
        x = self.x.at[2].set(-beta).at[3].set(3 * np.cos(beta)).at[4].set(3 * np.sin(beta))
        tracking = model.tracking_state(x, self.cfg)
        self.assertAlmostEqual(float(tracking[2]), 0., places=6)
        self.assertAlmostEqual(float(tracking[3]), 3., places=6)
        np.testing.assert_array_equal(model.envelope_excess(x, self.cfg), np.zeros(4))

    def test_tire_saturation_differs_from_kinematic_yaw_prediction(self):
        cfg = replace(self.cfg, TIRE_MU=.3)
        x = self.x.at[3].set(6.).at[7].set(6.)
        step = jax.jit(model.step, static_argnames=('cfg',))
        for _ in range(5):
            x = step(x, jnp.array([0., .4]), cfg)
        kinematic_rate = float(x[3]) * np.tan(.4) / cfg.WB
        self.assertLess(float(x[5]), kinematic_rate * .75)
        self.assertGreater(abs(float(x[4])), .1)


if __name__ == '__main__':
    unittest.main()
