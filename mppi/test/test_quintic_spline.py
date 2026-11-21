"""Quintic spline reference for MPPI and the acados tracker, no ROS needed."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import make_interp_spline

from test_control_step import controller as c
import quintic_spline as qs
from acados_tracker import prepare_reference

RADIUS = 5.0
SPEED = 2.0


def circle_waypoints(n=80, jitter=0.0, seed=0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = theta + np.random.default_rng(seed).uniform(-jitter, jitter, n)
    return np.column_stack((RADIUS * np.cos(theta), RADIUS * np.sin(theta),
                            theta + np.pi / 2, np.full(n, SPEED)))


class QuinticSplineTest(unittest.TestCase):
    def test_segments_reproduce_periodic_quintic(self):
        waypoints = circle_waypoints(jitter=0.02)
        closed = np.vstack((waypoints[:, :2], waypoints[:1, :2]))
        h = np.hypot(*np.diff(closed, axis=0).T)
        knots = np.r_[0.0, np.cumsum(h)]
        reference = make_interp_spline(knots, closed, k=5, bc_type='periodic')
        p, v, a = (reference(knots, nu) for nu in range(3))
        coeffs = qs._hermite_coefficients(p[:-1], v[:-1], a[:-1], p[1:], v[1:], a[1:], h)
        s = np.random.default_rng(1).uniform(0, knots[-1], 500)
        segment = np.searchsorted(knots, s, side='right') - 1
        tau = s - knots[segment]
        mine = np.einsum('nk,nkd->nd', tau[:, None] ** np.arange(6), coeffs[segment])
        np.testing.assert_allclose(mine, reference(s), atol=1e-10)

    def test_repeated_endpoint_is_dropped(self):
        waypoints = circle_waypoints()
        spline = qs.fit_quintic_spline(np.vstack((waypoints, waypoints[:1])))
        self.assertEqual(spline.coeffs.shape[0], len(waypoints))

    def test_reference_follows_circle(self):
        spline = qs.fit_quintic_spline(circle_waypoints())
        ref = np.asarray(qs.sample_quintic_reference(RADIUS, 0.0, spline, 0.05, 20))
        self.assertEqual(ref.shape, (21, 4))
        np.testing.assert_allclose(np.hypot(ref[:, 0], ref[:, 1]), RADIUS, atol=1e-3)
        tangent = np.arctan2(ref[:, 1], ref[:, 0]) + np.pi / 2
        np.testing.assert_allclose(np.angle(np.exp(1j * (ref[:, 2] - tangent))), 0, atol=1e-3)
        steps = np.hypot(*np.diff(ref[:, :2], axis=0).T)
        np.testing.assert_allclose(steps, SPEED * 0.05, rtol=1e-2)
        np.testing.assert_allclose(ref[:, 3], SPEED)

    def test_reference_wraps_across_lap_end(self):
        spline = qs.fit_quintic_spline(circle_waypoints())
        start = qs.evaluate_pose(spline, spline.knots[-1] - 0.1)[0]
        ref = np.asarray(qs.sample_quintic_reference(*start, spline, 0.05, 10))
        np.testing.assert_allclose(np.hypot(ref[:, 0], ref[:, 1]), RADIUS, atol=1e-3)
        self.assertTrue(np.all(np.hypot(*np.diff(ref[:, :2], axis=0).T) < 0.2))

    def test_speed_ramps_from_current_speed(self):
        spline = qs.fit_quintic_spline(circle_waypoints())
        ref = qs.sample_quintic_reference(RADIUS, 0.0, spline, 0.05, 10,
                                          v0=0.5, max_accel=3.0)
        np.testing.assert_allclose(ref[:4, 3], [0.5, 0.65, 0.8, 0.95], atol=1e-6)


class QuinticControlStepTest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(H=5, T=8)
        self.grid = c.og.OccupancyGrid.create(
            width=1.0, height=2.0, middle=(0.5, 0.3), resolution=0.2,
            num_beams=21, distance_cap=0.2, inflation_radius=0.1)
        self.course = c.make_course(circle_waypoints())
        self.state = c.ControllerState(
            jax.random.PRNGKey(7), jnp.zeros((self.cfg.H, self.cfg.NU)),
            jnp.broadcast_to(jnp.array(self.cfg.U_STD), (self.cfg.H, self.cfg.NU)))
        self.observation = c.ControlObservation(
            jnp.array([RADIUS, 0., np.pi / 2, 1.]), jnp.full((21,), 3.),
            jnp.float32(0.05), jnp.bool_(False))

    def test_quintic_is_default_and_drives_reference(self):
        self.assertEqual(c.MPPIConfig().REFERENCE, 'quintic')
        result = c.control_step(self.state, self.observation, self.course,
                                self.grid, self.cfg)
        x0 = self.observation.x0
        expected = qs.sample_quintic_reference(
            x0[0], x0[1], self.course.spline, self.cfg.DT, self.cfg.H,
            v0=x0[3], max_accel=self.cfg.MAX_ACCEL)
        np.testing.assert_allclose(result.reference, expected, atol=1e-6)
        self.assertTrue(np.isfinite(result.action).all())

    def test_resident_dispatch_with_quintic_course(self):
        # observation's range_max default is a host scalar; place it on device too
        args = (self.state, jax.device_put(self.observation),
                jax.device_put(self.course), self.grid, self.cfg)
        jax.block_until_ready(c.control_step(*args))
        with jax.transfer_guard('disallow'):
            jax.block_until_ready(c.control_step(*args))


class TrackerReferenceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(H=4)
        self.x0 = np.array([0., 0., np.pi - 0.05, 1.])
        self.states = np.tile(self.x0, (5, 1))
        self.controls = np.zeros((4, 2))
        self.previous = np.zeros(2)

    def test_nominal_is_target_without_reference(self):
        _, nominal, _, targets = prepare_reference(
            self.x0, self.states, self.controls, self.previous, self.cfg)
        np.testing.assert_allclose(targets, self.states)
        np.testing.assert_allclose(nominal[1:, :4], self.states[1:])

    def test_spline_reference_is_target_with_aligned_yaw(self):
        reference = self.states.copy()
        reference[:, 0] = np.arange(5) * 0.1
        reference[:, 2] = [np.pi - 0.04, -np.pi + 0.02, -np.pi + 0.06, -np.pi + 0.1,
                           -np.pi + 0.14]  # crosses +-pi
        _, nominal, _, targets = prepare_reference(
            self.x0, self.states, self.controls, self.previous, self.cfg, reference)
        np.testing.assert_allclose(targets[:, 0], reference[:, 0])
        self.assertTrue(np.all(np.abs(np.diff(targets[:, 2])) < 0.1))
        self.assertLess(abs(targets[0, 2] - self.x0[2]), 0.1)
        np.testing.assert_allclose(nominal[1:, :4], self.states[1:])

    def test_reference_shape_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            prepare_reference(self.x0, self.states, self.controls, self.previous,
                              self.cfg, np.zeros((3, 4)))


if __name__ == '__main__':
    unittest.main()
