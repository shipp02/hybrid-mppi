"""Direct slip-aware MPPI integration and publication contracts, no ROS needed."""
import types
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from test_control_step import controller as c


class DynamicMPPItest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(DYNAMICS='dynamic', H=6, T=16, MAX_SPEED=4.)
        grid = c.og.OccupancyGrid.create(
            width=6., height=8., middle=(3., 2.), resolution=.1,
            inflation_radius=.3, distance_cap=.4, num_beams=21)
        self.grid = replace(grid, cells=jnp.full(grid.spec.shape, c.og.FREE, dtype=jnp.uint8),
                            free_distance=jnp.full(grid.spec.shape, .4))
        self.x0 = jnp.array([0., 0., 0., 2., 0., 0., 0., 2.])
        self.U = jnp.zeros((self.cfg.H, 2))
        self.X = c.simulate(self.x0, self.U, self.cfg)
        x = jnp.linspace(0., 10., 101)
        self.course = jnp.stack((x, x * 0, x * 0, x * 0 + 2), axis=1)
        self.state = c.ControllerState(jax.random.PRNGKey(7), self.U,
            jnp.broadcast_to(jnp.array(self.cfg.U_STD), self.U.shape))
        self.obs = c.ControlObservation(self.x0, jnp.full((21,), 20.),
                                       jnp.float32(.05), jnp.bool_(False))

    def test_compiled_direct_control_and_visualization(self):
        for emit_viz in (False, True):
            args = self.state, self.obs, self.course, self.grid, self.cfg
            result = c.control_step(*args, emit_viz=emit_viz)
            jax.block_until_ready(result)
            with jax.transfer_guard('disallow'):
                resident = c.control_step(*args, emit_viz=emit_viz)
                jax.block_until_ready(resident)
            self.assertTrue(bool(result.feasible))
            self.assertTrue(np.isfinite(result.action).all())
            nominal = c.simulate(self.x0, result.state.U, self.cfg)
            self.assertTrue(bool(c.db.trajectory_is_feasible(nominal, result.state.U, self.cfg)))
            if emit_viz:
                self.assertEqual(result.visualization.reference.shape, (self.cfg.H + 1, 4))
                self.assertEqual(result.visualization.plan.shape, (self.cfg.H + 1, 8))
                np.testing.assert_allclose(result.visualization.plan, nominal, atol=1e-6)
            else:
                self.assertIsNone(result.visualization)

    def test_sampling_retains_nominal_and_respects_command_slew(self):
        samples = c.sample_rollout_controls(self.state.key, self.U, self.state.U_std,
                                           self.x0, self.cfg)
        np.testing.assert_array_equal(samples[0], self.U)
        steering = np.concatenate((np.full((self.cfg.T, 1), float(self.x0[6])),
                                    np.asarray(samples[:, :, 1])), axis=1)
        self.assertTrue(np.all(np.abs(np.diff(steering, axis=1))
                               <= self.cfg.MAX_STEER_RATE * self.cfg.DT + 1e-6))

    def test_weighted_nominal_rejected_in_favor_of_feasible_sample(self):
        bad = jnp.tile(jnp.array([0., .4]), (self.cfg.H, 1))
        # First steering jump cannot be executed at the configured slew rate.
        bad_X = c.simulate(self.x0, bad, self.cfg)
        self.assertFalse(bool(c.db.trajectory_is_feasible(bad_X, bad, self.cfg)))
        samples = jnp.stack((bad, self.U))
        trajectories = jnp.stack((bad_X, self.X))
        selected = c.select_feasible_controls(bad, samples, trajectories,
            jnp.zeros((2, self.cfg.H)), self.x0, self.cfg)
        np.testing.assert_array_equal(selected, self.U)

    def test_feasible_mean_is_kept(self):
        selected = c.select_feasible_controls(self.U, self.U[None], self.X[None],
            jnp.zeros((1, self.cfg.H)), self.x0, self.cfg)
        np.testing.assert_array_equal(selected, self.U)

    def test_envelope_and_nonfinite_rejection(self):
        for column, value in ((4, 4.), (5, 10.), (3, -1.), (3, 5.), (0, float('nan'))):
            with self.subTest(column=column, value=value):
                invalid = self.X.at[-1, column].set(value)
                self.assertFalse(bool(c.db.trajectory_is_feasible(invalid, self.U, self.cfg)))

    def test_course_tracking_cost_does_not_penalize_planned_body_slip(self):
        cfg = replace(self.cfg, Q=(0., 0., 1., 0.), Qf=(0., 0., 1., 0.),
                      R=(0., 0.), Rd=(0., 0.))
        beta = .3
        drifting = self.X.at[:, 2].set(-beta).at[:, 3].set(2 * np.cos(beta)).at[:, 4].set(2 * np.sin(beta))
        ref = c.db.tracking_state(self.X, cfg)
        result = c.cost(drifting, self.U, ref, cfg, self.grid)
        np.testing.assert_allclose(result, np.zeros(self.cfg.H), atol=1e-6)

    def test_dynamic_acados_pairing_rejected_before_loading_solver(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.get_parameter = Mock(return_value=types.SimpleNamespace(value='acados'))
        with self.assertRaisesRegex(ValueError, 'tracker:=none'):
            node._create_tracker()

    def make_odom(self):
        return types.SimpleNamespace(
            child_frame_id='base_com',
            pose=types.SimpleNamespace(pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=1., y=2.),
                orientation=types.SimpleNamespace(w=1., x=0., y=0., z=0.))),
            twist=types.SimpleNamespace(twist=types.SimpleNamespace(
                linear=types.SimpleNamespace(x=3., y=-.4),
                angular=types.SimpleNamespace(z=.8))))

    def test_odometry_preserves_signed_body_motion_and_requires_frame(self):
        msg = self.make_odom()
        measured = c.measured_state(msg, self.cfg)
        np.testing.assert_allclose(measured, [1., 2., 0., 3., -.4, .8])
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node._last_drive_command = np.array([.1, 3.2])
        np.testing.assert_allclose(node._observed_vehicle_state(measured),
                                   [1., 2., 0., 3., -.4, .8, .1, 3.2])
        msg.child_frame_id = ''
        with self.assertRaisesRegex(ValueError, 'child_frame_id'):
            c.measured_state(msg, self.cfg)

    def test_feasible_result_publishes_without_checking_sensor_age(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.get_clock = Mock(side_effect=AssertionError('unexpected age check'))
        node._publish_stop, node._publish_drive = Mock(), Mock()
        result = c.ControlResult(self.state, self.grid, jnp.array([0., .1]), None, jnp.bool_(True))
        node._publish_control_result(result, self.obs, 'now')
        node._publish_drive.assert_called_once()
        node._publish_stop.assert_not_called()
        node.get_clock.assert_not_called()

    def test_odometry_does_not_require_timestamp(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.odom_callback(self.make_odom())
        np.testing.assert_allclose(node._latest_odom, [1., 2., 0., 3., -.4, .8])

    def test_publication_uses_same_speed_command_as_rollout(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.drive_publisher = Mock()
        now = Mock()
        action = np.array([2., .1])
        msg = types.SimpleNamespace(header=types.SimpleNamespace(), drive=types.SimpleNamespace())
        with patch.object(c, 'AckermannDriveStamped', return_value=msg, create=True):
            node._publish_drive(action, self.x0, now)
        predicted = c.step(self.x0, action, self.cfg)
        self.assertAlmostEqual(msg.drive.speed, float(predicted[7]), places=6)
        np.testing.assert_allclose(node._last_drive_command, predicted[6:8], atol=1e-6)

    def test_infeasible_result_stops_instead_of_publishing_action(self):
        node = object.__new__(c.MPPI)
        node._publish_stop, node._publish_drive, node.get_logger = Mock(), Mock(), Mock()
        result = c.ControlResult(self.state, self.grid, jnp.array([3., .4]), None, jnp.bool_(False))
        node._publish_control_result(result, self.obs, 'now')
        node._publish_stop.assert_called_once_with('now')
        node._publish_drive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
