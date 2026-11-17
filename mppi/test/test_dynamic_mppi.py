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
            self.assertTrue(bool(c.trajectory_is_feasible(nominal, result.state.U, result.grid, self.cfg)))
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
        self.assertFalse(bool(c.trajectory_is_feasible(bad_X, bad, self.grid, self.cfg)))
        samples = jnp.stack((bad, self.U))
        trajectories = jnp.stack((bad_X, self.X))
        selected = c.select_feasible_controls(bad, samples, trajectories,
            jnp.zeros((2, self.cfg.H)), self.x0, self.grid, self.cfg)
        np.testing.assert_array_equal(selected, self.U)

    def test_feasible_mean_is_kept(self):
        selected = c.select_feasible_controls(self.U, self.U[None], self.X[None],
            jnp.zeros((1, self.cfg.H)), self.x0, self.grid, self.cfg)
        np.testing.assert_array_equal(selected, self.U)

    def test_blocked_horizon_reports_infeasible(self):
        blocked = replace(self.grid, cells=jnp.full(self.grid.spec.shape, c.og.OCCUPIED, dtype=jnp.uint8))
        solution = c.optimize_controls(self.state, self.x0,
            c.build_reference(self.x0, self.course, self.cfg), blocked, self.cfg)
        result = c.select_control_outputs(solution, self.x0,
            c.build_reference(self.x0, self.course, self.cfg), blocked, self.cfg, False)
        self.assertFalse(bool(result.feasible))
        self.assertTrue(np.isfinite(result.action).all())
        self.assertLessEqual(float(result.action[0]), 0.)

    def test_envelope_and_nonfinite_rejection(self):
        for column, value in ((4, 4.), (5, 10.), (3, -1.), (3, 5.), (0, float('nan'))):
            with self.subTest(column=column, value=value):
                invalid = self.X.at[-1, column].set(value)
                self.assertFalse(bool(c.trajectory_is_feasible(invalid, self.U, self.grid, self.cfg)))

    def test_oriented_footprint_and_out_of_grid_are_checked(self):
        local = jnp.array([[0., 0., 0.]])
        self.assertTrue(bool(c.db.footprint_is_free(local, self.grid, self.cfg)[0]))
        # Occupy front disc's center when body is rotated sideways, not COM cell.
        row = int(self.grid.spec.middle[1] / self.grid.spec.resolution)
        col = int((self.grid.spec.middle[0] + .2083) / self.grid.spec.resolution)
        blocked = replace(self.grid, cells=self.grid.cells.at[row, col].set(c.og.OCCUPIED))
        self.assertTrue(bool(c.db.footprint_is_free(local, blocked, self.cfg)[0]))
        turned = local.at[0, 2].set(jnp.pi / 2)
        self.assertFalse(bool(c.db.footprint_is_free(turned, blocked, self.cfg)[0]))
        outside = local.at[0, 0].set(5.95)
        self.assertFalse(bool(c.db.footprint_is_free(outside, self.grid, self.cfg)[0]))
        # Unknown self-occluded cells are allowed only inside present footprint.
        unknown = replace(self.grid, cells=jnp.full(self.grid.spec.shape, c.og.UNKNOWN, dtype=jnp.uint8))
        self.assertTrue(bool(c.db.footprint_is_free(local, unknown, self.cfg)[0]))
        moved = local.at[0, 0].set(.5)
        self.assertFalse(bool(c.db.footprint_is_free(moved, unknown, self.cfg)[0]))
        occupied_body = replace(unknown, cells=unknown.cells.at[row, int(3. / .1)].set(c.og.OCCUPIED))
        self.assertFalse(bool(c.db.footprint_is_free(local, occupied_body, self.cfg)[0]))

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
            header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=10, nanosec=50000000)),
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

    def test_dynamic_sensor_freshness_uses_message_clock(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.odom_callback(self.make_odom())
        self.assertAlmostEqual(node._latest_odom_stamp, 10.05)
        for now, sec, nanosec, expected in ((10.1, 10, 0, True),
                                           (10.3, 10, 0, False),
                                           (10.1, 9, 0, False),
                                           (10.1, 11, 0, False)):
            scan_stamp = types.SimpleNamespace(sec=sec, nanosec=nanosec)
            self.assertEqual(node._sensors_fresh(now, scan_stamp), expected)

    def test_plan_expiring_during_compute_is_not_published(self):
        node = object.__new__(c.MPPI)
        node.config = self.cfg
        node.get_clock, node.get_logger = Mock(), Mock()
        node.get_clock.return_value.now.return_value.nanoseconds = 11_000_000_000
        node._latest_odom_stamp = 10.05
        node._latest_scan = (types.SimpleNamespace(sec=10, nanosec=0),)
        self.assertFalse(node._control_result_is_publishable(True))
        node.get_clock.return_value.now.return_value.nanoseconds = 10_100_000_000
        self.assertTrue(node._control_result_is_publishable(True))

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
