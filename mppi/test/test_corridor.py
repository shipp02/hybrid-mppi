"""Hybrid MPPI-MPC: corridor, race-line target, solver corridor and node path."""
import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from test_control_step import controller as c
import corridor as cr
import quintic_spline as qs
from acados_tracker import AcadosTracker, INACTIVE_RADIUS

BEAMS, ANGLE = 819, 2.356


def hallway_grid(half_width=1.0, cfg=c.MPPIConfig()):
    """LiDAR grid of a straight hallway, walls at +-half_width."""
    angles = np.linspace(-ANGLE, ANGLE, BEAMS)
    sin = np.abs(np.sin(angles))
    ranges = np.minimum(np.where(sin > 1e-6, half_width / np.maximum(sin, 1e-6), 25.), 25.)
    grid = c.og.OccupancyGrid.create(
        lidar_pose=(0.275, 0., 0.), width=3., height=7., middle=(1.5, 1.), resolution=.05,
        inflation_radius=.15, distance_cap=cfg.CORRIDOR_R_MAX + .15,
        num_beams=BEAMS, angle_min=-ANGLE, angle_max=ANGLE)
    return grid.from_scan(jnp.asarray(ranges, jnp.float32), 25.)


def straight_plan(lateral=0.0, length=5.0, points=41):
    x = np.linspace(0., length, points)
    return jnp.asarray(np.column_stack((x, x * 0 + lateral, x * 0, x * 0 + 2.)), jnp.float32)


class CorridorTest(unittest.TestCase):
    def setUp(self):
        self.cfg, self.grid, self.x0 = c.MPPIConfig(), hallway_grid(), jnp.array([0., 0., 0.])

    def test_balls_are_centered_on_the_plan(self):
        plan = straight_plan(lateral=0.3)
        centers, _ = cr.plan_corridor(plan, self.x0, self.grid, self.cfg)
        np.testing.assert_allclose(centers, plan[:, :2])

    def test_radius_is_clearance_minus_car_radius_capped(self):
        seen = slice(12, 22)  # past the blind spot under the car, inside the grid
        _, centered = cr.plan_corridor(straight_plan(), self.x0, self.grid, self.cfg)
        np.testing.assert_allclose(np.asarray(centered)[seen], self.cfg.CORRIDOR_R_MAX, atol=1e-6)
        _, near_wall = cr.plan_corridor(straight_plan(lateral=0.6), self.x0, self.grid, self.cfg)
        near_wall = np.asarray(near_wall)[seen]
        # 0.4 m to the wall, minus the 0.15 m car radius, within one 0.05 m cell
        np.testing.assert_allclose(near_wall, 0.25, atol=0.05)

    def test_nonpositive_room_invalidates_corridor(self):
        from dataclasses import replace
        grid = replace(self.grid, free_distance=jnp.zeros_like(self.grid.free_distance))
        _, radii = cr.plan_corridor(straight_plan(), self.x0, grid, self.cfg)
        np.testing.assert_array_equal(np.asarray(radii), 0.0)

    def test_occupied_centers_are_not_given_large_balls(self):
        from dataclasses import replace
        grid = replace(self.grid, log_odds=jnp.ones_like(self.grid.log_odds))
        _, radii = cr.plan_corridor(straight_plan(), self.x0, grid, self.cfg)
        np.testing.assert_array_equal(np.asarray(radii), 0.0)

    def test_out_of_grid_centers_are_invalid(self):
        plan = straight_plan(length=20.)
        _, radii = cr.plan_corridor(plan, self.x0, self.grid, self.cfg)
        self.assertEqual(float(radii[-1]), 0.0)

    def test_corridor_is_deterministic(self):
        first = cr.plan_corridor(straight_plan(), self.x0, self.grid, self.cfg)
        second = cr.plan_corridor(straight_plan(), self.x0, self.grid, self.cfg)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_line_reference_follows_plan_progress(self):
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        spline = qs.fit_quintic_spline(np.column_stack(
            (5 * np.cos(theta), 5 * np.sin(theta), theta, np.full(80, 6.))))
        angles = np.linspace(0, 0.6, 9)
        plan = jnp.asarray(np.column_stack((5.3 * np.cos(angles), 5.3 * np.sin(angles),
                                            angles + np.pi / 2, np.linspace(1, 2, 9))), jnp.float32)
        line = np.asarray(cr.line_reference_along_plan(plan, spline))
        np.testing.assert_allclose(np.hypot(line[:, 0], line[:, 1]), 5.0, atol=1e-2)  # on the line
        np.testing.assert_allclose(np.arctan2(line[:, 1], line[:, 0]), angles, atol=2e-2)
        np.testing.assert_allclose(line[:, 3], np.linspace(1, 2, 9), atol=1e-6)  # plan's speed


class TrackerCorridorTest(unittest.TestCase):
    def tracker(self, **config):
        tracker = object.__new__(AcadosTracker)
        tracker.cfg = c.MPPIConfig(H=3, **config)
        tracker.solver = Mock()
        return tracker

    def test_corridor_sets_balls_weights_and_steer_rate(self):
        tracker = self.tracker(CORRIDOR_STEER_RATE_WEIGHT=200.)
        centers, radii = np.arange(8.).reshape(4, 2), np.full(4, .4)
        tracker._load_corridor((centers, radii), np.zeros((4, 4)))
        params = [call.args[2] for call in tracker.solver.set.call_args_list]
        np.testing.assert_allclose(params, np.column_stack((centers, radii)))
        weights = tracker.solver.cost_set.call_args_list[0].args[2]
        self.assertEqual(weights[-1, -1], tracker.cfg.CORRIDOR_Q)
        self.assertEqual(weights[7, 7], 200.)  # steering change
        np.testing.assert_array_equal(np.diag(weights)[:3], 0.)
        self.assertEqual(weights[3, 3], 5.5)  # retain MPPI speed target
        self.assertGreater(weights[6, 6], 0.)  # acceleration change
        terminal = tracker.solver.cost_set.call_args_list[-1].args[2]
        np.testing.assert_array_equal(np.diag(terminal)[:3], 0.)
        self.assertEqual(terminal[3, 3], 5.5)
        self.assertEqual(weights[-1, -1], 0.)  # reference-free position by default

    def test_no_corridor_is_inactive_around_the_warm_start(self):
        tracker = self.tracker()
        nominal = np.column_stack((np.arange(4.), np.ones(4), np.zeros(4), np.ones(4)))
        tracker._load_corridor(None, nominal)
        params = np.array([call.args[2] for call in tracker.solver.set.call_args_list])
        np.testing.assert_allclose(params[:, :2], nominal[:, :2])
        np.testing.assert_allclose(params[:, 2], INACTIVE_RADIUS)
        weights = tracker.solver.cost_set.call_args_list[0].args[2]
        self.assertEqual((weights[-1, -1], weights[7, 7]), (0.0, 10.0))
        self.assertTrue(np.all(np.diag(weights)[:3] > 0.))

    def test_center_regularization_can_be_enabled_explicitly(self):
        tracker = self.tracker(CORRIDOR_Q=.02)
        tracker._load_corridor((np.zeros((4, 2)), np.full(4, .4)), np.zeros((4, 4)))
        weights = tracker.solver.cost_set.call_args_list[0].args[2]
        np.testing.assert_array_equal(np.diag(weights)[-2:], .02)

    def test_corridor_shape_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'corridor'):
            self.tracker()._load_corridor((np.zeros((2, 2)), np.zeros(2)), np.zeros((4, 4)))

    def test_invalid_corridor_is_rejected_before_setting_solver_balls(self):
        for radius in (0., -1.):
            tracker = self.tracker()
            with self.assertRaisesRegex(ValueError, 'Invalid corridor'):
                tracker._load_corridor((np.zeros((4, 2)), np.full(4, radius)), np.zeros((4, 4)))
            tracker.solver.set.assert_not_called()

    def test_valid_disconnected_balls_remain_soft_constraints(self):
        tracker = self.tracker()
        centers = np.column_stack((np.arange(4.) * 2., np.zeros(4)))
        tracker._load_corridor((centers, np.full(4, .1)), np.zeros((4, 4)))
        self.assertEqual(tracker.solver.set.call_count, 4)


class DelayCompensationTest(unittest.TestCase):
    def test_zero_delay_leaves_state(self):
        x0 = np.array([1., 2., .3, 2.])
        np.testing.assert_array_equal(c.compensate_delay(x0, [1., .1], c.MPPIConfig()), x0)

    def test_predicts_motion_under_applied_command(self):
        cfg = c.MPPIConfig(MAX_SPEED=6., DELAY_COMPENSATION=0.1)
        np.testing.assert_allclose(c.compensate_delay(np.array([0., 0., 0., 2.]), [0., 0.], cfg),
                                   [0.2, 0., 0., 2.], atol=1e-9)
        turning = c.compensate_delay(np.array([0., 0., 0., 2.]), [1., 0.2], cfg)
        self.assertGreater(turning[2], 0.)
        self.assertAlmostEqual(turning[3], 2.1)


class NodeCorridorTest(unittest.TestCase):
    def make_node(self, reference='corridor'):
        node = object.__new__(c.MPPI)
        node.config, node.tracker_reference = c.MPPIConfig(H=2), reference
        node.tracker = Mock()
        node.tracker.solve.return_value = np.array([[0.5, 0.1], [0.4, 0.2]])
        node._last_tracker_action = np.zeros(2)
        node.control_state = c.ControllerState(jax.random.PRNGKey(0), jnp.zeros((2, 2)), jnp.ones((2, 2)))
        node.publish_top_paths_enabled = node.publish_selected_paths_enabled = False
        node.publish_occupancy_grid_enabled = False
        node._publish_drive, node._publish_stop, node.get_logger = Mock(), Mock(), Mock()
        corridor = (np.zeros((3, 2)), np.full(3, .5))
        result = Mock(state=node.control_state, grid=None, reference=np.ones((3, 4)), top_paths=None)
        node._plan_for_tracker = Mock(return_value=(result, np.zeros((3, 4)),
                                                    corridor if reference == 'corridor' else None))
        return node

    def observation(self):
        return c.ControlObservation(np.array([0., 0., 0., 1.], np.float32), None, 0.05, False)

    def test_corridor_payload_uses_mppi_nominal_without_spline_projection(self):
        node = self.make_node()
        nominal = jnp.array([[0., 0., .2, 1.], [.1, .1, .3, 1.1], [.2, .1, .3, 1.2]])
        result = c.ControlResult(node.control_state, None, jnp.zeros(2), None)
        balls = (nominal[:, :2], jnp.full(3, .5))
        with patch.object(c, 'tracking_control_step', return_value=(result, nominal)), \
                patch.object(c, 'plan_corridor', return_value=balls):
            output, seed, _ = c.corridor_control_step.__wrapped__(
                node.control_state, self.observation(), None, None, node.config, jnp.zeros(3))
        np.testing.assert_array_equal(output.reference, nominal)
        np.testing.assert_array_equal(seed, nominal)

    def test_corridor_mode_preserves_mppi_targets_and_feeds_back(self):
        node = self.make_node()
        node._track_and_publish(self.observation(), None)
        args = node.tracker.solve.call_args.args
        np.testing.assert_array_equal(args[4], np.ones((3, 4)))  # MPPI targets; XY/yaw unweighted
        np.testing.assert_array_equal(args[5][1], np.full(3, .5))  # corridor radii
        np.testing.assert_allclose(node.control_state.U, [[0.5, 0.1], [0.4, 0.2]])  # Alg. 1
        np.testing.assert_array_equal(node._publish_drive.call_args.args[0], [0.5, 0.1])

    def test_other_modes_leave_mppi_mean_alone(self):
        node = self.make_node('spline')
        node._track_and_publish(self.observation(), None)
        self.assertIsNone(node.tracker.solve.call_args.args[5])
        np.testing.assert_allclose(node.control_state.U, 0.)

    def test_failed_solve_stops(self):
        node = self.make_node()
        node.tracker.solve.side_effect = RuntimeError('solve failed')
        node._track_and_publish(self.observation(), None)
        node._publish_stop.assert_called_once()
        node._publish_drive.assert_not_called()

    def test_invalid_corridor_stops_without_feeding_back_controls(self):
        node = self.make_node()
        node.tracker.solve.side_effect = ValueError('Invalid corridor')
        node._track_and_publish(self.observation(), None)
        node._publish_stop.assert_called_once()
        node._publish_drive.assert_not_called()
        np.testing.assert_array_equal(node.control_state.U, 0.)


if __name__ == '__main__':
    unittest.main()
