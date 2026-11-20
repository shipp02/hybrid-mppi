"""Speed-scheduled planner switch and its use in the node, no ROS needed."""
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_control_step import controller as c
from planner_switch import PlannerSwitch
import selected_path_publisher as spp


class PlannerSwitchTest(unittest.TestCase):
    def test_fixed_never_switches(self):
        switch = PlannerSwitch.fixed('mpc')
        self.assertEqual({switch.select(v) for v in (0., 3., 10., 0.)}, {'mpc'})
        self.assertEqual(switch.uses, {'mpc'})

    def test_hysteresis_between_thresholds(self):
        switch = PlannerSwitch('mpc', 'mppi', high_enter=3., high_exit=2.)
        speeds = [0., 2.5, 3., 2.5, 2.1, 2., 2.5]
        self.assertEqual([switch.select(v) for v in speeds],
                         ['mpc', 'mpc', 'mppi', 'mppi', 'mppi', 'mpc', 'mpc'])

    def test_dwell_delays_switch_back(self):
        switch = PlannerSwitch('mpc', 'mppi', high_enter=3., high_exit=2., min_dwell=3)
        self.assertEqual(switch.select(4.), 'mppi')  # starts eligible to switch
        self.assertEqual([switch.select(0.) for _ in range(3)], ['mppi', 'mppi', 'mpc'])

    def test_reset_returns_to_low_speed_planner(self):
        switch = PlannerSwitch('mppi', 'mpc', high_enter=1.5, high_exit=1.)
        switch.select(5.)
        switch.reset()
        self.assertEqual(switch.active, 'mppi')
        self.assertEqual(switch.select(5.), 'mpc')

    def test_invalid_configuration(self):
        # exit must sit strictly below enter: zero-width hysteresis is rejected
        for args in (('mpc', 'pid', 3., 2.), ('mpc', 'mppi', 2., 3.), ('mpc', 'mppi', 3., 3.)):
            with self.assertRaises(ValueError):
                PlannerSwitch(*args)
        with self.assertRaises(ValueError):
            PlannerSwitch('mpc', 'mppi', 3., 2., min_dwell=-1)
        self.assertEqual(PlannerSwitch.fixed('mppi').high_enter, math.inf)


class NodePlannerTest(unittest.TestCase):
    def make_node(self, mode, switch, tracker_error=None):
        node = object.__new__(c.MPPI)
        node.planner_mode, node.planner_switch = mode, switch
        node.tracker_reference = 'spline'
        node.tracker = Mock()
        node.tracker.solve.side_effect = tracker_error
        node.tracker.solve.return_value = np.array([[0.5, 0.1]])
        node._last_tracker_action = np.zeros(2)
        node.get_logger = Mock()
        return node

    def plan(self):
        return (np.array([1., -0.2]), np.zeros((3, 4)), np.zeros((2, 2)), np.ones((3, 4)))

    def test_mppi_and_mpc_commands(self):
        node = self.make_node('hybrid', PlannerSwitch('mpc', 'mppi', 3., 2.))
        x0 = np.zeros(4)
        action, planner = node._planner_action('mppi', self.plan(), x0)
        np.testing.assert_array_equal(action, [1., -0.2])
        node.tracker.solve.assert_not_called()
        action, planner = node._planner_action('mpc', self.plan(), x0)
        np.testing.assert_array_equal(action, [0.5, 0.1])
        self.assertEqual(planner, 'mpc')
        reference = node.tracker.solve.call_args.args[4]
        np.testing.assert_array_equal(reference, np.ones((3, 4)))

    def test_hybrid_falls_back_to_mppi_when_tracker_fails(self):
        node = self.make_node('hybrid', PlannerSwitch('mpc', 'mppi', 3., 2.),
                              RuntimeError('solve failed'))
        action, planner = node._planner_action('mpc', self.plan(), np.zeros(4))
        np.testing.assert_array_equal(action, [1., -0.2])
        self.assertEqual(planner, 'mppi')

    def test_fixed_mpc_stops_when_tracker_fails(self):
        node = self.make_node('mpc', PlannerSwitch.fixed('mpc'), RuntimeError('solve failed'))
        action, planner = node._planner_action('mpc', self.plan(), np.zeros(4))
        self.assertIsNone(action)
        self.assertEqual(planner, 'mpc')

    def test_pose_jump_restarts_schedule(self):
        node = self.make_node('hybrid', PlannerSwitch('mpc', 'mppi', 3., 2.))
        fast = c.ControlObservation(np.array([0., 0., 0., 4.]), None, 0.05, False)
        self.assertEqual(node._select_planner(fast), 'mppi')
        slow_after_jump = fast._replace(x0=np.array([0., 0., 0., 2.5]), reset=True)
        self.assertEqual(node._select_planner(slow_after_jump), 'mpc')


class PlannerSideCarTest(unittest.TestCase):
    def test_shared_planner_keeps_latest(self):
        import multiprocessing as mp
        shared = spp.SharedPlanner(mp.get_context('spawn'))
        self.assertEqual(shared.snapshot(), (-1, 0))
        shared.record('mpc')
        shared.record('mppi')
        self.assertEqual(shared.snapshot(), (spp.PLANNER_NAMES.index('mppi'), 2))
        with self.assertRaises(ValueError):
            shared.record('pid')

    def test_child_publishes_only_new_updates(self):
        import multiprocessing as mp
        shared = spp.SharedPlanner(mp.get_context('spawn'))
        node = Mock()
        publication = spp.PlannerPublication(node, shared)
        publication.publish()  # nothing recorded yet
        shared.record('mpc')
        publication.publish()
        publication.publish()  # same sequence: no repeat
        shared.record('mppi')
        publication.publish()
        sent = [call.args[0].data for call in publication.publisher.publish.call_args_list]
        self.assertEqual(sent, ['mpc', 'mppi'])

    def test_control_loop_hands_planner_to_side_car(self):
        node = object.__new__(c.MPPI)
        node.control_state, node.course, node.grid, node.config = None, None, None, c.MPPIConfig()
        node.publish_top_paths_enabled = node.publish_selected_paths_enabled = False
        node.planner_mode, node.planner_switch = 'mppi', PlannerSwitch.fixed('mppi')
        node.path_publisher = Mock()
        node._last_tracker_action = np.zeros(2)
        for name in ('_publish_occupancy_grid', '_publish_drive', '_publish_top_paths', '_publish_stop'):
            setattr(node, name, Mock())
        result = Mock(action=np.array([1., 0.1]), reference=np.zeros((3, 4)), top_paths=None)
        observation = c.ControlObservation(np.array([0., 0., 0., 1.]), None, 0.05, False)
        with patch.object(c, 'tracking_control_step', return_value=(result, np.zeros((3, 4)))), \
                patch.object(c.jax, 'device_get', side_effect=lambda tree: tree):
            node._plan_and_publish(observation, Mock())
        node.path_publisher.planner.record.assert_called_once_with('mppi')
        node._publish_drive.assert_called_once()
        self.assertFalse(hasattr(node, 'planner_publisher'))


if __name__ == '__main__':
    unittest.main()
