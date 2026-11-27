"""Speed and latency metric helpers, runnable without ROS."""
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_route_speed_uses_nearest_waypoint(self):
        waypoints = np.array([[0., 0., 0., 1.0], [1., 0., 0., 2.0], [2., 0., 0., 3.0]])
        self.assertEqual(controller.route_speed_at(waypoints, 1.2, 0.3), 2.0)
        self.assertEqual(controller.route_speed_at(waypoints, -5., 0.), 1.0)

    def test_mode_and_fallback_counters_accumulate_once_per_tick(self):
        node = Mock()
        node.create_publisher.side_effect = lambda *args: Mock()
        with patch.object(controller, 'Float32', types.SimpleNamespace, create=True):
            metrics = controller.ControlMetricsPublisher(node)
            self.assertTrue(all(value == 0 for value in metrics._counts.values()))
            events = (
                [True, False, False, False],
                [False, True, False, True],
                [False, False, True, False],
                [False, True, False, True],
            )
            for event in events:
                metrics.publish({'compute_ms': 2.}, selection=jnp.asarray(event))
            self.assertEqual(metrics._counts, {
                'selected_mode_nominal': 1, 'selected_mode_left': 2,
                'selected_mode_right': 1, 'fallback_used': 2})
            self.assertEqual(sum(metrics._counts[name] for name in metrics.COUNTERS[:3]), 4)
            for name, expected in metrics._counts.items():
                metrics._publishers[name].publish.assert_any_call(
                    types.SimpleNamespace(data=float(expected)))

    def test_zero_increments_publish_unchanged_counters(self):
        with patch.object(controller, 'Float32', types.SimpleNamespace, create=True):
            node = Mock()
            node.create_publisher.side_effect = lambda *args: Mock()
            metrics = controller.ControlMetricsPublisher(node)
            metrics.publish({'compute_ms': 2.}, selection=jnp.zeros(4, dtype=bool))
            self.assertTrue(all(value == 0 for value in metrics._counts.values()))
            metrics._publishers['fallback_used'].publish.assert_called_once_with(
                types.SimpleNamespace(data=0.))

    def test_callback_forwards_resident_metrics_in_direct_and_tracker_paths(self):
        selection = jnp.array([False, True, False, True])
        result = types.SimpleNamespace(selection_metrics=selection)
        for tracker in (None, Mock()):
            node = object.__new__(controller.MPPI)
            node.get_clock = Mock()
            node._prepare_control_observation = Mock(return_value=Mock())
            node.tracker = tracker
            node.control_state = node.course = node.grid = node.config = Mock()
            node.publish_selected_paths_enabled = node.publish_top_paths_enabled = False
            node._publish_control_result = Mock()
            node._track_and_publish = Mock(return_value=result)
            node._publish_metrics = Mock()
            node._log_control_stats = Mock()
            with patch.object(controller, 'control_step', return_value=result):
                node.control_callback()
            node._publish_metrics.assert_called_once()
            self.assertIs(node._publish_metrics.call_args.args[-1], selection)

    def test_stamp_age(self):
        stamp = types.SimpleNamespace(sec=10, nanosec=500_000_000)
        self.assertAlmostEqual(controller.stamp_age_ms(10_520_000_000, stamp), 20.0)
        self.assertTrue(math.isnan(controller.stamp_age_ms(1, None)))


if __name__ == '__main__':
    unittest.main()
