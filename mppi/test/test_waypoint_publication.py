"""Static waypoint publication contract, runnable without ROS or JAX."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock


def marker_message():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='', stamp=None),
        pose=SimpleNamespace(orientation=SimpleNamespace(w=0.0)),
        scale=SimpleNamespace(x=0.0, y=0.0),
        color=SimpleNamespace(g=0.0, a=0.0),
        lifetime=SimpleNamespace(sec=0, nanosec=0),
    )


def load_publication():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'mppi_node.py'
    tree = ast.parse(path.read_text())
    node_class = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == 'MPPI')
    method = next(node for node in node_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_publish_waypoints')
    marker = Mock(side_effect=marker_message, POINTS=8, ADD=0)
    scope = dict(
        Marker=marker, Point=lambda **kwargs: SimpleNamespace(**kwargs),
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        HistoryPolicy=SimpleNamespace(KEEP_LAST='keep_last'),
        ReliabilityPolicy=SimpleNamespace(RELIABLE='reliable'),
        DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL='transient_local'),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_publish_waypoints'], node_class


class WaypointPublicationTest(unittest.TestCase):
    def test_static_marker_and_latched_qos(self):
        publish, _ = load_publication()
        node = Mock()
        node.waypoints = MagicMock()
        node.waypoints.__getitem__.return_value = [(1., 2.), (-2., 5.)]
        node.get_parameter.return_value.value = 'custom_map'
        publish(node)
        message_type, topic, qos = node.create_publisher.call_args.args
        self.assertEqual(topic, '/mppi/waypoints')
        self.assertEqual(qos.depth, 1)
        self.assertEqual(qos.history, 'keep_last')
        self.assertEqual(qos.reliability, 'reliable')
        self.assertEqual(qos.durability, 'transient_local')
        node.waypoints_publisher.publish.assert_called_once()
        marker = node.waypoints_publisher.publish.call_args.args[0]
        self.assertEqual(marker.header.frame_id, 'custom_map')
        self.assertEqual(marker.header.stamp, node.get_clock().now().to_msg())
        self.assertEqual([(p.x, p.y, p.z) for p in marker.points],
                         [(1., 2., 0.), (-2., 5., 0.)])
        self.assertEqual(marker.type, message_type.POINTS)
        self.assertEqual(marker.action, message_type.ADD)
        self.assertEqual(marker.pose.orientation.w, 1.)
        self.assertGreater(marker.scale.x, 0.)
        self.assertGreater(marker.color.a, 0.)
        self.assertEqual((marker.lifetime.sec, marker.lifetime.nanosec), (0, 0))
        node.create_timer.assert_not_called()

    def test_publication_called_only_at_startup(self):
        _, node_class = load_publication()
        callers = [method.name for method in node_class.body
                   if isinstance(method, ast.FunctionDef)
                   for call in ast.walk(method)
                   if isinstance(call, ast.Call)
                   and isinstance(call.func, ast.Attribute)
                   and call.func.attr == '_publish_waypoints']
        self.assertEqual(callers, ['__init__'])


if __name__ == '__main__':
    unittest.main()
