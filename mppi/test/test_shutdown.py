"""Stop-on-shutdown: the node leaves the car stopped, no ROS graph needed."""
import types
import unittest
from unittest.mock import Mock, patch

from test_control_step import controller as c


class Drive:
    """Stand-in for AckermannDriveStamped; the test loader strips ROS imports."""

    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None)
        self.drive = types.SimpleNamespace(speed=None, steering_angle=None)


class StopOnShutdownTest(unittest.TestCase):
    def setUp(self):
        import signal
        for name, value in (('rclpy', Mock()), ('signal', signal), ('AckermannDriveStamped', Drive),
                            ('SignalHandlerOptions', types.SimpleNamespace(NO='NO')),
                            ('ExternalShutdownException', type('ExternalShutdown', (Exception,), {}))):
            patcher = patch.object(c, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_node(self):
        node = object.__new__(c.MPPI)
        node.drive_publisher = Mock()
        node.get_clock = Mock()
        node._last_tracker_action = node._last_drive_command = None
        return node

    def test_stop_vehicle_publishes_zero_commands(self):
        node = self.make_node()
        with patch.object(c.rclpy, 'ok', return_value=True):
            node.stop_vehicle(repeats=3)
        sent = [call.args[0].drive for call in node.drive_publisher.publish.call_args_list]
        self.assertEqual(len(sent), 3)
        self.assertTrue(all(d.speed == 0.0 and d.steering_angle == 0.0 for d in sent))

    def test_stop_vehicle_skips_after_shutdown(self):
        node = self.make_node()
        with patch.object(c.rclpy, 'ok', return_value=False):
            node.stop_vehicle()
        node.drive_publisher.publish.assert_not_called()

    def test_main_stops_vehicle_before_shutdown(self):
        node = Mock()
        order = []
        node.stop_vehicle.side_effect = lambda: order.append('stop')
        node.destroy_node.side_effect = lambda: order.append('destroy')
        with patch.object(c, 'MPPI', return_value=node), \
                patch.object(c.rclpy, 'init') as init, \
                patch.object(c.rclpy, 'spin', side_effect=KeyboardInterrupt), \
                patch.object(c.rclpy, 'try_shutdown', side_effect=lambda: order.append('shutdown')), \
                patch.object(c.signal, 'signal') as handlers:
            c.main()
        self.assertEqual(order, ['stop', 'destroy', 'shutdown'])
        self.assertEqual(init.call_args.kwargs['signal_handler_options'],
                         c.SignalHandlerOptions.NO)
        self.assertEqual({call.args[0] for call in handlers.call_args_list},
                         {c.signal.SIGINT, c.signal.SIGTERM})


if __name__ == '__main__':
    unittest.main()
