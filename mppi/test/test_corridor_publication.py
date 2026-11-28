"""Bounded corridor transport and 2 Hz sidecar visualization without ROS."""
import multiprocessing as mp
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_control_step import controller as c
import selected_path_publisher as publication


class Point:
    def __init__(self, **values):
        self.__dict__.update(values)


class Marker:
    ADD, LINE_STRIP, POINTS = 0, 4, 8

    def __init__(self):
        self.header = types.SimpleNamespace(frame_id='', stamp=types.SimpleNamespace(sec=0, nanosec=0))
        self.pose = types.SimpleNamespace(orientation=types.SimpleNamespace(w=0.))
        self.scale = types.SimpleNamespace(x=0., y=0., z=0.)
        self.color = types.SimpleNamespace(r=0., g=0., b=0., a=0.)
        self.lifetime = types.SimpleNamespace(sec=0)


class MarkerArray:
    def __init__(self):
        self.markers = []


class CorridorPublicationTest(unittest.TestCase):
    def setUp(self):
        self.shared = publication.SharedCorridor(mp.get_context('spawn'), 2)
        self.corridor = np.array([[1., 2.], [3., 4.]]), np.array([.3, 0.])
        self.modules = {
            'geometry_msgs.msg': types.SimpleNamespace(Point=Point),
            'visualization_msgs.msg': types.SimpleNamespace(Marker=Marker, MarkerArray=MarkerArray),
        }

    def test_latest_snapshot_and_invalid_radius_are_preserved(self):
        self.assertTrue(self.shared.try_record(self.corridor, 123))
        self.assertTrue(self.shared.try_record(self.corridor, 456))
        snapshot = self.shared.snapshot()
        self.assertEqual((snapshot.sequence, snapshot.stamp), (2, 456))
        np.testing.assert_allclose(snapshot.balls[:, :2], self.corridor[0])
        np.testing.assert_allclose(snapshot.balls[:, 2], self.corridor[1])

    def test_contended_and_invalid_writes_drop_without_waiting(self):
        self.shared._lock.acquire()
        try:
            self.assertFalse(self.shared.try_record(self.corridor, 123))
        finally:
            self.shared._lock.release()
        self.assertFalse(self.shared.try_record((np.zeros((3, 2)), np.ones(3)), 123))
        self.assertFalse(self.shared.try_record((np.full((2, 2), np.nan), np.ones(2)), 123))
        self.assertEqual(self.shared.snapshot().sequence, 0)

    def test_markers_are_world_frame_closed_rings_and_red_invalid_points(self):
        self.shared.try_record(self.corridor, 1_234_567_890)
        with patch.dict('sys.modules', self.modules):
            markers = publication.build_corridor_markers(self.shared.snapshot(), 'map').markers
        circle, invalid = markers
        self.assertEqual(circle.type, Marker.LINE_STRIP)
        self.assertEqual((circle.header.frame_id, circle.header.stamp.sec,
                          circle.header.stamp.nanosec), ('map', 1, 234_567_890))
        xy = np.array([[point.x, point.y] for point in circle.points])
        np.testing.assert_allclose(np.linalg.norm(xy - [1., 2.], axis=1), .3, atol=1e-6)
        np.testing.assert_allclose(xy[0], xy[-1])
        self.assertEqual(invalid.type, Marker.POINTS)
        self.assertEqual(invalid.color.r, 1.)
        self.assertEqual(len(invalid.points), 1)
        self.assertEqual(circle.lifetime.sec, 2)

    def test_sidecar_timer_publishes_latest_snapshot_at_two_hz(self):
        node = Mock()
        with patch.dict('sys.modules', self.modules), \
                patch.object(publication, 'publisher_qos', return_value=object()):
            publisher = publication.CorridorPublication(node, self.shared, 'map')
            self.assertEqual(node.create_timer.call_args.args[0], .5)
            self.assertEqual(node.create_publisher.call_args.args[1], '/mppi/corridor')
            publisher.publish()
            node.create_publisher.return_value.publish.assert_not_called()
            self.shared.try_record(self.corridor, 123)
            publisher.publish()
            publisher.publish()
            node.create_publisher.return_value.publish.assert_called_once()

    def node(self):
        node = object.__new__(c.MPPI)
        node.publish_corridor_enabled, node.last_corridor_export = True, None
        node.path_publisher = Mock()
        node.path_publisher.corridor.try_record.return_value = True
        return node

    def test_controller_only_exports_at_two_hz(self):
        node = self.node()
        for stamp in (0, 100_000_000, 490_000_000, 500_000_000, 600_000_000, 1_000_000_000):
            node._publish_corridor(self.corridor, types.SimpleNamespace(nanoseconds=stamp))
        self.assertEqual(node.path_publisher.corridor.try_record.call_count, 3)

    def test_disabled_output_and_non_corridor_modes_do_nothing(self):
        node = self.node()
        node._publish_corridor(None, None)
        node.publish_corridor_enabled = False
        node._publish_corridor(self.corridor, None)
        node.path_publisher.corridor.try_record.assert_not_called()

    def test_dropped_write_can_retry_next_tick(self):
        node = self.node()
        node.path_publisher.corridor.try_record.side_effect = [False, True]
        node._publish_corridor(self.corridor, types.SimpleNamespace(nanoseconds=0))
        self.assertIsNone(node.last_corridor_export)
        node._publish_corridor(self.corridor, types.SimpleNamespace(nanoseconds=1))
        self.assertIsNotNone(node.last_corridor_export)


if __name__ == '__main__':
    unittest.main()
