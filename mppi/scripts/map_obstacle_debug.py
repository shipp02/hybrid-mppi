"""Host-side map perception owned by the controller node; visualization only."""
import numpy as np

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from map_obstacles import (ClusterConfig, MapMatcher, STATIC, UNMAPPED, UNCERTAIN,
                           cluster_scan, scan_points)


def rotation_matrix(q):
    """Normalize a ROS quaternion and return its full 3-D rotation matrix."""
    values = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid transform/map quaternion')
    x, y, z, w = values / norm
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class MapObstacleDebug:
    """Cache map and latest scan, classify once per scan, retry unavailable TF.

    Runs on a modest-rate host timer, never inside the compiled controller.
    Marker clusters are per-scan surfaces, not persistent object IDs or tracks.
    """

    def __init__(self, node):
        self.node = node
        self.matcher = None
        self.frame = ''
        self.scan = None
        self.scan_sequence = 0
        self.processed = None
        self.generation = 0
        self.allowance = self._parameter('map_match_allowance', 0.10)
        self.threshold = self._parameter('map_occupied_threshold', 65)
        self.max_age = self._parameter('obstacle_marker_max_scan_age', 0.5)
        if not np.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError('obstacle_marker_max_scan_age must be positive and finite')
        self.cfg = ClusterConfig(**{
            name: self._parameter('cluster_' + name, default)
            for name, default in vars(ClusterConfig()).items()})
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = node.create_subscription(OccupancyGrid, '/map', self._map, qos)
        self.publisher = node.create_publisher(MarkerArray, '/mppi/map_obstacles', 1)
        self.timer = node.create_timer(0.1, self._update)

    def _parameter(self, name, default):
        self.node.declare_parameter(name, default)
        return self.node.get_parameter(name).value

    def observe(self, msg):
        self.scan = msg
        self.scan_sequence += 1

    def _map(self, msg):
        try:
            self.matcher = self._build_matcher(msg)
            self.frame = msg.header.frame_id
        except (ValueError, TypeError) as exc:
            self.matcher = None
            self.node.get_logger().warning(f'Map classification unavailable: {exc}')
        self.generation += 1
        self._clear()

    def _build_matcher(self, msg):
        if not msg.header.frame_id:
            raise ValueError('map frame is empty')
        rotation = rotation_matrix(msg.info.origin.orientation)
        if not np.allclose(rotation[2], [0, 0, 1], atol=1e-5):
            raise ValueError('map origin must be planar')
        position = msg.info.origin.position
        origin = (position.x, position.y, np.arctan2(rotation[1, 0], rotation[0, 0]))
        cells = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        return MapMatcher(cells, msg.info.resolution, origin, self.threshold, self.allowance)

    def _update(self):
        if self.matcher is None or self.scan is None:
            return
        scan = self.scan
        stamp = Time.from_msg(scan.header.stamp)
        age = (self.node.get_clock().now() - stamp).nanoseconds * 1e-9
        if age > self.max_age or age < -0.1:
            self._clear()
            return
        key = (self.scan_sequence, self.generation)
        if key == self.processed:
            return
        try:
            points, valid = self._map_points(scan)
        except (TransformException, ValueError) as exc:
            self._clear()
            self.node.get_logger().warning(
                f'Map obstacle TF/classification unavailable: {exc}', throttle_duration_sec=2.0)
            return
        labels = self.matcher.classify(points)
        clusters = cluster_scan(points, np.asarray(scan.ranges), labels, valid,
                                scan.angle_increment, self.cfg)
        self._publish(scan.header.stamp, points, labels, valid, clusters)
        self.processed = key

    def _map_points(self, scan):
        # Latest TF avoids scan-time extrapolation at the cost of temporal alignment.
        transform = self.node.tf_buffer.lookup_transform(
            self.frame, scan.header.frame_id, Time())
        points, valid = scan_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                    scan.range_min, scan.range_max)
        rotation = rotation_matrix(transform.transform.rotation)
        translation = transform.transform.translation
        mapped = points @ rotation[:2, :2].T + [translation.x, translation.y]
        return mapped, valid

    def _clear(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.publisher.publish(MarkerArray(markers=[marker]))

    def _marker(self, stamp, namespace, identifier, kind, color):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = stamp
        marker.ns, marker.id, marker.type = namespace, identifier, kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = Duration(seconds=self.max_age).to_msg()
        return marker

    def _publish(self, stamp, points, labels, valid, clusters):
        reset = Marker()
        reset.action = Marker.DELETEALL
        markers = [reset]
        styles = ((STATIC, 'mapped', (0.2, 0.6, 1.0, 1.0)),
                  (UNMAPPED, 'unmapped', (1.0, 0.25, 0.1, 1.0)),
                  (UNCERTAIN, 'uncertain', (0.6, 0.6, 0.6, 1.0)))
        for label, name, color in styles:
            marker = self._marker(stamp, name, 0, Marker.POINTS, color)
            marker.scale.x = marker.scale.y = 0.06
            marker.points = [Point(x=float(x), y=float(y), z=0.08)
                             for x, y in points[valid & (labels == label)]]
            markers.append(marker)
        for identifier, indices in enumerate(clusters):
            markers.append(self._cluster_box(stamp, identifier, points[indices]))
        self.publisher.publish(MarkerArray(markers=markers))

    def _cluster_box(self, stamp, identifier, points):
        marker = self._marker(stamp, 'clusters', identifier, Marker.LINE_STRIP,
                              (1.0, 0.85, 0.0, 1.0))
        marker.scale.x = 0.025
        low, high = points.min(axis=0) - 0.04, points.max(axis=0) + 0.04
        marker.points = [Point(x=float(x), y=float(y), z=0.12) for x, y in
                         [(low[0], low[1]), (high[0], low[1]), (high[0], high[1]),
                          (low[0], high[1]), (low[0], low[1])]]
        return marker
