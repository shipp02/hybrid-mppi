"""Bounded selected-path transport and off-process ROS marker publication.

Bounded XY and cost snapshots cross the process boundary. No JAX work belongs
in the publisher. Slow readers never block the control process.
"""
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np

HISTORY_SIZE = 10
PUBLISH_PERIOD = 0.5
TOPIC = '/mppi/selected_paths'
COST_NAMES = ('tracking', 'control', 'control_change', 'obstacle', 'envelope')


@dataclass(frozen=True)
class TopPathSnapshot:
    xy: np.ndarray
    totals: np.ndarray
    components: np.ndarray
    best_steps: np.ndarray
    stamp: int
    sequence: int


class SharedTopPaths:
    """Latest-only bounded diagnostics; same nonblocking writer contract as history."""

    def __init__(self, context, points, capacity=50):
        self.points = points
        self.capacity = capacity
        self._size = capacity * points * 2 + capacity * 6 + (points - 1) * 5
        self._data = context.RawArray('f', self._size)
        self._count = context.RawValue('i', 0)
        self._stamp = context.RawValue('q', 0)
        self._sequence = context.RawValue('Q', 0)
        self._lock = context.Lock()

    def try_record(self, paths, stamp_ns):
        count = len(paths.xy)
        if not 0 < count <= self.capacity or paths.xy.shape != (count, self.points, 2):
            return False
        components = np.stack([paths.components[name] for name in COST_NAMES], axis=1)
        steps = np.stack([paths.best_steps[name] for name in COST_NAMES], axis=1)
        data = np.concatenate((paths.xy.ravel(), paths.totals.ravel(),
                               components.ravel(), steps.ravel()))
        if steps.shape != (self.points - 1, 5) or not self._lock.acquire(False):
            return False
        try:
            np.frombuffer(self._data, dtype=np.float32)[:len(data)] = data
            self._count.value = count
            self._stamp.value = stamp_ns
            self._sequence.value += 1
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        with self._lock:
            data = np.frombuffer(self._data, dtype=np.float32).copy()
            count, stamp, sequence = self._count.value, self._stamp.value, self._sequence.value
        xy_end = count * self.points * 2
        totals_end = xy_end + count
        components_end = totals_end + count * 5
        return TopPathSnapshot(
            data[:xy_end].reshape(count, self.points, 2),
            data[xy_end:totals_end], data[totals_end:components_end].reshape(count, 5),
            data[components_end:components_end + (self.points - 1) * 5].reshape(-1, 5),
            stamp, sequence)


@dataclass(frozen=True)
class PathSnapshot:
    xy: np.ndarray
    stamps: np.ndarray
    sequence: int


class SharedPathHistory:
    """Single writer, single reader ring; rejected writes never wait for reader."""

    def __init__(self, context, points, capacity=HISTORY_SIZE):
        self.points = points
        self.capacity = capacity
        self._xy = context.RawArray('f', capacity * points * 2)
        self._stamps = context.RawArray('q', capacity)
        self._sequence = context.RawValue('Q', 0)
        self._lock = context.Lock()

    def try_record(self, xy, stamp_ns):
        """Copy one complete path, or drop it immediately on reader contention."""
        xy = np.asarray(xy, dtype=np.float32)
        if xy.shape != (self.points, 2) or not np.isfinite(xy).all():
            return False
        if not self._lock.acquire(False):
            return False
        try:
            slot = self._sequence.value % self.capacity
            np.frombuffer(self._xy, dtype=np.float32).reshape(self.capacity, self.points, 2)[slot] = xy
            np.frombuffer(self._stamps, dtype=np.int64)[slot] = stamp_ns
            self._sequence.value += 1
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        """Copy latest paths oldest-first; release lock before building messages."""
        with self._lock:
            sequence = self._sequence.value
            count = min(sequence, self.capacity)
            slots = np.arange(sequence - count, sequence, dtype=np.int64) % self.capacity
            xy = np.frombuffer(self._xy, dtype=np.float32).reshape(self.capacity, self.points, 2)[slots].copy()
            stamps = np.frombuffer(self._stamps, dtype=np.int64)[slots].copy()
        return PathSnapshot(xy, stamps, sequence)


@dataclass(frozen=True)
class CorridorSnapshot:
    balls: np.ndarray
    stamp: int
    sequence: int


class SharedCorridor:
    """Latest-only corridor snapshot; control writer never waits for renderer."""

    def __init__(self, context, points):
        self.points = points
        self._balls = context.RawArray('f', points * 3)
        self._stamp = context.RawValue('q', 0)
        self._sequence = context.RawValue('Q', 0)
        self._lock = context.Lock()

    def try_record(self, corridor, stamp_ns):
        centers, radii = (np.asarray(value, dtype=np.float32) for value in corridor)
        if centers.shape != (self.points, 2) or radii.shape != (self.points,):
            return False
        balls = np.column_stack((centers, radii))
        if not np.isfinite(balls).all() or not self._lock.acquire(False):
            return False
        try:
            np.frombuffer(self._balls, dtype=np.float32)[:] = balls.ravel()
            self._stamp.value = stamp_ns
            self._sequence.value += 1
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        with self._lock:
            return CorridorSnapshot(
                np.frombuffer(self._balls, dtype=np.float32).copy().reshape(self.points, 3),
                self._stamp.value, self._sequence.value)


def segment_geometry(xy):
    """LINE_LIST pairs with identical endpoint colors: solid red-to-blue segments."""
    segments = len(xy) - 1
    pairs = np.stack((xy[:-1], xy[1:]), axis=1).reshape((-1, 2))
    fraction = np.linspace(0., 1., segments, dtype=np.float32)
    colors = np.stack((1. - fraction, np.zeros(segments), fraction), axis=1)
    return pairs, np.repeat(colors, 2, axis=0)


def build_markers(snapshot, frame, capacity=HISTORY_SIZE):
    """Build self-contained history; stable IDs plus deletes tolerate dropped batches."""
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    markers = MarkerArray()
    for index, (xy, stamp) in enumerate(zip(snapshot.xy, snapshot.stamps)):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp.sec = int(stamp) // 1_000_000_000
        marker.header.stamp.nanosec = int(stamp) % 1_000_000_000
        marker.ns = 'selected_mppi_paths'
        marker.id = index
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.
        marker.scale.x = .025
        marker.lifetime.sec = 2
        pairs, colors = segment_geometry(xy)
        marker.points = [Point(x=float(x), y=float(y), z=0.) for x, y in pairs]
        alpha = .25 + .75 * (index + 1) / len(snapshot.xy)
        marker.color.a = alpha
        marker.colors = [ColorRGBA(r=float(r), g=float(g), b=float(b), a=alpha)
                         for r, g, b in colors]
        markers.markers.append(marker)
    for index in range(len(snapshot.xy), capacity):
        marker = Marker()
        marker.header.frame_id = frame
        marker.ns = 'selected_mppi_paths'
        marker.id = index
        marker.action = Marker.DELETE
        markers.markers.append(marker)
    return markers


def publisher_qos():
    """No delivery backlog, replay, or reliability requirement for debug output."""
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE)


class HistoryPublication:
    """Own snapshot acquisition and ROS work, exclusively in the child process."""

    def __init__(self, node, history, frame):
        from visualization_msgs.msg import MarkerArray

        self.history = history
        self.frame = frame
        self.publisher = node.create_publisher(MarkerArray, TOPIC, publisher_qos())
        self.timer = node.create_timer(PUBLISH_PERIOD, self.publish)

    def publish(self):
        snapshot = self.history.snapshot()
        if snapshot.sequence:
            self.publisher.publish(build_markers(snapshot, self.frame, self.history.capacity))


def debug_marker(frame, stamp, namespace, index, kind):
    from visualization_msgs.msg import Marker

    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp.sec = int(stamp) // 1_000_000_000
    marker.header.stamp.nanosec = int(stamp) % 1_000_000_000
    marker.ns, marker.id, marker.type = namespace, index, kind
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.
    marker.lifetime.sec = 2
    marker.color.a = .9
    return marker


def build_top_markers(snapshot, frame):
    """Rank-colored sampled paths, best sample first."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    result = MarkerArray()
    for rank, xy in enumerate(snapshot.xy):
        if not np.isfinite(xy).all():
            continue
        marker = debug_marker(frame, snapshot.stamp, 'top_paths', rank, Marker.LINE_STRIP)
        marker.scale.x = .025
        marker.color.r = float(rank / max(1, len(snapshot.xy) - 1))
        marker.color.g = 1. - marker.color.r
        marker.points = [Point(x=float(x), y=float(y), z=.05) for x, y in xy]
        result.markers.append(marker)
    return result


def build_corridor_markers(snapshot, frame):
    """Horizontal clearance rings; red points show invalid zero-radius stages."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    result = MarkerArray()
    angles = np.linspace(0., 2. * np.pi, 33)
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    for index, (x, y, radius) in enumerate(snapshot.balls):
        valid = radius > 0.
        kind = Marker.LINE_STRIP if valid else Marker.POINTS
        marker = debug_marker(frame, snapshot.stamp, 'mpc_corridor', index, kind)
        marker.scale.x = .015 if valid else .08
        marker.scale.y = .08
        marker.color.r = 0. if valid else 1.
        marker.color.g = .8 if valid else 0.
        marker.color.b = 1. if valid else 0.
        xy = np.array([x, y]) + radius * unit_circle if valid else np.array([[x, y]])
        marker.points = [Point(x=float(px), y=float(py), z=.08) for px, py in xy]
        result.markers.append(marker)
    return result


class CorridorPublication:
    """Render newest corridor at 2 Hz in the publisher process, never in control."""

    def __init__(self, node, shared, frame):
        from visualization_msgs.msg import MarkerArray

        self.shared, self.frame = shared, frame
        self.sequence = 0
        self.publisher = node.create_publisher(MarkerArray, '/mppi/corridor', publisher_qos())
        self.timer = node.create_timer(PUBLISH_PERIOD, self.publish)

    def publish(self):
        snapshot = self.shared.snapshot()
        if snapshot.sequence == self.sequence:
            return
        self.sequence = snapshot.sequence
        self.publisher.publish(build_corridor_markers(snapshot, self.frame))


class TopPathPublication:
    """Reuse child node/timer and latest-only transport for all debug ROS work."""

    def __init__(self, node, shared, frame):
        from std_msgs.msg import Float32MultiArray
        from visualization_msgs.msg import MarkerArray

        self.shared, self.frame = shared, frame
        self.sequence = 0
        self.paths = node.create_publisher(MarkerArray, '/mppi/top_paths', publisher_qos())
        self.costs = node.create_publisher(Float32MultiArray, '/mppi/best_path_costs', publisher_qos())
        self.timer = node.create_timer(PUBLISH_PERIOD, self.publish)

    def publish(self):
        from std_msgs.msg import Float32MultiArray

        snapshot = self.shared.snapshot()
        if snapshot.sequence == self.sequence:
            return
        self.sequence = snapshot.sequence
        self.paths.publish(build_top_markers(snapshot, self.frame))
        # COST_NAMES order, followed by total; preserve non-finite diagnostics.
        message = Float32MultiArray()
        message.data = [*map(float, snapshot.components[0]), float(snapshot.totals[0])]
        self.costs.publish(message)


class SharedOccupancyGrid:
    """Latest-only raw cells and matching world pose; writers never wait."""

    def __init__(self, context, settings):
        self.resolution = settings['resolution']
        self.middle = settings['middle']
        self.shape = (int(np.ceil(settings['height'] / self.resolution)),
                      int(np.ceil(settings['width'] / self.resolution)))
        self._cells = context.RawArray('B', int(np.prod(self.shape)))
        self._pose = context.RawArray('d', 3)
        self._stamp = context.RawValue('q', 0)
        self._sequence = context.RawValue('Q', 0)
        self._lock = context.Lock()

    def try_record(self, cells, pose, stamp_ns):
        if cells.shape != self.shape or not self._lock.acquire(False):
            return False
        try:
            np.frombuffer(self._cells, dtype=np.uint8)[:] = cells.ravel()
            np.frombuffer(self._pose, dtype=np.float64)[:] = pose[:3]
            self._stamp.value = stamp_ns
            self._sequence.value += 1
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        with self._lock:
            return (np.frombuffer(self._cells, dtype=np.uint8).copy().reshape(self.shape),
                    np.frombuffer(self._pose, dtype=np.float64).copy(),
                    self._stamp.value, self._sequence.value)


def build_occupancy_grid(cells, pose, stamp, shared, frame):
    """Transpose [forward, left] into ROS [y, x]; place origin in world frame."""
    from nav_msgs.msg import OccupancyGrid

    x, y, yaw = pose
    forward, left = -shared.middle[1], -shared.middle[0]
    message = OccupancyGrid()
    message.header.frame_id = frame
    message.header.stamp.sec = int(stamp) // 1_000_000_000
    message.header.stamp.nanosec = int(stamp) % 1_000_000_000
    message.info.resolution = shared.resolution
    message.info.width, message.info.height = cells.shape
    message.info.origin.position.x = float(x + np.cos(yaw) * forward - np.sin(yaw) * left)
    message.info.origin.position.y = float(y + np.sin(yaw) * forward + np.cos(yaw) * left)
    message.info.origin.position.z = 0.02  # above the map plane, which would hide it
    message.info.origin.orientation.z = float(np.sin(yaw / 2))
    message.info.origin.orientation.w = float(np.cos(yaw / 2))
    # Controller cell codes: UNKNOWN=1, FREE=2, OCCUPIED=3.
    message.data = np.array([-1, -1, 0, 100], dtype=np.int8)[cells.T].ravel().tolist()
    return message


class OccupancyGridPublication:
    def __init__(self, node, shared, frame):
        from nav_msgs.msg import OccupancyGrid

        self.shared, self.frame = shared, frame
        self.sequence = 0
        self.publisher = node.create_publisher(OccupancyGrid, '/mppi/occupancy_grid', publisher_qos())
        self.timer = node.create_timer(PUBLISH_PERIOD, self.publish)

    def publish(self):
        cells, pose, stamp, sequence = self.shared.snapshot()
        if sequence == self.sequence:
            return
        self.sequence = sequence
        self.publisher.publish(build_occupancy_grid(cells, pose, stamp, self.shared, self.frame))


def run_publisher(history, stop, frame, top_paths=None, publish_history=True, occupancy_grid=None,
                  corridor=None):
    """Spawn entry point: independent ROS node, no inherited CUDA context."""
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=[])
    node = Node('mppi_selected_path_publisher')
    publication = HistoryPublication(node, history, frame) if publish_history else None
    top_publication = TopPathPublication(node, top_paths, frame) if top_paths is not None else None
    grid_publication = OccupancyGridPublication(node, occupancy_grid, frame) if occupancy_grid is not None else None
    corridor_publication = CorridorPublication(node, corridor, frame) if corridor is not None else None
    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


class PathPublisherProcess:
    """Own spawned process and bounded shared history; call close on node shutdown."""

    def __init__(self, points, frame='map', *, publish_history=True, publish_top_paths=False,
                 grid_settings=None, publish_corridor=False):
        context = mp.get_context('spawn')
        self.history = SharedPathHistory(context, points)
        self.top_paths = SharedTopPaths(context, points) if publish_top_paths else None
        self.occupancy_grid = SharedOccupancyGrid(context, grid_settings) if grid_settings is not None else None
        self.corridor = SharedCorridor(context, points) if publish_corridor else None
        self._stop = context.Event()
        self._process = context.Process(target=run_publisher,
            args=(self.history, self._stop, frame, self.top_paths, publish_history,
                  self.occupancy_grid, self.corridor), daemon=True)
        self._process.start()

    def close(self):
        self._stop.set()
        self._process.join(timeout=2.)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.)
