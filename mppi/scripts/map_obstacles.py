"""Static-map classification and scan-order clustering; no ROS or JAX dependency."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

UNCERTAIN, STATIC, UNMAPPED = 0, 1, 2


@dataclass(frozen=True)
class ClusterConfig:
    base_gap: float = 0.08
    spacing_scale: float = 2.0
    max_link_distance: float = 0.30
    max_invalid_gap: int = 2
    max_angle_gap: float = 0.035
    min_points: int = 3
    bridge_nan: bool = True
    bridge_inf: bool = False

    def __post_init__(self):
        values = (self.base_gap, self.spacing_scale, self.max_link_distance,
                  self.max_angle_gap)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('cluster distances, angle and spacing scale must be positive and finite')
        if self.max_invalid_gap < 0 or self.min_points < 1:
            raise ValueError('max_invalid_gap must be >= 0 and min_points >= 1')


class MapMatcher:
    """Cache distances to occupied cell centres in a possibly rotated map."""

    def __init__(self, cells, resolution, origin, occupied_threshold=65, allowance=0.10):
        cells = np.asarray(cells)
        if cells.ndim != 2 or cells.size == 0:
            raise ValueError('map must be a nonempty 2-D array')
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('map resolution must be positive and finite')
        if not np.isfinite(allowance) or allowance < 0:
            raise ValueError('map match allowance must be nonnegative and finite')
        if not 1 <= occupied_threshold <= 100:
            raise ValueError('occupied threshold must be in [1, 100]')
        if not np.isfinite(origin).all():
            raise ValueError('map origin must be finite')
        self.cells = cells.copy()
        self.resolution = resolution
        self.origin = np.asarray(origin, dtype=float)
        self.threshold = occupied_threshold
        self.tolerance = allowance + resolution / np.sqrt(2.0)
        occupied = cells >= occupied_threshold
        self.distance = (distance_transform_edt(~occupied) * resolution if occupied.any()
                         else np.full(cells.shape, np.inf))

    def classify(self, points):
        """Classify world-frame points; unknown/outside endpoints remain uncertain."""
        points = np.asarray(points)
        labels = np.full(len(points), UNCERTAIN, dtype=np.uint8)
        finite = np.isfinite(points).all(axis=1)
        indices = np.flatnonzero(finite)
        delta = points[finite] - self.origin[:2]
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        local = delta @ np.array([[c, -s], [s, c]])
        col, row = np.floor(local / self.resolution).astype(np.int64).T
        inside = ((row >= 0) & (row < self.cells.shape[0]) &
                  (col >= 0) & (col < self.cells.shape[1]))
        indices, row, col = indices[inside], row[inside], col[inside]
        matched = self.distance[row, col] <= self.tolerance
        known_free = (self.cells[row, col] >= 0) & (self.cells[row, col] < self.threshold)
        labels[indices[known_free & ~matched]] = UNMAPPED
        labels[indices[matched]] = STATIC
        return labels


def scan_points(ranges, angle_min, angle_increment, range_min, range_max):
    """Return sensor-frame endpoints and validity; invalid beams keep their indices."""
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(len(ranges)) * angle_increment
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max) & (ranges > 0)
    safe = np.where(valid, ranges, np.nan)
    return np.column_stack((safe * np.cos(angles), safe * np.sin(angles))), valid


def cluster_scan(points, ranges, labels, valid, angle_increment, cfg):
    """Group unmapped hits; bridge only bounded, explicitly allowed invalid runs.

    Returns arrays of original beam indices. Never interpolates missing points.
    Valid non-unmapped hits and invalid finite/out-of-range returns always split.
    """
    clusters, current = [], []
    for i in range(len(ranges)):
        eligible = valid[i] and labels[i] == UNMAPPED
        bridgeable = (not valid[i] and
                      ((cfg.bridge_nan and np.isnan(ranges[i])) or
                       (cfg.bridge_inf and np.isposinf(ranges[i]))))
        if not eligible:
            if not bridgeable:
                _finish_cluster(clusters, current, cfg.min_points)
                current = []
            continue
        if current and not _can_link(current[-1], i, points, ranges, angle_increment, cfg):
            _finish_cluster(clusters, current, cfg.min_points)
            current = []
        current.append(i)
    _finish_cluster(clusters, current, cfg.min_points)
    return clusters


def _can_link(previous, current, points, ranges, increment, cfg):
    skipped = current - previous - 1
    angle = abs(increment) * (current - previous)
    spacing = 2 * min(ranges[previous], ranges[current]) * np.sin(angle / 2)
    limit = min(cfg.max_link_distance, cfg.base_gap + cfg.spacing_scale * spacing)
    return (skipped <= cfg.max_invalid_gap and angle <= cfg.max_angle_gap and
            np.linalg.norm(points[current] - points[previous]) <= limit)


def _finish_cluster(clusters, current, min_points):
    if len(current) >= min_points:
        clusters.append(np.asarray(current, dtype=int))
