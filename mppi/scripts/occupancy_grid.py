from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# --- 1. Constants & Configuration ---

L_PRIOR = 0.0
L_OCC = 0.85
L_FREE = -0.4
L_MIN, L_MAX = -4.0, 4.0          # clamp so stale evidence can always be overwritten
OCCUPIED_THRESHOLD = 0.5
UNKNOWN_IS_OBSTACLE = True # True = conservative: unobserved cells count as blocked

UNKNOWN, FREE, OCCUPIED = 1, 2, 3  # same values as CellState


@dataclass(frozen=True)
class GridConfig:
    width: float = 3.0
    height: float = 7.0
    middle: tuple[float, float] = (1.5, 1.0)   # (left, forward) offset of the robot in the window
    resolution: float = 0.05
    inflation_radius: float = 0.30
    distance_cap: float = 1.0
    angle_min: float = -2.35
    angle_max: float = 2.35
    num_beams: int = 180
    lidar_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0 or self.resolution <= 0.0:
            raise ValueError("width, height, and resolution must be positive")
        if self.inflation_radius < 0.0 or self.distance_cap <= 0.0:
            raise ValueError("inflation_radius must be >= 0 and distance_cap > 0")
        if self.angle_min >= self.angle_max or self.num_beams < 2:
            raise ValueError("need angle_min < angle_max and num_beams >= 2")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(np.ceil(self.height / self.resolution)), int(np.ceil(self.width / self.resolution)))

    @property
    def max_ray(self) -> float:
        return float(np.hypot(self.width, self.height))


class GridState(NamedTuple):
    log_odds: jax.Array
    free_distance: jax.Array


# --- 2. Pure NumPy Cached Helpers ---

@functools.lru_cache(maxsize=None)
def _distance_stencil(cfg: GridConfig):
    r = int(np.ceil(cfg.distance_cap / cfg.resolution))
    i, j = np.mgrid[-r:r + 1, -r:r + 1]
    dists = np.hypot(i, j) * cfg.resolution
    mask = dists <= (cfg.distance_cap + 1e-5)
    return i[mask], j[mask], dists[mask]


@functools.lru_cache(maxsize=None)
def _polar(cfg: GridConfig):
    rows_n, cols_n = cfg.shape
    forward = (np.arange(rows_n) + 0.5) * cfg.resolution - cfg.middle[1]
    left = (np.arange(cols_n) + 0.5) * cfg.resolution - cfg.middle[0]
    f, l = np.meshgrid(forward, left, indexing="ij")
    sensor_forward, sensor_left, sensor_yaw = cfg.lidar_pose
    df, dl = f - sensor_forward, l - sensor_left
    c, s = np.cos(sensor_yaw), np.sin(sensor_yaw)
    f, l = c * df + s * dl, -s * df + c * dl
    inc = (cfg.angle_max - cfg.angle_min) / (cfg.num_beams - 1)
    beam = np.rint((np.arctan2(l, f) - cfg.angle_min) / inc).astype(int)
    in_fov = (beam >= 0) & (beam < cfg.num_beams)
    return np.hypot(f, l), np.clip(beam, 0, cfg.num_beams - 1), in_fov


# --- 3. JIT-Compiled Update Function ---

@functools.partial(jax.jit, static_argnums=0)
def update_grid(cfg: GridConfig, rolled_log_odds: jax.Array, ranges: jax.Array) -> GridState:
    rows_n, cols_n = cfg.shape

    cell_range_np, beam_np, in_fov_np = _polar(cfg)
    cell_range = jnp.asarray(cell_range_np)
    beam = jnp.asarray(beam_np)
    in_fov = jnp.asarray(in_fov_np)

    ranges = jnp.asarray(ranges, dtype=jnp.result_type(float))
    valid = jnp.isfinite(ranges) & (ranges > 0.0)
    safe = jnp.where(valid, ranges, 0.0)

    hit = in_fov & valid[beam] & (jnp.abs(cell_range - safe[beam]) < cfg.resolution)
    free = in_fov & valid[beam] & (cell_range < safe[beam] - cfg.resolution / 2.0)

    sensor_delta = jnp.where(hit, L_OCC - L_PRIOR, 0.0)
    sensor_delta = jnp.where(free, L_FREE - L_PRIOR, sensor_delta)

    # Bayesian fusion, clamped so cells can always change their minds
    new_log_odds = jnp.clip(rolled_log_odds + sensor_delta, L_MIN, L_MAX)

    obstacle_mask = new_log_odds >= OCCUPIED_THRESHOLD
    if UNKNOWN_IS_OBSTACLE:
        obstacle_mask = obstacle_mask | (new_log_odds == L_PRIOR)

    s_rows, s_cols, s_dists = _distance_stencil(cfg)
    stencil_rows = jnp.asarray(s_rows)
    stencil_cols = jnp.asarray(s_cols)
    stencil_dists = jnp.asarray(s_dists)

    y_coords, x_coords = jnp.indices((rows_n, cols_n))

    @jax.vmap
    def compute_cell_distance(r, c):
        nbr_r = jnp.clip(r + stencil_rows, 0, rows_n - 1)
        nbr_c = jnp.clip(c + stencil_cols, 0, cols_n - 1)
        has_obstacle = obstacle_mask[nbr_r, nbr_c]
        effective_dists = jnp.where(has_obstacle, stencil_dists, cfg.distance_cap)
        return jnp.min(effective_dists)

    free_distance = compute_cell_distance(y_coords.ravel(), x_coords.ravel()).reshape(rows_n, cols_n)
    return GridState(log_odds=new_log_odds, free_distance=free_distance)


WARP_METHOD = "scipy"   # "bilinear" (manual), "nearest" (manual), or "scipy" (jax.scipy.ndimage)


@functools.lru_cache(maxsize=None)
def _cell_centers(cfg: GridConfig):
    """Cell-centre coordinates (forward, left) in the robot frame, as NumPy arrays."""
    rows_n, cols_n = cfg.shape
    forward = (np.arange(rows_n) + 0.5) * cfg.resolution - cfg.middle[1]
    left = (np.arange(cols_n) + 0.5) * cfg.resolution - cfg.middle[0]
    return np.meshgrid(forward, left, indexing="ij")


@functools.partial(jax.jit, static_argnums=(0, 5))
def warp_grid(cfg: GridConfig, log_odds, t_fwd, t_left, dyaw, method: str = "bilinear"):
    """
    Re-express the grid in the new robot frame (inverse warp).

    (t_fwd, t_left, dyaw) is the NEW pose expressed in the OLD robot frame.
    For every new cell centre p_new we find where it was in the old grid:
        p_old = R(dyaw) @ p_new + t
    and sample the old grid there. Cells that fall outside the old grid become unknown.
    All three motion inputs are traced values, so arbitrary motion never re-triggers compilation.
    """
    rows_n, cols_n = cfg.shape
    f_new, l_new = (jnp.asarray(a) for a in _cell_centers(cfg))
    c, s = jnp.cos(dyaw), jnp.sin(dyaw)
    f_old = c * f_new - s * l_new + t_fwd
    l_old = s * f_new + c * l_new + t_left

    # metres -> fractional (row, col) index into the old grid
    r = (f_old + cfg.middle[1]) / cfg.resolution - 0.5
    q = (l_old + cfg.middle[0]) / cfg.resolution - 0.5

    if method == "scipy":
        return jax.scipy.ndimage.map_coordinates(log_odds, [r, q], order=1, mode="constant", cval=L_PRIOR)

    def at(ri, ci):
        ok = (ri >= 0) & (ri < rows_n) & (ci >= 0) & (ci < cols_n)
        return jnp.where(ok, log_odds[jnp.clip(ri, 0, rows_n - 1), jnp.clip(ci, 0, cols_n - 1)], L_PRIOR)

    if method == "nearest":
        return at(jnp.rint(r).astype(jnp.int32), jnp.rint(q).astype(jnp.int32))

    r0, c0 = jnp.floor(r), jnp.floor(q)
    fr, fc = r - r0, q - c0
    r0, c0 = r0.astype(jnp.int32), c0.astype(jnp.int32)
    return ((1 - fr) * (1 - fc) * at(r0, c0) + (1 - fr) * fc * at(r0, c0 + 1)
            + fr * (1 - fc) * at(r0 + 1, c0) + fr * fc * at(r0 + 1, c0 + 1))


def relative_pose(prev_pos, prev_yaw, pos, yaw):
    """New pose expressed in the previous robot frame (float64 NumPy, so no precision loss)."""
    dx, dy = pos[0] - prev_pos[0], pos[1] - prev_pos[1]
    t_fwd = np.cos(prev_yaw) * dx + np.sin(prev_yaw) * dy
    t_left = -np.sin(prev_yaw) * dx + np.cos(prev_yaw) * dy
    d = yaw - prev_yaw
    return t_fwd, t_left, float(np.arctan2(np.sin(d), np.cos(d)))


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("log_odds", "free_distance", "pose", "has_pose"),
    meta_fields=("spec",),
)
@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """Immutable controller-facing grid; arrays remain dynamic PyTree leaves.

    Frame: +forward, +left. Arrays use [forward, left] indexing.
    Supply pose to from_scan() to motion-compensate accumulated evidence.
    """

    spec: GridConfig
    log_odds: jax.Array
    free_distance: jax.Array
    pose: jax.Array
    has_pose: jax.Array

    @classmethod
    def create(cls, *, lidar_pose=(0.0, 0.0, 0.0), **settings):
        """Prepare fixed sensor geometry and initialize an unknown grid."""
        spec = GridConfig(lidar_pose=tuple(lidar_pose), **settings)
        return cls(
            spec=spec,
            log_odds=jnp.full(spec.shape, L_PRIOR, dtype=jnp.float32),
            free_distance=jnp.full(spec.shape, spec.distance_cap, dtype=jnp.float32),
            pose=jnp.zeros(3, dtype=jnp.float32),
            has_pose=jnp.bool_(False),
        )

    @property
    def cells(self):
        """Expose replay-compatible categories for ROS publication."""
        cells = jnp.full(self.spec.shape, UNKNOWN, dtype=jnp.uint8)
        cells = jnp.where(self.log_odds < L_PRIOR, jnp.uint8(FREE), cells)
        return jnp.where(self.log_odds >= OCCUPIED_THRESHOLD, jnp.uint8(OCCUPIED), cells)

    def _prior_for_scan(self, pose, reset):
        """Align previous evidence with the current vehicle frame."""
        if pose is None:
            return jnp.where(reset, L_PRIOR, self.log_odds)
        dx, dy = pose[0] - self.pose[0], pose[1] - self.pose[1]
        c, s = jnp.cos(self.pose[2]), jnp.sin(self.pose[2])
        rolled = warp_grid(
            self.spec, self.log_odds, c * dx + s * dy, -s * dx + c * dy,
            pose[2] - self.pose[2], method=WARP_METHOD)
        return jnp.where(reset | ~self.has_pose, L_PRIOR, rolled)

    @jax.jit
    def from_scan(self, ranges, range_max=jnp.inf, *, pose=None, reset=False):
        """Fuse a scan, optionally warping history or resetting after a pose jump.

        range_max is retained for controller API compatibility; scan validity
        and no-return behavior belong to update_grid().
        """
        ranges = jnp.asarray(ranges, dtype=jnp.result_type(float))
        if ranges.shape != (self.spec.num_beams,):
            raise ValueError(f"expected ({self.spec.num_beams},) LiDAR ranges, received {ranges.shape}")
        prior = self._prior_for_scan(pose, reset)
        state = update_grid(self.spec, prior, ranges)
        return replace(
            self, log_odds=state.log_odds, free_distance=state.free_distance,
            pose=self.pose if pose is None else jnp.asarray(pose),
            has_pose=jnp.bool_(pose is not None),
        )

    def _indices(self, pos):
        """Map batched vehicle-frame poses to nearest in-grid cells."""
        row = jnp.floor((pos[:, 0] + self.spec.middle[1]) / self.spec.resolution).astype(int)
        col = jnp.floor((pos[:, 1] + self.spec.middle[0]) / self.spec.resolution).astype(int)
        rows_n, cols_n = self.spec.shape
        return jnp.clip(row, 0, rows_n - 1), jnp.clip(col, 0, cols_n - 1)

    def is_free(self, pos: jax.Array) -> jax.Array:
        """Check occupancy and inflated clearance; out-of-grid queries clip."""
        row, col = self._indices(pos)
        values = self.log_odds[row, col]
        blocked = values >= OCCUPIED_THRESHOLD
        if UNKNOWN_IS_OBSTACLE:
            blocked = blocked | (values == L_PRIOR)
        return ~blocked & (self.free_distance[row, col] >= self.spec.inflation_radius)

    def clearance(self, pos: jax.Array) -> jax.Array:
        """Query capped clearance; out-of-grid poses use nearest edge cell."""
        row, col = self._indices(pos)
        return self.free_distance[row, col]

