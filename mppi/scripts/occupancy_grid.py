"""JAX version of LocalOccupancyGrid: pure functions that work under jit, vmap and scan.

Design rules that make this traceable:
  * Config is a frozen (hashable) dataclass passed as a *static* argument.
  * State is a NamedTuple of arrays (a pytree), returned by `update`, never mutated.
  * Every array shape is fixed by the config.
  * Exact and highly optimized distance transform built using a pre-computed relative stencil
    for local neighborhood distance evaluation.

Frame: +forward, +left. Arrays are indexed [row, column] = [forward, left].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

UNKNOWN, FREE, OCCUPIED = 1, 2, 3  # same values as CellState


@dataclass(frozen=True, unsafe_hash=True)
class GridConfig:
    width: float = 3.0
    height: float = 7.0
    middle: tuple[float, float] = (1.5, 1.0)
    resolution: float = 0.05
    inflation_radius: float = 0.30
    distance_cap: float = 1.0  # distances saturate here (and obstacles this far outside the grid count)
    angle_min: float = -2.35
    angle_max: float = 2.35
    num_beams: int = 1080
    block_unknown: bool = True
    start_radius: float = 0.42  # >= max |axis offset| + disc radius used by MPPI
    start_clearance: float = 0.10
    car_length: float = 0.58
    car_width: float = 0.31
    footprint_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0 or self.resolution <= 0.0:
            raise ValueError("width, height, and resolution must be positive")
        if self.inflation_radius < 0.0 or self.distance_cap <= 0.0:
            raise ValueError("inflation_radius must be >= 0 and distance_cap > 0")
        if self.angle_min >= self.angle_max or self.num_beams < 2:
            raise ValueError("need angle_min < angle_max and num_beams >= 2")
        if self.start_radius < 0.0 or self.start_clearance < 0.0:
            raise ValueError("start_radius and start_clearance must be non-negative")

    # -- static geometry (plain Python/NumPy, evaluated at trace time) ----------
    @property
    def shape(self) -> tuple[int, int]:
        return (int(np.ceil(self.height / self.resolution)), int(np.ceil(self.width / self.resolution)))

    @property
    def pad(self) -> int:
        return int(np.ceil(max(self.distance_cap, self.inflation_radius) / self.resolution))

    @property
    def max_ray(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def ray_step(self) -> float:
        return self.resolution / 2.0

    @property
    def n_ray_samples(self) -> int:
        return int(np.ceil(self.max_ray / self.ray_step)) + 1


class GridState(NamedTuple):
    cells: jax.Array       # (rows, cols) uint8 of UNKNOWN/FREE/OCCUPIED
    free_distance: jax.Array  # (rows, cols) metres to the nearest obstacle, unknown cell or grid edge


class GridPrecomputations:
    """Container holding precomputed static arrays for the grid configuration.
    Initialize this once outside your hot loop to avoid any runtime overhead or XLA recompilations.
    """
    def __init__(self, cfg: GridConfig, lidar_pose=(0.0, 0.0, 0.0)):
        self.cfg = cfg
        rows_n, cols_n = cfg.shape

        # 1. Constants & vehicle footprint mask
        inc = (cfg.angle_max - cfg.angle_min) / (cfg.num_beams - 1)
        angles = cfg.angle_min + np.arange(cfg.num_beams) * inc
        cx = (np.arange(cols_n) + 0.5) * cfg.resolution
        cy = (np.arange(rows_n) + 0.5) * cfg.resolution
        self.cos_angles = jnp.array(np.cos(angles))
        self.sin_angles = jnp.array(np.sin(angles))
        self.at_car = jnp.array(np.hypot(cx[None, :] - cfg.middle[0], cy[:, None] - cfg.middle[1]) <= cfg.start_radius)

        # 2. Fixed sensor geometry: inverse pose maps car-frame cells into LiDAR.
        # lidar_pose is [forward, left, yaw] in the car frame.
        forward = (np.arange(rows_n) + 0.5) * cfg.resolution - cfg.middle[1]
        left = (np.arange(cols_n) + 0.5) * cfg.resolution - cfg.middle[0]
        forward, left = np.meshgrid(forward, left, indexing="ij")
        sensor_forward, sensor_left, sensor_yaw = lidar_pose
        df, dl = forward - sensor_forward, left - sensor_left
        c, s = np.cos(sensor_yaw), np.sin(sensor_yaw)
        f, l = c * df + s * dl, -s * df + c * dl
        beam = np.rint((np.arctan2(l, f) - cfg.angle_min) / inc).astype(int)
        self.cell_range = jnp.array(np.hypot(f, l))
        self.in_fov = jnp.array((beam >= 0) & (beam < cfg.num_beams))
        self.beam = jnp.array(np.clip(beam, 0, cfg.num_beams - 1))

        # 3. Distance stencil offsets
        r = int(np.ceil(cfg.distance_cap / cfg.resolution))
        i, j = np.mgrid[-r:r + 1, -r:r + 1]
        dists = np.hypot(i, j) * cfg.resolution
        mask = dists <= (cfg.distance_cap + 1e-5)
        
        self.stencil_rows = jnp.array(i[mask])
        self.stencil_cols = jnp.array(j[mask])
        self.stencil_dists = jnp.array(dists[mask])

        # 4. Coordinate indices for distance evaluation
        y_coords, x_coords = jnp.indices((rows_n, cols_n))
        self.y_flat = y_coords.ravel()
        self.x_flat = x_coords.ravel()


@partial(jax.jit, static_argnums=(0, 1))
def update(cfg: GridConfig, precomps: GridPrecomputations, ranges) -> GridState:
    """Build a car-frame grid using precomputed, fixed LiDAR geometry."""
    ranges = jnp.asarray(ranges, dtype=jnp.result_type(float))
    if ranges.shape != (cfg.num_beams,):
        raise ValueError(f"expected ({cfg.num_beams},) LiDAR ranges, received {ranges.shape}")
    
    valid = jnp.isfinite(ranges) & (ranges > 0.0)
    safe = jnp.where(valid, ranges, 0.0)
    rows_n, cols_n = cfg.shape
    
    cells = jnp.full((rows_n, cols_n), UNKNOWN, dtype=jnp.uint8)

    beam = precomps.beam
    in_fov = precomps.in_fov
    cell_range = precomps.cell_range

    # Detect hit cells
    hit = in_fov & valid[beam] & (jnp.abs(cell_range - safe[beam]) < cfg.resolution)
    cells = jnp.where(hit, jnp.uint8(OCCUPIED), cells)

    # Detect free cells
    free = in_fov & valid[beam] & (cell_range < safe[beam] - cfg.resolution / 2.0)
    cells = jnp.where(free, jnp.uint8(FREE), cells)

    # Obstacles to run distance transform on
    obstacle_mask = (cells == OCCUPIED) | (cells == UNKNOWN)

    # Compute distance field via a localized gathering on each coordinate
    @jax.vmap
    def compute_cell_distance(r, c):
        nbr_r = jnp.clip(r + precomps.stencil_rows, 0, rows_n - 1)
        nbr_c = jnp.clip(c + precomps.stencil_cols, 0, cols_n - 1)

        has_obstacle = obstacle_mask[nbr_r, nbr_c]
        effective_dists = jnp.where(has_obstacle, precomps.stencil_dists, cfg.distance_cap)
        return jnp.min(effective_dists)

    free_distance = compute_cell_distance(precomps.y_flat, precomps.x_flat).reshape(rows_n, cols_n)
    return GridState(cells=cells, free_distance=free_distance)


def is_free_poses(cfg: GridConfig, state: GridState, pos: jax.Array) -> jax.Array:
    """Check car-frame poses [forward, left, yaw]; yaw is unused."""
    row = jnp.floor((pos[:, 0] + cfg.middle[1]) / cfg.resolution).astype(int)
    col = jnp.floor((pos[:, 1] + cfg.middle[0]) / cfg.resolution).astype(int)

    rows_n, cols_n = cfg.shape
    row = jnp.clip(row, 0, rows_n - 1)
    col = jnp.clip(col, 0, cols_n - 1)

    is_cell_free = (state.cells[row, col] == FREE)
    is_safe_distance = (state.free_distance[row, col] >= cfg.inflation_radius)
    return is_cell_free & is_safe_distance


def find_distances(cfg: GridConfig, state: GridState, pos: jax.Array) -> jax.Array:
    """Return clearance at car-frame poses [forward, left, yaw]; yaw is unused."""
    row = jnp.floor((pos[:, 0] + cfg.middle[1]) / cfg.resolution).astype(int)
    col = jnp.floor((pos[:, 1] + cfg.middle[0]) / cfg.resolution).astype(int)

    rows_n, cols_n = cfg.shape
    row = jnp.clip(row, 0, rows_n - 1)
    col = jnp.clip(col, 0, cols_n - 1)
    return state.free_distance[row, col]

