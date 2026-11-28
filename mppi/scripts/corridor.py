"""Clearance-based corridor around the MPPI plan, and the MPC's race-line target.

Hybrid MPPI-MPC after MPPI-IPDDP (Kim et al., arXiv 2208.02439): MPPI gives a
coarse plan, a corridor of balls B_r(c) around it bounds where the smoothing MPC
may go, and the MPC tracks the task (the race line) inside it.

The paper inflates each ball by sampling (eq. 27). Here each ball keeps its center
on the plan, c_t = p_t, and takes the largest collision-free radius directly from
the LiDAR grid's clearance:

    r_t = min(r_max, clearance(p_t) - car radius)

the paper's solution when nothing pulls the center sideways. It is deterministic,
so the corridor does not jitter from one tick to the next.
"""
from functools import partial

import jax
import jax.numpy as jnp

def to_car_frame(xy, pose):
    """World [x, y] -> car [forward, left] at pose [x, y, yaw]."""
    c, s = jnp.cos(pose[2]), jnp.sin(pose[2])
    d = xy - pose[:2]
    return jnp.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), axis=-1)


def corridor_centers_are_usable(local, grid):
    """Require in-grid free centers instead of trusting clipped edge queries."""
    row = (local[:, 0] + grid.spec.middle[1]) / grid.spec.resolution
    col = (local[:, 1] + grid.spec.middle[0]) / grid.spec.resolution
    rows, cols = grid.spec.shape
    in_grid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return in_grid & grid.is_free(local)


@partial(jax.jit, static_argnames=('cfg',))
def plan_corridor(nominal, scan_pose, grid, cfg):
    """Ball centers ((H+1, 2), world) and radii ((H+1,)) along the MPPI plan.

    scan_pose is the measured pose the grid was built at. Blocked, zero-clearance,
    and out-of-grid centers get radius zero: an invalid-corridor sentinel rejected
    by the host tracker before solving. Unknown cells follow the grid's occupancy
    policy; a clearance ball is not a certificate of observed free space.
    """
    local = to_car_frame(nominal[:, :2], scan_pose)
    clearance = grid.clearance(local)
    room = clearance - grid.spec.inflation_radius
    valid = corridor_centers_are_usable(local, grid) & (room > 0.0)
    radii = jnp.where(valid, jnp.minimum(room, cfg.CORRIDOR_R_MAX), 0.0)
    return nominal[:, :2], radii


@jax.jit
def line_reference_along_plan(nominal, spline):
    """MPC target timed by the MPPI plan: [x, y, yaw] of the race-line point
    nearest each plan state, and the plan's own speed.

    A time-indexed spline reference runs ahead of (or behind) the plan, so its
    targets would sit outside the balls of the same stage and fight them.
    """
    import quintic_spline as qs
    s = jax.vmap(lambda p: qs.project_onto_spline(spline, p))(nominal[:, :2])
    position, yaw = qs.evaluate_pose(spline, s)
    return jnp.column_stack((position, yaw, nominal[:, 3]))
