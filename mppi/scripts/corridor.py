"""Collision-free corridor around the MPPI plan, and the MPC's race-line target.

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

# Stages the grid cannot vouch for (unknown cells under the car, past the sensor)
# get this radius: effectively no constraint, matching the tracker's inactive ball.
UNCONSTRAINED_RADIUS = 10.0


def to_car_frame(xy, pose):
    """World [x, y] -> car [forward, left] at pose [x, y, yaw]."""
    c, s = jnp.cos(pose[2]), jnp.sin(pose[2])
    d = xy - pose[:2]
    return jnp.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), axis=-1)


@partial(jax.jit, static_argnames=('cfg',))
def plan_corridor(nominal, scan_pose, grid, cfg):
    """Ball centers ((H+1, 2), world) and radii ((H+1,)) along the MPPI plan.

    scan_pose is the measured pose the grid was built at. A stage whose plan
    point the grid cannot confirm as free gets an unconstrained ball, so the
    blind spot under the car never drags the MPC sideways.
    """
    clearance = grid.clearance(to_car_frame(nominal[:, :2], scan_pose))
    room = clearance - grid.spec.inflation_radius
    radii = jnp.where(room > 0.0, jnp.minimum(room, cfg.CORRIDOR_R_MAX), UNCONSTRAINED_RADIUS)
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
