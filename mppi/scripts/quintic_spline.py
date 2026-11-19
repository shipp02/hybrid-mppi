"""Closed quintic spline reference: fit once on the host, sample on device."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class QuinticSpline(NamedTuple):
    knots: jax.Array   # (N+1,) arc length at each waypoint, knots[-1] = lap length
    coeffs: jax.Array  # (N, 6, 2) per-segment x/y coefficients in local arc length
    speed: jax.Array   # (N,) waypoint speeds


def _open_loop(points):
    """Drop a repeated final waypoint so the loop closes exactly once."""
    return points[:-1] if np.allclose(points[0, :2], points[-1, :2]) else points


def _hermite_coefficients(p0, v0, a0, p1, v1, a1, h):
    """Quintic matching position/slope/curvature at both ends of a segment."""
    hh = h[:, None]
    c2 = a0 / 2
    # end conditions on c3..c5 once c0..c2 are fixed by the start
    residual = np.stack((p1 - (p0 + v0 * hh + c2 * hh ** 2),
                         v1 - (v0 + 2 * c2 * hh),
                         a1 - 2 * c2), axis=1)  # (N, 3, D)
    system = np.stack((
        np.stack((h ** 3, h ** 4, h ** 5), axis=1),
        np.stack((3 * h ** 2, 4 * h ** 3, 5 * h ** 4), axis=1),
        np.stack((6 * h, 12 * h ** 2, 20 * h ** 3), axis=1)), axis=1)  # (N, 3, 3)
    high = np.linalg.solve(system, residual)
    return np.concatenate((np.stack((p0, v0, c2), axis=1), high), axis=1)


def fit_quintic_spline(waypoints):
    """Periodic C4 quintic through (x, y), parameterized by chord length."""
    from scipy.interpolate import make_interp_spline

    points = _open_loop(np.asarray(waypoints, dtype=np.float64))
    closed = np.vstack((points[:, :2], points[:1, :2]))
    h = np.hypot(*np.diff(closed, axis=0).T)
    if np.any(h <= 0):
        raise ValueError('quintic spline needs distinct consecutive waypoints')
    knots = np.concatenate(([0.0], np.cumsum(h)))
    spline = make_interp_spline(knots, closed, k=5, bc_type='periodic')
    p, v, a = (spline(knots, nu) for nu in range(3))
    coeffs = _hermite_coefficients(p[:-1], v[:-1], a[:-1], p[1:], v[1:], a[1:], h)
    return QuinticSpline(knots.astype(np.float32), coeffs.astype(np.float32),
                         points[:, -1].astype(np.float32))


def _locate(spline, s):
    """Segment index and local arc length of arc-length positions s."""
    s = s % spline.knots[-1]
    segment = jnp.clip(jnp.searchsorted(spline.knots, s, side='right') - 1,
                       0, spline.coeffs.shape[0] - 1)
    return segment, s - spline.knots[segment]


def evaluate_pose(spline, s):
    """Position and tangent heading on the spline at arc lengths s."""
    segment, tau = _locate(spline, s)
    c = spline.coeffs[segment]  # (..., 6, 2)
    powers = tau[..., None] ** jnp.arange(6)
    position = jnp.einsum('...k,...kd->...d', powers, c)
    slope_powers = jnp.arange(1, 6) * tau[..., None] ** jnp.arange(5)
    tangent = jnp.einsum('...k,...kd->...d', slope_powers, c[..., 1:, :])
    return position, jnp.arctan2(tangent[..., 1], tangent[..., 0])


def project_onto_spline(spline, point):
    """Arc length of the point on the waypoint polygon closest to `point`."""
    start = spline.coeffs[:, 0, :]
    chord = jnp.roll(start, -1, axis=0) - start
    t = jnp.clip(jnp.sum((point - start) * chord, axis=1)
                 / jnp.sum(chord ** 2, axis=1), 0.0, 1.0)
    distance = jnp.linalg.norm(start + t[:, None] * chord - point, axis=1)
    nearest = jnp.argmin(distance)
    return spline.knots[nearest] + t[nearest] * jnp.diff(spline.knots)[nearest]


def _speed_at(spline, s):
    """Waypoint speed, linearly interpolated so the profile never overshoots."""
    return jnp.interp(s, spline.knots[:-1], spline.speed, period=spline.knots[-1])


def _advance_along(spline, s0, dt, H, v0, max_accel):
    """Arc lengths and speeds of H steps of dt, ramping from v0 at max_accel."""
    def limit(speed, i):
        if v0 is None or max_accel is None:
            return speed
        return jnp.minimum(speed, v0 + max_accel * i * dt)

    speed0 = limit(_speed_at(spline, s0), 0)

    def advance(carry, i):
        s, speed = carry
        s = s + speed * dt
        speed = limit(_speed_at(spline, s), i)
        return (s, speed), (s, speed)

    _, (s, speeds) = jax.lax.scan(advance, (s0, speed0), jnp.arange(1, H + 1))
    return jnp.concatenate((s0[None], s)), jnp.concatenate((speed0[None], speeds))


@partial(jax.jit, static_argnames=('dt', 'H'))
def sample_quintic_reference(x, y, spline, dt, H, v0=None, max_accel=None):
    """Reference [x, y, yaw, v] for H steps of dt, sampled from the spline."""
    s0 = project_onto_spline(spline, jnp.array([x, y]))
    s, speeds = _advance_along(spline, s0, dt, H, v0, max_accel)
    position, yaw = evaluate_pose(spline, s)
    return jnp.column_stack((position, yaw, speeds))
