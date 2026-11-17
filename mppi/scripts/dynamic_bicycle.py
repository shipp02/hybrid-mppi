"""Pure JAX forward-driving bicycle with bounded tire forces.

State: [COM x, COM y, body yaw, body vx, body vy, yaw rate,
        previous steering command, previous speed setpoint].
Input: [speed-setpoint slew (m/s^2), steering command (rad)].
Steering command slew is modeled, but unmeasured servo lag is not.
"""
import jax
import jax.numpy as jnp


def project_controls(U, x0, cfg):
    """Make each command reachable from the preceding published command."""
    def advance(previous, desired):
        speed, steering = previous
        next_speed = jnp.clip(speed + cfg.DT * desired[0], 0., cfg.MAX_SPEED)
        next_steering = jnp.clip(desired[1],
                                 steering - cfg.MAX_STEER_RATE * cfg.DT,
                                 steering + cfg.MAX_STEER_RATE * cfg.DT)
        next_steering = jnp.clip(next_steering, cfg.MIN_STEER, cfg.MAX_STEER)
        action = jnp.array([(next_speed - speed) / cfg.DT, next_steering])
        return (next_speed, next_steering), action

    bounded = jnp.clip(U, jnp.array([cfg.MIN_ACCEL, cfg.MIN_STEER]),
                       jnp.array([cfg.MAX_ACCEL, cfg.MAX_STEER]))
    _, commands = jax.lax.scan(advance, (x0[7], x0[6]), bounded)
    return commands


def tire_forces(state, command_speed, steering, cfg):
    """Allocate longitudinal demand and saturate each axle's combined grip."""
    vx, vy, rate = state[3:6]
    lf, lr = cfg.FRONT_LENGTH, cfg.WB - cfg.FRONT_LENGTH
    normal = cfg.MASS * 9.81 * jnp.array([lr, lf]) / cfg.WB
    capacity = cfg.TIRE_MU * normal
    accel = jnp.clip((command_speed - vx) / cfg.SPEED_RESPONSE_TIME,
                     cfg.MIN_ACCEL, cfg.MAX_ACCEL)
    split = jnp.array([cfg.FRONT_DRIVE_FRACTION, 1. - cfg.FRONT_DRIVE_FRACTION])
    longitudinal = jnp.clip(cfg.MASS * accel * split, -capacity, capacity)
    lateral_capacity = jnp.sqrt(jnp.maximum(capacity ** 2 - longitudinal ** 2, 0.))
    safe_vx = jnp.maximum(vx, cfg.BLEND_LOW_SPEED)
    slip = jnp.array([steering - jnp.arctan2(vy + lf * rate, safe_vx),
                      -jnp.arctan2(vy - lr * rate, safe_vx)])
    stiffness = jnp.array([cfg.FRONT_STIFFNESS, cfg.REAR_STIFFNESS])
    lateral = lateral_capacity * jnp.tanh(stiffness * slip / jnp.maximum(lateral_capacity, 1e-6))
    return longitudinal, lateral


def physical_flow(state, command_speed, steering, cfg):
    """Blend low-speed relaxation with force-driven lateral/yaw dynamics."""
    yaw, vx, vy, rate = state[2:6]
    longitudinal, lateral = tire_forces(state, command_speed, steering, cfg)
    fx_front, fx_rear = longitudinal
    fy_front, fy_rear = lateral
    c, s = jnp.cos(steering), jnp.sin(steering)
    front_side = fx_front * s + fy_front * c
    dynamic_vy = (front_side + fy_rear) / cfg.MASS - vx * rate
    dynamic_rate = (cfg.FRONT_LENGTH * front_side
                    - (cfg.WB - cfg.FRONT_LENGTH) * fy_rear) / cfg.YAW_INERTIA
    target_rate = vx * jnp.tan(steering) / cfg.WB
    target_vy = (cfg.WB - cfg.FRONT_LENGTH) * target_rate
    blend = jnp.clip((vx - cfg.BLEND_LOW_SPEED)
                     / (cfg.BLEND_HIGH_SPEED - cfg.BLEND_LOW_SPEED), 0., 1.)
    dvy = blend * dynamic_vy + (1. - blend) * (target_vy - vy) / cfg.LATERAL_RESPONSE_TIME
    drate = blend * dynamic_rate + (1. - blend) * (target_rate - rate) / cfg.LATERAL_RESPONSE_TIME
    dvx = (fx_front * c - fy_front * s + fx_rear) / cfg.MASS + blend * vy * rate
    return jnp.array([vx * jnp.cos(yaw) - vy * jnp.sin(yaw),
                      vx * jnp.sin(yaw) + vy * jnp.cos(yaw), rate, dvx, dvy, drate])


def integrate_physics(state, speed, steering, cfg):
    """Substep RK4; do not clip physical velocity to conceal infeasibility."""
    dt = cfg.DT / cfg.DYNAMIC_SUBSTEPS

    def advance(_, physical):
        k1 = physical_flow(physical, speed, steering, cfg)
        k2 = physical_flow(physical + dt * k1 / 2, speed, steering, cfg)
        k3 = physical_flow(physical + dt * k2 / 2, speed, steering, cfg)
        k4 = physical_flow(physical + dt * k3, speed, steering, cfg)
        return physical + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return jax.lax.fori_loop(0, cfg.DYNAMIC_SUBSTEPS, advance, state)


def step(x, u, cfg):
    """Advance physical state and the command state consumed by this model."""
    speed = jnp.clip(x[7] + cfg.DT * u[0], 0., cfg.MAX_SPEED)
    physical = integrate_physics(x[:6], speed, u[1], cfg)
    return jnp.concatenate((physical, jnp.array([u[1], speed])))


def tracking_state(X, cfg):
    """Compare path tangent with velocity course, not drifting body heading."""
    course = X[..., 2] + jnp.arctan2(X[..., 4], jnp.maximum(X[..., 3], 1e-6))
    speed = jnp.hypot(X[..., 3], X[..., 4])
    return jnp.stack((X[..., 0], X[..., 1], course, speed), axis=-1)


def envelope_excess(X, cfg):
    """Allow bounded sideslip; penalize only excursions, not slip itself."""
    beta = jnp.arctan2(X[..., 4], jnp.maximum(X[..., 3], cfg.BLEND_LOW_SPEED))
    slip = jnp.maximum(jnp.abs(beta) - cfg.MAX_SIDESLIP, 0.)
    rate = jnp.maximum(jnp.abs(X[..., 5]) - cfg.MAX_YAW_RATE, 0.)
    speed = jnp.maximum(jnp.hypot(X[..., 3], X[..., 4]) - cfg.MAX_SPEED, 0.)
    reverse = jnp.maximum(-X[..., 3], 0.)
    return jnp.stack((slip, rate, speed, reverse), axis=-1)


def trajectory_is_feasible(X, U, cfg):
    """Check finite, bounded motion and reachable commands, not occupancy."""
    excess = envelope_excess(X[1:], cfg)
    previous_steer = jnp.concatenate((X[:1, 6], U[:-1, 1]))
    slew_ok = jnp.all(jnp.abs(U[:, 1] - previous_steer) <= cfg.MAX_STEER_RATE * cfg.DT + 1e-5)
    bounds_ok = jnp.all((U >= jnp.array([cfg.MIN_ACCEL, cfg.MIN_STEER]) - 1e-5)
                        & (U <= jnp.array([cfg.MAX_ACCEL, cfg.MAX_STEER]) + 1e-5))
    return (jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(U))
            & jnp.all(excess <= 1e-5) & slew_ok & bounds_ok)
