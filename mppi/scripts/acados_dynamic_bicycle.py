"""Standalone CasADi counterpart of MPPI's slip-capable discrete dynamics.

Not connected to the acados OCP, tracker, code generation or ROS node.
"""


def _tire_forces(ca, state, command_speed, steering, cfg):
    """Static axle loads and combined-grip saturation, matching dynamic_bicycle."""
    vx, vy, rate = state[3], state[4], state[5]
    lf, lr = cfg.FRONT_LENGTH, cfg.WB - cfg.FRONT_LENGTH
    capacity = cfg.TIRE_MU * cfg.MASS * 9.81 * ca.vertcat(lr, lf) / cfg.WB
    accel = ca.fmin(ca.fmax((command_speed - vx) / cfg.SPEED_RESPONSE_TIME,
                            cfg.MIN_ACCEL), cfg.MAX_ACCEL)
    demand = cfg.MASS * accel * ca.vertcat(cfg.FRONT_DRIVE_FRACTION,
                                          1. - cfg.FRONT_DRIVE_FRACTION)
    longitudinal = ca.fmin(ca.fmax(demand, -capacity), capacity)
    # Positive floor keeps CasADi Jacobians finite at full longitudinal saturation.
    # Unlike MPPI's zero floor, this permits at most 1e-6 N lateral force per axle.
    lateral_capacity = ca.sqrt(ca.fmax(capacity ** 2 - longitudinal ** 2, 1e-12))
    safe_vx = ca.fmax(vx, cfg.BLEND_LOW_SPEED)
    slip = ca.vertcat(steering - ca.atan2(vy + lf * rate, safe_vx),
                      -ca.atan2(vy - lr * rate, safe_vx))
    stiffness = ca.vertcat(cfg.FRONT_STIFFNESS, cfg.REAR_STIFFNESS)
    lateral = lateral_capacity * ca.tanh(stiffness * slip / ca.fmax(lateral_capacity, 1e-6))
    return longitudinal, lateral


def _physical_flow(ca, state, command_speed, steering, cfg):
    """COM motion and blended force-driven/low-speed lateral and yaw response."""
    yaw, vx, vy, rate = state[2], state[3], state[4], state[5]
    longitudinal, lateral = _tire_forces(ca, state, command_speed, steering, cfg)
    fx_front, fx_rear = longitudinal[0], longitudinal[1]
    fy_front, fy_rear = lateral[0], lateral[1]
    c, s = ca.cos(steering), ca.sin(steering)
    front_side = fx_front * s + fy_front * c
    dynamic_vy = (front_side + fy_rear) / cfg.MASS - vx * rate
    dynamic_rate = (cfg.FRONT_LENGTH * front_side
                    - (cfg.WB - cfg.FRONT_LENGTH) * fy_rear) / cfg.YAW_INERTIA
    target_rate = vx * ca.tan(steering) / cfg.WB
    target_vy = (cfg.WB - cfg.FRONT_LENGTH) * target_rate
    blend = ca.fmin(ca.fmax((vx - cfg.BLEND_LOW_SPEED)
                            / (cfg.BLEND_HIGH_SPEED - cfg.BLEND_LOW_SPEED), 0.), 1.)
    dvy = blend * dynamic_vy + (1. - blend) * (target_vy - vy) / cfg.LATERAL_RESPONSE_TIME
    drate = blend * dynamic_rate + (1. - blend) * (target_rate - rate) / cfg.LATERAL_RESPONSE_TIME
    dvx = (fx_front * c - fy_front * s + fx_rear) / cfg.MASS + blend * vy * rate
    return ca.vertcat(vx * ca.cos(yaw) - vy * ca.sin(yaw),
                      vx * ca.sin(yaw) + vy * ca.cos(yaw), rate, dvx, dvy, drate)


def dynamic_bicycle_step(ca, x, u, cfg):
    """Return an eight-state CasADi expression suitable for disc_dyn_expr.

    x = [COM x, COM y, body yaw, body vx, body vy, yaw rate,
         previous steering command, previous speed setpoint]
    u = [speed-setpoint slew (m/s^2), steering command (rad)]

    Holds the updated speed setpoint and steering command over cfg.DT, using
    cfg.DYNAMIC_SUBSTEPS RK4 steps, exactly as MPPI does. Does not clip physical
    velocity or project steering: actuator bounds/slew belong to the future OCP.
    Hard clips/blending are piecewise differentiable, not globally smooth.
    No acados dependency or solver construction is required; pass CasADi as ca.
    """
    if x.numel() != 8 or u.numel() != 2:
        raise ValueError('dynamic bicycle requires 8 states and 2 controls')
    speed = ca.fmin(ca.fmax(x[7] + cfg.DT * u[0], 0.), cfg.MAX_SPEED)
    steering = u[1]
    physical = x[:6]
    dt = cfg.DT / cfg.DYNAMIC_SUBSTEPS
    for _ in range(cfg.DYNAMIC_SUBSTEPS):
        k1 = _physical_flow(ca, physical, speed, steering, cfg)
        k2 = _physical_flow(ca, physical + dt * k1 / 2, speed, steering, cfg)
        k3 = _physical_flow(ca, physical + dt * k2 / 2, speed, steering, cfg)
        k4 = _physical_flow(ca, physical + dt * k3, speed, steering, cfg)
        physical = physical + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return ca.vertcat(physical, steering, speed)
