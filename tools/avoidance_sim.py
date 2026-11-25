"""Closed-loop obstacle avoidance check, without ROS or the car.

A straight corridor with a 30 cm box on the route (default 5 m ahead of the start).
Each tick simulates the LiDAR, runs the real control_step with the given config,
and moves the car with the controller's own kinematic model. Prints, per seed,
whether the car got past the box, the smallest body gap (the whole body, so turning
back too early shows), the speed beside the box and how often the steering
switched sides (dithering). The steering delay from the config is simulated.

    python3 tools/avoidance_sim.py                        # config/car_config.yaml
    python3 tools/avoidance_sim.py --config my.yaml --box-left 0.25 --seeds 5

Needs JAX and PyYAML (e.g. the .venv-jax on the Jetson). About 20 s per seed on CPU.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mppi' / 'test'))
from test_control_step import controller as c  # noqa: E402  (controller code without ROS)

BEAM_ANGLES = np.linspace(-2.3562, 2.3557, 811)   # SICK TiM5xx on the car
# Body around the pose point (the LiDAR, near the front of the car) [m]
BODY_FRONT, BODY_REAR, HALF_WIDTH = 0.13, 0.37, 0.155
BOX_HALF = 0.15


def load_config(path):
    """MPPIConfig and obstacle inflation from a car_config.yaml."""
    params = yaml.safe_load(Path(path).read_text())['mppi_node']['ros__parameters']
    values = {}
    for field in c.tunable_config_fields():
        name = c.config_parameter_name(field.name)
        if name in params:
            values[field.name] = c.config_value(name, field.default, params[name])
    # The simulated car is the controller's own model, so the measured curvature gain
    # of the real steering does not apply; the steering delay is simulated.
    values.update(STEERING_CURVATURE_GAIN=1.0)
    cfg = replace(c.MPPIConfig(), **values)
    return cfg, float(params.get('obstacle_inflation', 0.22)), float(params.get('clearance_range', 0.25))


def lidar(x, y, yaw, box, wall):
    """Ranges from the pose to the corridor walls (y = +-wall) and the box."""
    angles = yaw + BEAM_ANGLES
    dx, dy = np.cos(angles), np.sin(angles)
    ranges = np.full(angles.shape, 10.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        for wall_y in (-wall, wall):
            t = (wall_y - y) / dy
            ranges = np.where((t > 0) & (t < ranges), t, ranges)
        bx, by = box
        tx = np.stack(((bx - BOX_HALF - x) / dx, (bx + BOX_HALF - x) / dx))
        ty = np.stack(((by - BOX_HALF - y) / dy, (by + BOX_HALF - y) / dy))
        near = np.maximum(tx.min(0), ty.min(0))
        far = np.minimum(tx.max(0), ty.max(0))
    hit = (far >= near) & (near > 0) & (near < ranges)
    return np.where(hit, near, ranges).astype(np.float32)


def body_gap(x, y, yaw, box):
    """Smallest distance from the body rectangle to the box outline (< 0: contact)."""
    edge = np.linspace(-BOX_HALF, BOX_HALF, 7)
    corners = [(u, v) for u in edge for v in (-BOX_HALF, BOX_HALF)]
    corners += [(u, v) for v in edge for u in (-BOX_HALF, BOX_HALF)]
    px, py = np.array(corners).T + np.array(box)[:, None]
    forward = np.cos(yaw) * (px - x) + np.sin(yaw) * (py - y)
    left = -np.sin(yaw) * (px - x) + np.cos(yaw) * (py - y)
    outside = np.maximum(forward - BODY_FRONT, -BODY_REAR - forward)
    return float(np.min(np.maximum(outside, np.abs(left) - HALF_WIDTH)))


def steering_reversals(commands, threshold=0.05):
    """How often the steering command changes side (left/right dithering)."""
    signs = np.sign([s for s in commands if abs(s) > threshold])
    return int(np.sum(signs[1:] != signs[:-1]))


def drive_past_box(cfg, inflation, clearance_range, box, wall, speed, seed, pose_noise=0.0, start_speed=None,
                   max_steps=250):
    """Returns (passed, smallest body gap, lateral position at the box, steering reversals,
    speed at the box)."""
    xs = np.arange(-5.0, 40.0, 0.05)
    course = jnp.asarray(np.column_stack((xs, 0 * xs, 0 * xs, 0 * xs + speed)).astype(np.float32))
    grid = c.og.OccupancyGrid.create(
        width=3.0, height=7.0, middle=(1.5, 1.0), resolution=0.05, num_beams=811,
        inflation_radius=inflation, distance_cap=inflation + clearance_range)
    state = c.ControllerState(jax.random.PRNGKey(seed), jnp.zeros((cfg.H, cfg.NU)),
                              jnp.broadcast_to(jnp.array(cfg.U_STD), (cfg.H, cfg.NU)))
    x = np.array([0.0, 0.0, 0.0, speed if start_speed is None else start_speed], np.float32)
    delay = cfg.steering_delay_steps
    noise = np.random.default_rng(seed)
    sent = [0.0] * delay            # steering commands not yet applied by the car
    gap, side, commands, speed_at_box = np.inf, 0.0, [], np.nan
    for _ in range(max_steps):
        prefix = np.array(sent[-delay:], np.float32) if delay else None
        # Localization error: the controller sees a jittered pose, the scan is taken
        # from the true one (like the particle filter on the car).
        seen = x + np.array([pose_noise, pose_noise, pose_noise / 2, 0.0], np.float32) \
            * noise.standard_normal(4).astype(np.float32)
        observation = c.ControlObservation(
            seen, lidar(*x[:3], box, wall), np.float32(cfg.DT), np.bool_(False),
            np.float32(10.0), prefix)
        result = c.control_step(state, observation, course, grid, cfg)
        state, grid = result.state, result.grid
        steer = float(result.action[1])
        applied = sent[-delay] if delay else steer
        sent.append(steer)
        commands.append(steer)
        x = np.asarray(c.step(jnp.asarray(x), jnp.array([result.action[0], applied]), cfg), np.float32)
        gap = min(gap, body_gap(*x[:3], box))
        if abs(x[0] - box[0]) < 0.1:
            side, speed_at_box = float(x[1]), float(x[3])
        if x[0] > box[0] + 1.5:
            return True, gap, side, steering_reversals(commands), speed_at_box
    return False, gap, side, steering_reversals(commands), speed_at_box


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default=str(ROOT / 'config' / 'car_config.yaml'))
    parser.add_argument('--box-left', type=float, default=0.0, help='box offset from the route [m]')
    parser.add_argument('--box-ahead', type=float, default=5.0, help='box distance from the start [m]')
    parser.add_argument('--wall', type=float, default=1.4, help='corridor half width [m]')
    parser.add_argument('--speed', type=float, default=1.5, help='route speed [m/s]')
    parser.add_argument('--start-speed', type=float, default=None,
                        help='speed at the start [m/s], default the route speed (0 = from rest)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--pose-noise', type=float, default=0.0,
                        help='std of the localization error [m]; the yaw gets half of it in rad')
    args = parser.parse_args()

    cfg, inflation, clearance_range = load_config(args.config)
    cfg = replace(cfg, MAX_SPEED=max(cfg.MAX_SPEED, args.speed))
    print(f'inflation {inflation} m, q {cfg.Q}, steering delay {cfg.steering_delay_steps * cfg.DT:.2f} s, '
          f'box {args.box_left:+.2f} m from route at {args.box_ahead} m, corridor +-{args.wall} m, '
          f'{args.speed} m/s, pose noise {args.pose_noise} m')
    passed = 0
    for seed in range(args.seeds):
        ok, gap, side, reversals, box_speed = drive_past_box(
            cfg, inflation, clearance_range, (args.box_ahead, args.box_left), args.wall, args.speed, seed,
            args.pose_noise, args.start_speed)
        passed += ok and gap > 0
        verdict = 'passed' if ok and gap > 0 else 'CONTACT' if gap <= 0 else 'stopped'
        print(f'  seed {seed}: {verdict:8s} min body gap {gap * 100:5.1f} cm, '
              f'lateral position at the box {side:+.2f} m, speed there {box_speed:.2f} m/s, '
              f'steering reversals {reversals}')
    print(f'{passed}/{args.seeds} passed')
    return 0 if passed == args.seeds else 1


if __name__ == '__main__':
    sys.exit(main())
