"""Closed-loop obstacle avoidance check, without ROS or the car.

A straight corridor with a 30 cm box on the route; the car starts 4 m before it.
Each tick simulates the LiDAR, runs the real control_step with the given config,
and moves the car with the controller's own kinematic model. Prints, per seed,
whether the car got past the box and the smallest gap between body and box.

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
HALF_LENGTH, HALF_WIDTH = 0.25, 0.155               # body around the pose point [m]
BOX_HALF = 0.15


def load_config(path):
    """MPPIConfig and obstacle inflation from a car_config.yaml."""
    params = yaml.safe_load(Path(path).read_text())['mppi_node']['ros__parameters']
    values = {}
    for field in c.tunable_config_fields():
        name = c.config_parameter_name(field.name)
        if name in params:
            values[field.name] = c.config_value(name, field.default, params[name])
    # Delay compensation needs a command history this simple loop does not keep.
    values.update(STEERING_DELAY_S=0.0, STEERING_CURVATURE_GAIN=1.0)
    cfg = replace(c.MPPIConfig(), **values)
    return cfg, float(params.get('obstacle_inflation', 0.22))


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
    return float(np.min(np.maximum(np.abs(forward) - HALF_LENGTH, np.abs(left) - HALF_WIDTH)))


def drive_past_box(cfg, inflation, box, wall, speed, seed, max_steps=200):
    xs = np.arange(-5.0, 40.0, 0.05)
    course = jnp.asarray(np.column_stack((xs, 0 * xs, 0 * xs, 0 * xs + speed)).astype(np.float32))
    grid = c.og.OccupancyGrid.create(
        width=3.0, height=7.0, middle=(1.5, 1.0), resolution=0.05, num_beams=811,
        inflation_radius=inflation, distance_cap=inflation + 0.25)
    state = c.ControllerState(jax.random.PRNGKey(seed), jnp.zeros((cfg.H, cfg.NU)),
                              jnp.broadcast_to(jnp.array(cfg.U_STD), (cfg.H, cfg.NU)))
    x = np.array([0.0, 0.0, 0.0, speed], np.float32)
    gap, side = np.inf, 0.0
    for _ in range(max_steps):
        observation = c.ControlObservation(
            x.copy(), lidar(*x[:3], box, wall), np.float32(cfg.DT), np.bool_(False))
        result = c.control_step(state, observation, course, grid, cfg)
        state, grid = result.state, result.grid
        x = np.asarray(c.step(jnp.asarray(x), result.action, cfg), np.float32)
        gap = min(gap, body_gap(*x[:3], box))
        if abs(x[0] - box[0]) < 0.1:
            side = float(x[1])
        if x[0] > box[0] + 1.5:
            return True, gap, side
    return False, gap, side


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default=str(ROOT / 'config' / 'car_config.yaml'))
    parser.add_argument('--box-left', type=float, default=0.0, help='box offset from the route [m]')
    parser.add_argument('--wall', type=float, default=1.4, help='corridor half width [m]')
    parser.add_argument('--speed', type=float, default=1.5, help='route speed [m/s]')
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    cfg, inflation = load_config(args.config)
    cfg = replace(cfg, MAX_SPEED=max(cfg.MAX_SPEED, args.speed))
    print(f'inflation {inflation} m, q {cfg.Q}, box {args.box_left:+.2f} m from route, '
          f'corridor +-{args.wall} m, {args.speed} m/s')
    passed = 0
    for seed in range(args.seeds):
        ok, gap, side = drive_past_box(cfg, inflation, (4.0, args.box_left), args.wall, args.speed, seed)
        passed += ok and gap > 0
        verdict = 'passed' if ok and gap > 0 else 'CONTACT' if gap <= 0 else 'stopped'
        print(f'  seed {seed}: {verdict:8s} min body gap {gap * 100:5.1f} cm, '
              f'lateral position at the box {side:+.2f} m')
    print(f'{passed}/{args.seeds} passed')
    return 0 if passed == args.seeds else 1


if __name__ == '__main__':
    sys.exit(main())
