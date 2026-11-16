#!/usr/bin/env python3
"""Generate/build or load acados solver and run CPU-only closed-loop bicycle tracking."""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mppi' / 'scripts'))
from acados_tracker import AcadosTracker, _bicycle_step


def create_demo(prebuilt_directory=None):
    """Configure matching nonlinear plant and a curved nominal trajectory."""
    cfg = SimpleNamespace(H=20, DT=0.05, WB=0.33, INTEGRATOR='rk4',
                          MIN_ACCEL=-3., MAX_ACCEL=3., MIN_STEER=-0.4189,
                          MAX_STEER=0.4189, MIN_SPEED=0., MAX_SPEED=6.)
    state, control = ca.SX.sym('state', 4), ca.SX.sym('control', 2)
    plant = ca.Function('bicycle_step', [state, control], [_bicycle_step(ca, state, control, cfg)])
    directory = prebuilt_directory or ROOT.parent / 'acados_generated' / 'demo'
    tracker = AcadosTracker(cfg, export_directory=directory,
                            generate=prebuilt_directory is None)
    return cfg, plant, tracker


def build_nominal(plant, cfg, steps):
    """Constant-curvature path exercises nonzero-steering online Jacobians."""
    controls = np.tile([0., 0.08], (steps + cfg.H, 1))
    states = [np.array([0., 0., 0., 2.])]
    for control in controls:
        states.append(np.asarray(plant(states[-1], control)).ravel())
    return np.asarray(states), controls


def track_path(tracker, plant, reference, controls, steps):
    """Run receding-horizon feedback with an initial lateral/heading error."""
    state = reference[0] + np.array([0., 0.25, 0.10, 0.])
    previous = np.zeros(2)
    elapsed, errors = [], []
    for tick in range(steps):
        start = time.perf_counter()
        plan = tracker.solve(state, reference[tick:tick + tracker.cfg.H + 1],
                             controls[tick:tick + tracker.cfg.H], previous)
        elapsed.append((time.perf_counter() - start) * 1000)
        previous = plan[0]
        state = np.asarray(plant(state, previous)).ravel()
        errors.append(np.linalg.norm(state[:2] - reference[tick + 1, :2]))
        if tick % 20 == 0 or tick == steps - 1:
            print(f'tick={tick:3d} position_error={errors[-1]:.4f} m '
                  f'accel={previous[0]:+.3f} steer={previous[1]:+.4f} rad '
                  f'solve_ms={elapsed[-1]:.2f}')
    return np.asarray(elapsed), np.asarray(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prebuilt-dir', type=Path,
                        help='Load existing native library and acados_ocp.json; never generate/build')
    args = parser.parse_args()
    cfg, plant, tracker = create_demo(args.prebuilt_dir)
    reference, controls = build_nominal(plant, cfg, steps=100)
    elapsed, errors = track_path(tracker, plant, reference, controls, steps=100)
    print(f'100 solves completed; final position error={errors[-1]:.6f} m; '
          f'latency mean/median/p95/max={elapsed.mean():.2f}/{np.median(elapsed):.2f}/'
          f'{np.percentile(elapsed, 95):.2f}/{elapsed.max():.2f} ms '
          '(Python input/output included; generation/loading excluded)')
    steady = elapsed[1:]
    print(f'First solve={elapsed[0]:.2f} ms; remaining 99 solves '
          f'mean/median/p95/max={steady.mean():.2f}/{np.median(steady):.2f}/'
          f'{np.percentile(steady, 95):.2f}/{steady.max():.2f} ms')
    print(f'Solver directory: {args.prebuilt_dir or ROOT.parent / "acados_generated" / "demo"}')
    if not np.isfinite(errors).all() or errors[-1] > 0.05:
        raise SystemExit('Tracking did not converge within 5 cm')


if __name__ == '__main__':
    main()
