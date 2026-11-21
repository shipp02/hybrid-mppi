#!/usr/bin/env python3
"""Headless MPPI vs MPC (acados) comparison in the f1tenth gym, no ROS graph.

Steps the real GymBridge (same ST dynamics, LiDAR, base_link odometry) in lock
step with the controller: one control tick of DT, then DT / 0.01 sim steps.

usage: compare_planners.py --track levine --planner mppi --max-speed 4 [--out runs/]
needs: sourced ROS + sim_ws install, ACADOS_SOURCE_DIR, LD_LIBRARY_PATH with
       acados/lib (only for --planner mpc/hybrid), -DMPPI_BUILD_ACADOS=ON install
"""
import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'mppi' / 'scripts'
sys.path.insert(0, str(SCRIPTS))

TRACKS = {  # map_path, start pose (base_link), waypoints
    'levine': ('maps/levine_blocked', (-12.0, 0.0, 0.0), 'levine.csv'),
    'spielberg': ('maps/Spielberg', (14.59, 3.92, -2.877), 'spielberg.csv'),
}
SIM_DT = 0.01
SPEED_BANDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0)


def start_bridge(track):
    """GymBridge with the stock sim.yaml, overridden for this track."""
    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from f1tenth_gym_ros.gym_bridge import GymBridge

    share = get_package_share_directory('f1tenth_gym_ros')
    params = yaml.safe_load(open(os.path.join(share, 'config', 'sim.yaml')))
    bridge = dict(params['bridge']['ros__parameters'])
    map_path, (sx, sy, stheta), _ = TRACKS[track]
    bridge.update(map_path=map_path, sx=sx, sy=sy, stheta=stheta, num_agents=1,
                  kb_teleop=False, async_mode=True)
    path = Path(tempfile.mkdtemp()) / 'bridge.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': bridge}}))
    rclpy.init(args=['--ros-args', '--params-file', str(path)])
    return GymBridge()


def bridge_observation(bridge):
    """What the node sees: base_link pose, speed, and the scan."""
    x, y, yaw = bridge._cog_to_base_link(bridge.ego_pose)
    speed = float(np.hypot(bridge.ego_speed[0], bridge.ego_speed[1]))
    return np.array([x, y, yaw, speed]), np.asarray(bridge.ego_scan, dtype=np.float32)


def load_waypoints(track, max_speed):
    """Same loading as MPPI.__init__: recomputed yaw, speeds capped at max_speed."""
    import mppi_node as mn
    raw = np.loadtxt(SCRIPTS.parent / 'waypoints' / TRACKS[track][2], delimiter=',', skiprows=1)
    d = np.diff(raw[:, :2], axis=0)
    yaw = np.append(np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[-1, 1], d[-1, 0]))
    speed = np.minimum(raw[:, -1], max_speed)
    return mn.make_course(np.column_stack((raw[:, :2], yaw, speed)))


class Controller:
    """The node's compute path for one planner, without ROS."""

    def __init__(self, planner, cfg, course, bridge, lidar_offset, tracker_dir, grid_forward=6.0,
                 switch=None):
        import jax
        import mppi_node as mn
        import occupancy_grid as og
        from planner_switch import PlannerSwitch
        self.switch = switch if planner == 'hybrid' else PlannerSwitch.fixed(planner)
        self.mn, self.jax, self.cfg, self.course, self.planner = mn, jax, cfg, course, planner
        lidar = bridge.lidar_cfg
        self.grid = og.OccupancyGrid.create(
            lidar_pose=(lidar_offset, 0.0, 0.0), width=3.0, height=grid_forward + 1.0,
            middle=(1.5, 1.0),
            resolution=0.05, inflation_radius=0.15, distance_cap=0.25,
            num_beams=int(lidar.num_beams), angle_min=float(lidar.angle_min),
            angle_max=float(lidar.angle_max))
        self.range_max = np.float32(lidar.range_max)
        self.state = mn.ControllerState(
            jax.random.PRNGKey(0), jax.numpy.zeros((cfg.H, cfg.NU)),
            jax.numpy.broadcast_to(jax.numpy.array(cfg.U_STD), (cfg.H, cfg.NU)))
        self.tracker = None
        if planner != 'mppi':
            from acados_tracker import AcadosTracker
            self.tracker = AcadosTracker(cfg, export_directory=tracker_dir, generate=False)
        self.previous = np.zeros(2)  # last published [accel, steer]

    def act(self, x0, ranges):
        """Return [accel, steer], the planner that produced it, and timings."""
        obs = self.mn.ControlObservation(
            x0.astype(np.float32), ranges, np.float32(self.cfg.DT), np.bool_(False), self.range_max)
        t0 = time.perf_counter()
        result, nominal = self.mn.tracking_control_step(
            self.state, obs, self.course, self.grid, self.cfg)
        self.state, self.grid = result.state, result.grid
        action, states, controls, reference = self.jax.device_get(
            (result.action, nominal, result.state.U, result.reference))
        t1 = time.perf_counter()
        mode = self.switch.select(x0[3])
        if mode == 'mpc':
            try:
                action = self.tracker.solve(x0, states, controls, self.previous, reference)[0]
            except (RuntimeError, ValueError):  # node: hybrid falls back to MPPI, mpc stops
                action, mode = ((action, 'mppi') if self.planner == 'hybrid'
                                else (np.array([self.cfg.MIN_ACCEL, 0.0]), 'stop'))
        t2 = time.perf_counter()
        self.previous = np.asarray(action, dtype=float).copy()
        return self.previous, mode, (t1 - t0) * 1e3, (t2 - t1) * 1e3


def drive_command(action, x0, cfg):
    """Same as MPPI._publish_drive."""
    import mppi_node as mn
    return float(action[1]), mn.speed_setpoint(x0, action, cfg)


def cross_track(course, xy):
    import jax.numpy as jnp
    import quintic_spline as qs
    s = qs.project_onto_spline(course.spline, jnp.asarray(xy, jnp.float32))
    point, _ = qs.evaluate_pose(course.spline, s)
    return float(jnp.linalg.norm(point - xy))


def summarize(log, laps, lap_times, crashed, cfg):
    t, v, cte, steer, mppi_ms, mpc_ms = (np.array([row[k] for row in log]) for k in
                                         ('t', 'speed', 'cte', 'steer', 'mppi_ms', 'mpc_ms'))
    startup = t <= 3.0
    reached = np.nonzero(v >= 0.9 * cfg.MAX_SPEED)[0]
    bands = {}
    for lo, hi in zip(SPEED_BANDS[:-1], SPEED_BANDS[1:]):
        inside = (v >= lo) & (v < hi) & (t > 0.5)
        if inside.sum() >= 10:
            bands[f'{lo:g}-{hi:g}'] = dict(n=int(inside.sum()),
                                           mean_cte=float(cte[inside].mean()),
                                           p95_cte=float(np.percentile(cte[inside], 95)))
    steer_rate = np.abs(np.diff(steer)) / cfg.DT
    return dict(
        crashed=crashed, laps=laps, lap_times=lap_times, sim_time=float(t[-1]),
        startup=dict(mean_cte=float(cte[startup].mean()), max_cte=float(cte[startup].max()),
                     distance=float(np.sum(v[startup]) * cfg.DT),
                     t_to_90pct_speed=float(t[reached[0]]) if reached.size else None),
        overall=dict(mean_cte=float(cte.mean()), p95_cte=float(np.percentile(cte, 95)),
                     max_cte=float(cte.max()), mean_speed=float(v.mean()),
                     mean_steer_rate=float(steer_rate.mean())),
        bands=bands,
        compute_ms=dict(mppi_mean=float(mppi_ms.mean()), mpc_mean=float(mpc_ms.mean()),
                        total_p95=float(np.percentile(mppi_ms + mpc_ms, 95))),
    )


def config_overrides(pairs):
    """KEY=VALUE pairs for MPPIConfig, typed like the field defaults."""
    import ast
    import controller_config
    out = {}
    for pair in pairs:
        key, value = pair.split('=', 1)
        default = getattr(controller_config.MPPIConfig, key)
        parsed = value if isinstance(default, str) else ast.literal_eval(value)
        out[key] = tuple(parsed) if isinstance(default, tuple) else type(default)(parsed)
    return out


def run(args, controller_cls=Controller):
    import controller_config
    bridge = start_bridge(args.track)
    cfg = replace(controller_config.MPPIConfig(), MAX_SPEED=args.max_speed,
                  SPEED_LEAD_TIME=args.speed_lead, BRAKE_LEAD_TIME=args.brake_lead,
                  **config_overrides(args.set))
    course = load_waypoints(args.track, args.max_speed)
    from planner_switch import PlannerSwitch
    switch = PlannerSwitch(args.low_planner, args.high_planner, args.high_enter, args.high_exit,
                           int(round(args.dwell / cfg.DT)))
    ctrl = controller_cls(args.planner, cfg, course, bridge, args.lidar_offset, args.tracker_dir,
                          args.grid_forward, switch)
    steps = int(round(cfg.DT / SIM_DT))
    log, crashed, laps, lap_times = [], False, 0, []
    tick = 0
    while tick * cfg.DT < args.duration:
        x0, ranges = bridge_observation(bridge)
        action, mode, mppi_ms, mpc_ms = ctrl.act(x0, ranges)
        steer, speed_cmd = drive_command(action, x0, cfg)
        bridge.ego_steer, bridge.ego_requested_speed = steer, speed_cmd
        for _ in range(steps):
            bridge.drive_timer_callback()
            crashed = crashed or bridge.ego_collision
        tick += 1
        log.append(dict(t=tick * cfg.DT, x=x0[0], y=x0[1], speed=x0[3], steer=steer,
                        speed_cmd=speed_cmd, mode=mode, mppi_ms=mppi_ms, mpc_ms=mpc_ms,
                        cte=cross_track(course, x0[:2])))
        lap_count = int(bridge.env.unwrapped.lap_counts[0])
        if lap_count > laps:
            laps = lap_count
            lap_times.append(float(bridge.env.unwrapped.lap_times[0]))
        if crashed or laps >= args.laps:
            break
    summary = summarize(log, laps, lap_times, crashed, cfg)
    summary.update(track=args.track, planner=args.planner, max_speed=args.max_speed,
                   speed_lead=args.speed_lead, brake_lead=args.brake_lead,
                   grid_forward=args.grid_forward, tag=args.tag, overrides=args.set,
                   modes={m: sum(r['mode'] == m for r in log) for m in ('mppi', 'mpc', 'stop')})
    return summary, log


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--track', choices=TRACKS, default='levine')
    parser.add_argument('--planner', choices=('mppi', 'mpc', 'hybrid'), default='mppi')
    parser.add_argument('--max-speed', type=float, default=2.0)
    parser.add_argument('--laps', type=int, default=1)
    parser.add_argument('--speed-lead', type=float, default=0.05,
                        help='speed setpoint lead time [s] (node SPEED_LEAD_TIME)')
    parser.add_argument('--brake-lead', type=float, default=0.05,
                        help='braking lead time [s] (node BRAKE_LEAD_TIME)')
    parser.add_argument('--grid-forward', type=float, default=6.0,
                        help='occupancy grid length ahead of base_link [m] (node: 6)')
    parser.add_argument('--duration', type=float, default=120.0, help='sim seconds cap')
    parser.add_argument('--lidar-offset', type=float, default=0.275,
                        help='LiDAR ahead of base_link [m]; sim.yaml mounts it at 0.275')
    parser.add_argument('--tracker-dir', default=os.environ.get('MPPI_TRACKER_DIR', ''))
    parser.add_argument('--low-planner', default='mpc', help='hybrid: planner at low speed')
    parser.add_argument('--high-planner', default='mppi', help='hybrid: planner at high speed')
    parser.add_argument('--high-enter', type=float, default=4.0,
                        help='hybrid: enter the high-speed planner at [m/s]')
    parser.add_argument('--high-exit', type=float, default=3.0,
                        help='hybrid: leave the high-speed planner at [m/s]')
    parser.add_argument('--dwell', type=float, default=0.5, help='hybrid: min s between switches')
    parser.add_argument('--tag', default='', help='suffix for --out file names')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override an MPPIConfig field, e.g. --set "Q=(1,20,2,5)"')
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    summary, log = run(args)
    print(json.dumps(summary, indent=1))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = (f'{args.track}_{args.planner}_{args.max_speed:g}_lead{args.speed_lead:g}'
                f'_brake{args.brake_lead:g}_grid{args.grid_forward:g}{args.tag}')
        (out / f'{stem}.json').write_text(json.dumps(summary, indent=1))
        (out / f'{stem}.jsonl').write_text('\n'.join(json.dumps(r, default=float) for r in log))


if __name__ == '__main__':
    main()
