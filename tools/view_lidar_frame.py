#!/usr/bin/env python3
"""Inspect one bag frame with production or temporally fused log-odds grid, without ROS.

Dependencies: rosbags, jax, numpy, matplotlib.
Example: python tools/view_lidar_frame.py ../../rosbags/lidar_grid_debug
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mppi' / 'scripts'))
import occupancy_grid as og


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--frame', type=int, default=0, help='Zero-based scan index')
    parser.add_argument('--topic', default='/scan')
    parser.add_argument('--log-odds', action='store_true',
                        help='Fuse scans 0..frame with motion compensation, in sensor frame')
    parser.add_argument('--fixed-frame', default='map', help='Parent of direct LiDAR TF')
    parser.add_argument('--max-pose-gap', type=float, default=.2,
                        help='Maximum time to each interpolation sample [s]')
    parser.add_argument('--lidar-pose', type=float, nargs=3, default=None,
                        metavar=('FORWARD', 'LEFT', 'YAW'),
                        help='Sensor pose in grid frame; default: recorded base_link -> laser')
    parser.add_argument('--save', type=Path, help='Also save figure')
    parser.add_argument('--no-show', action='store_true', help='Save without opening GUI')
    args = parser.parse_args()
    if args.lidar_pose is None:
        args.lidar_pose = (0., 0., 0.) if args.log_odds else (0.27, 0., 0.)
    if args.log_odds and any(args.lidar_pose):
        parser.error('--log-odds currently requires sensor-frame --lidar-pose 0 0 0')
    if args.max_pose_gap <= 0:
        parser.error('--max-pose-gap must be positive')
    if args.frame < 0:
        parser.error('--frame must be non-negative')
    if args.no_show and args.save is None:
        parser.error('--no-show requires --save')
    return args


def read_scan(bag, topic, frame):
    with AnyReader([bag.resolve()]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f'Topic {topic!r} not found in {bag}')
        for index, (connection, stamp, raw) in enumerate(reader.messages(connections=connections)):
            if index == frame:
                return reader.deserialize(raw, connection.msgtype), stamp
    raise ValueError(f'Scan index {frame} exceeds available scans on {topic}')


def build_grid(scan, lidar_pose):
    # Match node dimensions and clearance settings, but use recorded beam angles.
    grid = og.OccupancyGrid.create(
        width=3., height=7., middle=(1.5, 1.), resolution=.05,
        inflation_radius=.15, distance_cap=.25,
        num_beams=len(scan.ranges), angle_min=float(scan.angle_min),
        angle_max=float(scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment),
        lidar_pose=lidar_pose,
    )
    return grid.from_scan(scan.ranges, range_max=float(scan.range_max))


def header_time(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def planar_pose(transform):
    t, q = transform.translation, transform.rotation
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return np.array([t.x, t.y, yaw], dtype=np.float64)


def read_replay(args):
    """Read scan prefix and direct fixed-frame -> sensor TF, using header times."""
    scans, poses = [], {}
    with AnyReader([args.bag.resolve()]) as reader:
        connections = [c for c in reader.connections if c.topic in (args.topic, '/tf')]
        for connection, stamp, raw in reader.messages(connections=connections):
            if connection.topic == args.topic:
                if len(scans) <= args.frame:
                    scans.append((reader.deserialize(raw, connection.msgtype), stamp))
            else:
                msg = reader.deserialize(raw, connection.msgtype)
                for tf in msg.transforms:
                    if tf.header.frame_id.lstrip('/') == args.fixed_frame.lstrip('/'):
                        key = tf.child_frame_id.lstrip('/')
                        poses.setdefault(key, {})[header_time(tf.header)] = planar_pose(tf.transform)
    if len(scans) <= args.frame:
        raise ValueError(f'Scan index {args.frame} exceeds available scans on {args.topic}')
    return scans, poses


def aligned_pose(scan, poses, max_gap):
    sensor = scan.header.frame_id.lstrip('/')
    samples = poses.get(sensor)
    if not samples:
        raise ValueError(f'No direct fixed-frame -> {sensor} TF; TF chains are not supported')
    times = np.array(sorted(samples), dtype=np.int64)
    stamp = header_time(scan.header)
    index = int(np.searchsorted(times, stamp))
    if index < len(times) and times[index] == stamp:
        return samples[int(times[index])]
    if index == 0 or index == len(times):
        raise ValueError(f'LiDAR TF does not bracket scan timestamp {stamp}')
    before, after = int(times[index - 1]), int(times[index])
    if max(stamp - before, after - stamp) > max_gap * 1e9:
        raise ValueError(f'LiDAR TF too stale at scan timestamp {stamp}')
    weight = (stamp - before) / (after - before)
    prev, nxt = samples[before], samples[after]
    delta = nxt - prev
    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
    return prev + weight * delta


def log_odds_config(scan):
    return og.GridConfig(
        inflation_radius=.15, distance_cap=.25, num_beams=len(scan.ranges),
        angle_min=float(scan.angle_min),
        angle_max=float(scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment),
    )


class LogOddsReplay:
    """Own persistent mapping state and advance it with each aligned scan."""

    def __init__(self, scan):
        self.cfg = log_odds_config(scan)
        self.grid = og.OccupancyGrid.create(**asdict(self.cfg))
        self.count = 0

    def observe(self, scan, pose):
        if log_odds_config(scan) != self.cfg:
            raise ValueError('Scan geometry changed during replay')
        self.grid = self.grid.from_scan(scan.ranges, float(scan.range_max), pose=pose)
        self.count += 1

    def snapshot(self):
        return SimpleNamespace(spec=self.grid.spec, cells=self.grid.cells,
                               free_distance=self.grid.free_distance,
                               allow_unknown=not og.UNKNOWN_IS_OBSTACLE,
                               fused_scans=self.count)


def replay_log_odds(args):
    scans, poses = read_replay(args)
    replay = LogOddsReplay(scans[0][0])
    previous_stamp = None
    for scan, _ in scans:
        stamp = header_time(scan.header)
        if previous_stamp is not None and stamp <= previous_stamp:
            raise ValueError('Scan header timestamps must strictly increase')
        replay.observe(scan, aligned_pose(scan, poses, args.max_pose_gap))
        previous_stamp = stamp
    scan, stamp = scans[-1]
    return scan, stamp, replay.snapshot()


def prepare_frame(args):
    if args.log_odds:
        return replay_log_odds(args)
    scan, stamp = read_scan(args.bag, args.topic, args.frame)
    return scan, stamp, build_grid(scan, args.lidar_pose)


def scan_points(scan, lidar_pose):
    ranges = np.asarray(scan.ranges)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    valid = np.isfinite(ranges) & (ranges > 0.)
    forward, left, yaw = lidar_pose
    return (left + ranges[valid] * np.sin(angles[valid] + yaw),
            forward + ranges[valid] * np.cos(angles[valid] + yaw))


def plot_frame(scan, stamp, grid, args):
    cfg = grid.spec
    cells = np.asarray(grid.cells)
    clearance = np.asarray(grid.free_distance)
    allow_unknown = getattr(grid, 'allow_unknown', not og.UNKNOWN_IS_OBSTACLE)
    safe = (clearance >= cfg.inflation_radius)
    if not allow_unknown:
        safe &= cells == og.FREE
    extent = (-cfg.middle[0], cfg.shape[1] * cfg.resolution - cfg.middle[0],
              -cfg.middle[1], cfg.shape[0] * cfg.resolution - cfg.middle[1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 8), constrained_layout=True)
    axes[0].scatter(*scan_points(scan, args.lidar_pose), s=5, color='black')
    axes[0].set_title('Raw positive finite returns')
    cmap = ListedColormap(['#999999', '#ffffff', '#202020'])
    image = axes[1].imshow(cells, origin='lower', extent=extent,
                           cmap=cmap, norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], 3))
    fig.colorbar(image, ax=axes[1], ticks=[1, 2, 3], shrink=.5).ax.set_yticklabels(
        ['Unknown', 'Free', 'Occupied'])
    fused = getattr(grid, 'fused_scans', 0)
    axes[1].set_title(f'Occupancy ({fused} scans fused)' if fused else 'Occupancy (no scan fusion)')
    image = axes[2].imshow(clearance, origin='lower', extent=extent,
                           vmin=0., vmax=cfg.distance_cap, cmap='viridis')
    overlay = np.ma.masked_where(~safe, np.ones_like(cells))
    axes[2].imshow(overlay, origin='lower', extent=extent,
                   cmap=ListedColormap(['#00ff80']), alpha=.5, vmin=0, vmax=1)
    fig.colorbar(image, ax=axes[2], shrink=.5, label='Clearance [m]')
    policy = 'unknown allowed' if allow_unknown else 'FREE only'
    axes[2].set_title(f'Green: clearance ≥ {cfg.inflation_radius:g} m ({policy})')
    for ax in axes:
        ax.plot(0., 0., 'r+', markersize=10, label='Grid origin')
        ax.plot(args.lidar_pose[1], args.lidar_pose[0], 'rx', label='LiDAR')
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='Left [m]', ylabel='Forward [m]')
        ax.set_aspect('equal')
    axes[0].legend(loc='upper right')
    fig.suptitle(f'{args.bag.name} | {args.topic} frame {args.frame} | bag time {stamp / 1e9:.6f} s\n'
                 f'{"Motion-compensated log-odds, sensor frame" if args.log_odds else "Production grid"}; ' 
                 f'LiDAR pose {tuple(args.lidar_pose)}; recorded beam angles; no range filtering')
    return fig


def report_frame(scan, grid, args):
    ranges = np.asarray(scan.ranges)
    cells = np.asarray(grid.cells)
    print(f'Beams: {len(ranges)}; zero: {np.sum(ranges == 0)}; '
          f'+inf: {np.sum(np.isposinf(ranges))}; NaN: {np.sum(np.isnan(ranges))}; '
          f'positive < range_min: {np.sum((ranges > 0) & (ranges < scan.range_min))}')
    print('Cells: ' + ', '.join(f'{name}={np.sum(cells == value)}' for name, value in
                              [('unknown', og.UNKNOWN), ('free', og.FREE), ('occupied', og.OCCUPIED)]))
    if args.log_odds:
        print(f'Origin: LiDAR. Fused {grid.fused_scans} scans, '
              f'using interpolated {args.fixed_frame} -> {scan.header.frame_id} TF. '
              f'Unknown allowed: {grid.allow_unknown}')
    else:
        print('Origin: base_link by default, not controller COM. '
              'Use --lidar-pose to inspect another origin.')


def present_figure(fig, args):
    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f'Saved {args.save}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    scan, stamp, grid = prepare_frame(args)
    report_frame(scan, grid, args)
    fig = plot_frame(scan, stamp, grid, args)
    present_figure(fig, args)


if __name__ == '__main__':
    main()
