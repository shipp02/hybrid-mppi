#!/usr/bin/env python3
from dataclasses import replace
from controller_config import MPPIConfig
from functools import partial
from typing import NamedTuple
import time
import json

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from rclpy.executors import ExternalShutdownException
from utils import calc_interpolated_ref_trajectory_gpu, shift_sequence
import occupancy_grid as og
import dynamic_bicycle as db

import os
from nav_msgs.msg import Odometry
from selected_path_publisher import PathPublisherProcess
from sensor_msgs.msg import LaserScan
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def measured_state(msg, cfg):
    """Decode Odometry; dynamic mode requires COM/body-frame twist (ROS contract)."""
    position = msg.pose.pose.position
    q = msg.pose.pose.orientation
    velocity = msg.twist.twist.linear
    yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    if cfg.DYNAMICS == 'dynamic':
        if not msg.child_frame_id:
            raise ValueError('dynamic odometry requires a body/COM child_frame_id')
        motion = [velocity.x, velocity.y, msg.twist.twist.angular.z]
    else:
        motion = [np.hypot(velocity.x, velocity.y)]
    return np.array([position.x, position.y, yaw, *motion], dtype=np.float64)


def dynamics(x, u, cfg):
    """
    Kinematic bicycle model f(x, u) of a single state (README Section II.a)
    Use jnp instead of np or math so JAX can trace it.
    :param x: state [p_x, p_y, yaw, v], shape (NX,)
    :param u: input [a, delta], shape (NU,)
    :param cfg: MPPIConfig, cfg.WB is the wheelbase
    :return: dx/dt, shape (NX,)
    """
    # TODO: fill in the kinematic model
    px, py, yaw, v = x
    a, delta = u

    return jnp.array([
        v * jnp.cos(yaw),
        v * jnp.sin(yaw),
        v * jnp.tan(delta) / cfg.WB,
        a
    ])


def step_euler(x, u, cfg):
    """
    Forward Euler step: x_{k+1} = x_k + f(x_k, u_k) * DT (README Section II.b.i)
    :return: next state, shape (NX,)
    """
    # TODO: fill in the Euler step
    return x + dynamics(x, u, cfg) * cfg.DT


def step_rk4(x, u, cfg):
    """
    Bonus: Runge-Kutta 4 step (README Section II.b.ii)
    :return: next state, shape (NX,)
    """
    # TODO: fill in the RK4 step
    dt = cfg.DT

    k1 = dynamics(x, u, cfg)
    k2 = dynamics(x + 0.5 * dt * k1, u, cfg)
    k3 = dynamics(x + 0.5 * dt * k2, u, cfg)
    k4 = dynamics(x + dt * k3, u, cfg)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step(x, u, cfg):
    """
    Discretized dynamics F(x, u): the state after DT, holding u constant
    """
    if cfg.DYNAMICS == 'dynamic':
        return db.step(x, u, cfg)
    if cfg.INTEGRATOR == "rk4":
        x_next = step_rk4(x, u, cfg)
    else:
        x_next = step_euler(x, u, cfg)
    # the published speed is clipped to [MIN_SPEED, MAX_SPEED], so the rollouts
    # must be too: otherwise a car at rest plans to reverse and steers backwards
    return x_next.at[3].set(jnp.clip(x_next[3], cfg.MIN_SPEED, cfg.MAX_SPEED))


def simulate(x0, U, cfg):
    """
    Simulate one control sequence from x0
    :param x0: initial state, shape (NX,)
    :param U: control sequence u_0 ... u_{H-1}, shape (H, NU)
    :return: states x_0 ... x_H, shape (H+1, NX), starting with x0
    """
    # TODO: apply step() H times. A for loop works, but jax.lax.scan compiles
    #       much faster: jax.lax.scan(f, x0, U) calls carry, out = f(carry, u)
    #       for each row u of U and returns the last carry and the stacked outs
    def f(carry, u):
        next_state = step(carry, u, cfg)
        return next_state, next_state

    _, states = jax.lax.scan(f, x0, U)

    return jnp.concatenate((x0[jnp.newaxis, :], states), axis=0)


def rollout(x0, U, cfg):
    """
    Simulate all T sampled control sequences from x0, in parallel
    :param x0: initial state, shape (NX,)
    :param U: sampled control sequences, shape (T, H, NU)
    :return: states, shape (T, H+1, NX)
    """
    # TODO: vectorize simulate() over the first axis of U with jax.vmap
    #       (its in_axes say which axis of each argument to map over, None for none)
    return jax.vmap(simulate, in_axes=(None, 0, None))(x0, U, cfg)


def trajectory_to_car_frame(ref_traj, x, y, yaw):
    """Convert states from world coordinates to car coordinates."""
    ref_traj = jnp.asarray(ref_traj)
    dx = ref_traj[:, 0] - x
    dy = ref_traj[:, 1] - y
    c, s = jnp.cos(yaw), jnp.sin(yaw)

    local_yaw = jnp.arctan2(
        jnp.sin(ref_traj[:, 2] - yaw),
        jnp.cos(ref_traj[:, 2] - yaw),
    )
    return ref_traj.at[:, :3].set(
        jnp.stack(
            (c * dx + s * dy, -s * dx + c * dy, local_yaw),
            axis=1,
        )
    )


def cost_components(X, U, ref, cfg, grid):
    """Per-timestep objective terms shared by optimization and diagnostics."""
    tracked = db.tracking_state(X, cfg) if cfg.DYNAMICS == 'dynamic' else X
    error = tracked[1:] - ref[1:]

    yaw_error = jnp.arctan2(
        jnp.sin(error[:, 2]),
        jnp.cos(error[:, 2])
    )

    error = error.at[:, 2].set(yaw_error)
    
    # Map rollout trajectory into local car frame for collision evaluation
    local_traj = trajectory_to_car_frame(X[1:], X[0, 0], X[0, 1], X[0, 2])
    
    # Evaluate collisions using the immutable JAX occupancy grid
    obs_blocked = grid.is_free(local_traj)
    # Arc length from the car to each rollout point, including the first segment.
    local_xy = jnp.concatenate((jnp.zeros((1, 2)), local_traj[:, :2]), axis=0)
    path_dist = jnp.cumsum(jnp.linalg.norm(jnp.diff(local_xy, axis=0), axis=1))
    # Distance in meters: 10 at the car, 5 at 1 m, decreasing farther ahead.
    blocked_penalty = 1e1 / (1.0 + path_dist)
    # Keep timestep costs separate: mppi_update owns horizon/cost-to-go sums.
    obs_block_cost = 10 * jnp.where(~obs_blocked, blocked_penalty, 0.0)

    # position error in the frame of the reference: [lag, lateral]. From rest the
    # reference runs ahead of the car, and a plain x/y error is then all lag,
    # which only the acceleration can fix: weighting the lateral error on its
    # own keeps a clear signal for the steering
    cos_ref = jnp.cos(ref[1:, 2])
    sin_ref = jnp.sin(ref[1:, 2])
    lag = cos_ref * error[:, 0] + sin_ref * error[:, 1]
    lateral = -sin_ref * error[:, 0] + cos_ref * error[:, 1]
    error = error.at[:, 0].set(lag).at[:, 1].set(lateral)

    Q = jnp.array(cfg.Q)
    Qf = jnp.array(cfg.Qf)
    R = jnp.array(cfg.R)
    Rd = jnp.array(cfg.Rd)

    state_cost = jnp.sum(Q * error ** 2, axis=1)
    final_state_cost = jnp.sum(Qf * error[-1] ** 2)

    state_cost = state_cost.at[-1].set(final_state_cost)

    input_cost = jnp.sum(R * U ** 2, axis=1)

    previous_U = jnp.concatenate((U[:1], U[:-1]), axis=0)
    input_difference = U - previous_U
    input_difference_cost = jnp.sum(
        Rd * input_difference ** 2,
        axis=1
    )

    envelope_cost = 0.
    if cfg.DYNAMICS == 'dynamic':
        envelope_cost = cfg.ENVELOPE_WEIGHT * jnp.sum(db.envelope_excess(X[1:], cfg) ** 2, axis=1)
    return dict(tracking=state_cost, control=input_cost,
                control_change=input_difference_cost, obstacle=obs_block_cost,
                envelope=jnp.broadcast_to(envelope_cost, state_cost.shape))


def cost(X, U, ref, cfg, grid):
    parts = cost_components(X, U, ref, cfg, grid)
    return (parts['tracking'] + parts['control'] + parts['control_change']
            + parts['obstacle'] + parts['envelope'])


class TopPaths(NamedTuple):
    xy: jax.Array
    totals: jax.Array
    components: dict


def select_top_paths(X, U, C, ref, cfg, grid):
    """Rank total costs on device; export only best 50 XY and objective sums."""
    totals = jnp.sum(C, axis=1)
    scores = jnp.where(jnp.isfinite(totals), totals, jnp.inf)
    indices = jnp.argsort(scores, stable=True)[:min(50, cfg.T)]
    parts = jax.vmap(cost_components, in_axes=(0, 0, None, None, None))(
        X[indices], U[indices], ref, cfg, grid)
    return TopPaths(X[indices, :, :2], totals[indices],
                    {name: jnp.sum(values, axis=1) for name, values in parts.items()})


def sample_controls(key, U, U_std, cfg):
    """
    Sample T control sequences around U, inside the input bounds
    :param U: mean control sequence, shape (H, NU)
    :param U_std: std of each input at each step, shape (H, NU)
    :return: sampled control sequences, shape (T, H, NU)
    """
    lower = jnp.array((cfg.MIN_ACCEL, cfg.MIN_STEER))
    upper = jnp.array((cfg.MAX_ACCEL, cfg.MAX_STEER))

    # standard normal noise, correlated over time: z_t = rho z_{t-1} + sqrt(1 - rho^2) e_t.
    # At rest the steering of the first step has no effect (yaw rate = v tan(delta) / L),
    # so with white noise its update is pure noise; correlated over time, it follows
    # the steering of the next steps, which does move the car
    eps = jax.random.normal(key, (cfg.H, cfg.T, cfg.NU))

    def ar1(z_prev, e):
        z = cfg.NOISE_RHO * z_prev + jnp.sqrt(1.0 - cfg.NOISE_RHO ** 2) * e
        return z, z

    _, z = jax.lax.scan(ar1, eps[0], eps[1:])
    z = jnp.concatenate((eps[:1], z), axis=0).transpose(1, 0, 2)  # (T, H, NU)

    if cfg.SAMPLING == "truncated":
        # Bonus: a Gaussian truncated to the bounds, so every sample is a valid
        # input without piling up on the bounds as clipping does. The inverse CDF
        # maps each N(0, 1) of z to N(0, 1) truncated to [lo, hi], and keeps
        # the correlation over time
        lo = jax.scipy.stats.norm.cdf((lower - U) / U_std)
        hi = jax.scipy.stats.norm.cdf((upper - U) / U_std)
        p = lo + jax.scipy.stats.norm.cdf(z) * (hi - lo)
        z = jax.scipy.special.ndtri(jnp.clip(p, 1e-6, 1.0 - 1e-6))
        return jnp.clip(U + z * U_std, lower, upper)
    if cfg.SAMPLING == "uniform":
        # Bonus: uniform on [U - sqrt(3) std, U + sqrt(3) std] (same std as the
        # Gaussian) truncated to the bounds; Phi(z) is uniform on [0, 1]
        lo = jnp.maximum(lower, U - jnp.sqrt(3.0) * U_std)
        hi = jnp.minimum(upper, U + jnp.sqrt(3.0) * U_std)
        return lo + jax.scipy.stats.norm.cdf(z) * (hi - lo)
    return jnp.clip(U + z * U_std, lower, upper)


def select_feasible_controls(U, samples, trajectories, costs, x0, cfg):
    """Recheck weighted mean: nonlinear feasibility is not preserved by averaging."""
    nominal = simulate(x0, U, cfg)
    mean_ok = db.trajectory_is_feasible(nominal, U, cfg)
    sample_ok = jax.vmap(db.trajectory_is_feasible, in_axes=(0, 0, None))(
        trajectories, samples, cfg)
    score = jnp.where(sample_ok, jnp.sum(costs, axis=1), jnp.inf)
    best = samples[jnp.argmin(score)]
    braking = db.project_controls(
        jnp.tile(jnp.array([cfg.MIN_ACCEL, 0.]), (cfg.H, 1)), x0, cfg)
    alternative = jnp.where(jnp.any(sample_ok), best, braking)
    return jnp.where(mean_ok, U, alternative)


def sample_rollout_controls(key, U, U_std, x0, cfg):
    """Sample reachable commands and preserve the current nominal as a candidate."""
    samples = sample_controls(key, U, U_std, cfg)
    if cfg.DYNAMICS == 'dynamic':
        samples = jax.vmap(db.project_controls, in_axes=(0, None, None))(samples, x0, cfg)
        samples = samples.at[0].set(db.project_controls(U, x0, cfg))
    return samples


def mppi_update(key, U, U_std, x0, ref, cfg, grid, *, export_top_paths=False):
    U_samples = sample_rollout_controls(key, U, U_std, x0, cfg)

    # 2. Simulate: roll out all T sequences from x0
    X_samples = rollout(x0, U_samples, cfg)

    C = jax.vmap(cost, in_axes=(0, 0, None, None, None))(
        X_samples, U_samples, ref, cfg, grid
    )
    if cfg.DYNAMICS == 'dynamic':
        C = jnp.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6)
    S = jnp.flip(jnp.cumsum(jnp.flip(C, axis=1), axis=1), axis=1)

    # 4. Weight: normalize S[:, t] to [0, 1] over the samples, then softmax
    S_min = jnp.min(S, axis=0, keepdims=True)
    S_max = jnp.max(S, axis=0, keepdims=True)
    S_normalized = (S - S_min) / (S_max - S_min + 1e-6)
    weights = jax.nn.softmax(-S_normalized / cfg.LAMBDA, axis=0)

    # 5. Update: U_new[t] = sum_k w[k, t] * u_k[t]
    U_new = jnp.sum(weights[:, :, jnp.newaxis] * U_samples, axis=0)
    if cfg.DYNAMICS == 'dynamic':
        # Per-timestep weights can break slew even when every sample obeys it.
        U_new = db.project_controls(U_new, x0, cfg)
        U_new = select_feasible_controls(U_new, U_samples, X_samples, C, x0, cfg)

    # Bonus: covariance update, Sigma_t = sum_k w[k, t] (u_k[t] - U_t)(u_k[t] - U_t)^T.
    # The truncated sampler draws each input independently, so it keeps the
    # diagonal of Sigma_t: the weighted variance of each input. The std is
    # blended with the previous one and floored so the search never collapses.
    variance = jnp.sum(
        weights[:, :, jnp.newaxis] * (U_samples - U_new) ** 2,
        axis=0
    )
    U_std_new = (1.0 - cfg.COV_ALPHA) * U_std + cfg.COV_ALPHA * jnp.sqrt(variance)
    U_std_new = jnp.clip(U_std_new, jnp.array(cfg.U_STD_MIN), jnp.array(cfg.U_STD))

    outputs = (U_new, U_std_new, X_samples, weights)
    if export_top_paths:
        return (*outputs, select_top_paths(X_samples, U_samples, C, ref, cfg, grid))
    return outputs


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(key, U, x0, ref, cfg, grid):
    U_std = jnp.broadcast_to(jnp.array(cfg.U_STD), (cfg.H, cfg.NU))
    U_new, _, X_samples, weights = mppi_update(key, U, U_std, x0, ref, cfg, grid)
    return U_new, X_samples, weights


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step_cov(key, U, U_std, x0, ref, cfg, grid):
    return mppi_update(key, U, U_std, x0, ref, cfg, grid)


class ControllerState(NamedTuple):
    key: jax.Array
    U: jax.Array
    U_std: jax.Array


class ControlObservation(NamedTuple):
    x0: jax.Array
    ranges: jax.Array
    elapsed: jax.Array
    reset: jax.Array


class OptimizationResult(NamedTuple):
    state: ControllerState
    samples: jax.Array
    weights: jax.Array
    top_paths: TopPaths | None = None


class ControlResult(NamedTuple):
    state: ControllerState
    grid: og.OccupancyGrid
    action: jax.Array
    path_xy: jax.Array | None
    feasible: jax.Array = True
    top_paths: TopPaths | None = None


def prepare_warm_start(state, observation, cfg):
    """Reset after a pose jump, then advance the previous horizon in time."""
    initial_std = jnp.broadcast_to(
        jnp.array(cfg.U_STD, dtype=state.U_std.dtype), state.U_std.shape)
    U = jnp.where(observation.reset, jnp.zeros_like(state.U), state.U)
    U = shift_sequence(U, observation.elapsed, U[-1], cfg.DT)
    if cfg.COV_UPDATE:
        U_std = jnp.where(observation.reset, initial_std, state.U_std)
        U_std = shift_sequence(U_std, observation.elapsed, initial_std[0], cfg.DT)
    else:
        U_std = initial_std
    return ControllerState(state.key, U, U_std)


def build_reference(x0, course, cfg):
    """Build a speed-ramped world-frame reference from the current pose."""
    cx, cy, cyaw, cv = course.T
    return calc_interpolated_ref_trajectory_gpu(
        x0[0], x0[1], cx, cy, cyaw, cv, cfg.DT, cfg.H,
        v0=jnp.hypot(x0[3], x0[4]) if cfg.DYNAMICS == 'dynamic' else x0[3],
        max_accel=cfg.MAX_ACCEL)


def optimize_controls(state, x0, reference, grid, cfg, export_top_paths=False):
    """Own sampling RNG advancement and fixed/adaptive covariance policy."""
    key, sample_key = jax.random.split(state.key)
    outputs = mppi_update(
        sample_key, state.U, state.U_std, x0, reference, cfg, grid,
        export_top_paths=export_top_paths)
    U, updated_std, samples, weights = outputs[:4]
    top_paths = outputs[4] if export_top_paths else None
    U_std = updated_std if cfg.COV_UPDATE else state.U_std
    return OptimizationResult(ControllerState(key, U, U_std), samples, weights, top_paths)


def select_control_outputs(solution, x0, grid, cfg, export_path):
    """Export only selected XY, reusing the nominal required by dynamic checks."""
    path_xy = None
    feasible = jnp.bool_(True)
    if export_path or cfg.DYNAMICS == 'dynamic':
        nominal = simulate(x0, solution.state.U, cfg)
        if cfg.DYNAMICS == 'dynamic':
            feasible = db.trajectory_is_feasible(nominal, solution.state.U, cfg)
        if export_path:
            path_xy = nominal[:, :2]
    return ControlResult(
        solution.state, grid, solution.state.U[0], path_xy, feasible, solution.top_paths)


@partial(jax.jit, static_argnames=("cfg", "export_path", "export_top_paths"))
def control_step(state, observation, course, grid, cfg, *, export_path=False,
                 export_top_paths=False):
    """One compiled dispatch; helpers preserve domain responsibility boundaries."""
    state = prepare_warm_start(state, observation, cfg)
    grid = grid.from_scan(observation.ranges)
    reference = build_reference(observation.x0, course, cfg)
    solution = optimize_controls(
        state, observation.x0, reference, grid, cfg, export_top_paths)
    return select_control_outputs(
        solution, observation.x0, grid, cfg, export_path)


@partial(jax.jit, static_argnames=("cfg", "export_top_paths"))
def tracking_control_step(state, observation, course, grid, cfg, *, export_top_paths=False):
    """Keep optimization on device; export only selected nominal, never sample batch."""
    result = control_step(
        state, observation, course, grid, cfg, export_top_paths=export_top_paths)
    nominal = simulate(observation.x0, result.state.U, cfg)
    return result, nominal


def calc_waypoint_yaw(x, y):
    """Heading toward the next waypoint on a closed track, in radians."""
    points = np.column_stack((x, y))
    if len(points) < 2:
        raise ValueError('At least two waypoints are required to calculate yaw')
    # Some CSVs repeat the first point at the end of the lap.
    repeated_endpoint = np.array_equal(points[0], points[-1])
    track = points[:-1] if repeated_endpoint else points
    delta = np.roll(track, -1, axis=0) - track
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return np.append(yaw, yaw[0]) if repeated_endpoint else yaw

class MPPI(Node):
    """
    Implement Kinematic MPPI on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mppi_node')
        self._configure_device()
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'levine')
        self.declare_parameter('odom_topic', '')
        track = self.get_parameter('track').value
        
        overrides = {}
        for field in ('MAX_SPEED', 'LAMBDA', 'INTEGRATOR', 'SAMPLING', 'COV_UPDATE',
                      'DYNAMICS', 'MASS', 'YAW_INERTIA', 'WB', 'FRONT_LENGTH', 'TIRE_MU',
                      'FRONT_STIFFNESS', 'REAR_STIFFNESS', 'FRONT_DRIVE_FRACTION',
                      'SPEED_RESPONSE_TIME', 'MAX_STEER_RATE', 'DYNAMIC_SUBSTEPS',
                      'BLEND_LOW_SPEED', 'BLEND_HIGH_SPEED', 'LATERAL_RESPONSE_TIME',
                      'MAX_SIDESLIP', 'MAX_YAW_RATE', 'ENVELOPE_WEIGHT'):
            default = getattr(MPPIConfig, field)
            self.declare_parameter(field.lower(), default)
            overrides[field] = type(default)(self.get_parameter(field.lower()).value)
        self.declare_parameter('horizon', MPPIConfig.H)
        overrides['H'] = int(self.get_parameter('horizon').value)
        if overrides['H'] < 1:
            raise ValueError('horizon must be positive')
        self.config = replace(MPPIConfig(), **overrides)
        self.declare_parameter('tracker', 'none')
        self.declare_parameter('tracker_prebuilt_dir', '')
        self.declare_parameter('tracker_max_steer_rate', float(np.pi))
        self.tracker = self._create_tracker()
        self._last_tracker_action = np.zeros(2)
        self._last_drive_command = np.zeros(2)  # [steering, speed setpoint]

        self.control_state = ControllerState(
            jax.random.PRNGKey(0),
            jnp.zeros((self.config.H, self.config.NU)),
            jnp.broadcast_to(
                jnp.array(self.config.U_STD), (self.config.H, self.config.NU)),
        )
        self.last_stamp = None
        self.last_x0 = None

        odom_topic = self.get_parameter('odom_topic').value
        assert odom_topic != ''
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 1,
        )
        self.control_timer = self.create_timer(
            self.config.DT, self.control_callback,
        )
        self._last_control_time = None
        self._latest_odom = None
        self._odom_child_frame = ''
        self._latest_scan = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._control_latencies = []
        self._control_periods = []
        self._last_control_start = None
        self._stats_window_start = None

        # Geometry is initialized after the fixed LiDAR pose becomes available.
        self._grid_settings = dict(
            width=3.0,
            height=7.0,
            middle=(1.5, 1.0),
            resolution=0.05,
            inflation_radius=0.15,
            num_beams=811,
            distance_cap=0.25,
        )
        self.grid = None
        self._grid_frames = None

        # TODO: load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('mppi'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.
        waypoint_path = os.path.join(
            get_package_share_directory('mppi'),
            'waypoints',
            track if track.endswith('.csv') else track + '.csv'
        )
        self.waypoints = np.loadtxt(
            waypoint_path, delimiter=',', skiprows=1
        )
        if self.waypoints.shape[1] not in (3, 4):
            raise ValueError('Waypoint CSV must contain x,y,vel or x,y,theta,vel')
        yaw = calc_waypoint_yaw(self.waypoints[:, 0], self.waypoints[:, 1])
        self.waypoints = np.column_stack((
            self.waypoints[:, :2], yaw, self.waypoints[:, -1],
        ))
        d = np.diff(self.waypoints[:, :2], axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        yaw = np.append(yaw, yaw[-1])
        # self.waypoints[:, 3] = np.minimum(self.waypoints[:, 3], self.config.MAX_SPEED)
        # Fixed course geometry: compute once, upload once.
        self.course = jax.device_put(np.column_stack((
            self.waypoints[:, :2], yaw, self.waypoints[:, 3],
        )).astype(np.float32), self.device)

        self.declare_parameter('publish_top_paths', False)
        self.top_paths_publisher = None
        self.top_costs_publisher = None
        if self.get_parameter('publish_top_paths').value:
            self.top_paths_publisher = self.create_publisher(
                MarkerArray, '/mppi/top_paths', 1)
            self.top_costs_publisher = self.create_publisher(
                String, '/mppi/top_paths/costs', 1)
        self.declare_parameter('publish_selected_paths', False)
        self.declare_parameter('path_frame', 'map')
        self._publish_waypoints()
        self.path_publisher = (
            PathPublisherProcess(self.config.H + 1, self.get_parameter('path_frame').value)
            if self.get_parameter('publish_selected_paths').value else None)

        self.get_logger().info(f'MPPI initialized on {track}; waiting for LiDAR extrinsics')

    def _publish_waypoints(self):
        """Publish static course once; retain it for transient-local late subscribers."""
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.waypoints_publisher = self.create_publisher(Marker, '/mppi/waypoints', qos)
        marker = Marker()
        marker.header.frame_id = self.get_parameter('path_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'waypoints'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color.g = 1.0
        marker.color.a = 1.0
        # Default zero lifetime keeps the marker visible without refreshing it.
        marker.points = [Point(x=float(x), y=float(y), z=0.0)
                         for x, y in self.waypoints[:, :2]]
        self.waypoints_publisher.publish(marker)

    def _create_tracker(self):
        """Optional host solver; default MPPI path needs no acados installation."""
        tracker = self.get_parameter('tracker').value
        if tracker == 'none':
            return None
        if tracker != 'acados':
            raise ValueError('tracker must be none or acados')
        if self.config.DYNAMICS == 'dynamic':
            raise ValueError('dynamic MPPI requires tracker:=none; acados assumes zero sideslip')
        from acados_tracker import AcadosTracker
        prebuilt_directory = self.get_parameter('tracker_prebuilt_dir').value
        if not prebuilt_directory:
            prebuilt_directory = os.path.join(
                get_package_prefix('mppi'), 'lib', 'mppi', 'acados_tracker')
        if not os.path.isfile(os.path.join(prebuilt_directory, 'acados_ocp.json')):
            raise RuntimeError(
                'Installed acados tracker missing; rebuild with '
                'colcon build --packages-select mppi --cmake-args -DMPPI_BUILD_ACADOS=ON')
        return AcadosTracker(
            self.config,
            max_steer_rate=self.get_parameter('tracker_max_steer_rate').value,
            export_directory=prebuilt_directory,
            generate=False)

    def _configure_device(self):
        """Select backend before allocating state; GPU requests never fall back."""
        self.declare_parameter('use_cpu', False)
        backend = 'cpu' if self.get_parameter('use_cpu').value else 'gpu'
        try:
            devices = jax.devices(backend)
        except RuntimeError as exc:
            raise RuntimeError(
                f'JAX {backend} backend unavailable. For CPU simulation, set '
                'ROS parameter use_cpu:=true.') from exc
        if not devices:
            raise RuntimeError(f'No JAX {backend} devices available')
        self.device = devices[0]
        # ROS callbacks run outside device contexts, so set the process default.
        jax.config.update('jax_default_device', self.device)
        self.get_logger().info(f'JAX controller device: {self.device}')

    def _warm_up(self):
        """Compile only after the actual sensor geometry is available."""
        observation = ControlObservation(
            np.zeros(self.config.NX, dtype=np.float32),
            np.zeros(self.grid.spec.num_beams, dtype=np.float32),
            np.float32(self.config.DT), np.bool_(False),
        )
        if self.tracker is None:
            result = control_step(
                self.control_state, observation, self.course, self.grid,
                self.config, export_path=self.path_publisher is not None,
                export_top_paths=getattr(self, 'top_paths_publisher', None) is not None)
        else:
            result = tracking_control_step(
                self.control_state, observation, self.course, self.grid, self.config,
                export_top_paths=getattr(self, 'top_paths_publisher', None) is not None)
        jax.block_until_ready(result)
        self.get_logger().info(f'MPPI ready: {self.config}')

    def odom_callback(self, msg):
        """Cache measured motion; actuator command state is attached at each tick."""
        try:
            self._latest_odom = measured_state(msg, self.config)
        except ValueError as exc:
            self._latest_odom = None
            self.get_logger().error(str(exc), throttle_duration_sec=1.0)
            self._publish_stop(self.get_clock().now())
            return
        self._odom_child_frame = msg.child_frame_id if msg.child_frame_id else 'odom'

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32).copy()
        self._latest_scan = (
            ranges,
            float(msg.angle_min),
            float(msg.angle_increment),
            float(msg.range_min),
            float(msg.range_max),
            msg.header.frame_id,
            msg.header.stamp,
        )

    def _lookup_lidar_pose(self, scan_frame, scan_stamp):
        """Return fixed car <- LiDAR [forward, left, yaw], or None if unavailable."""
        # TODO: Put actual value
        # if not self._odom_child_frame or not scan_frame:
        #     self.get_logger().warning('Missing odometry child or scan frame; stopping',
        #                               throttle_duration_sec=1.0)
        #     return None
        # t = transform.transform.translation
        # q = transform.transform.rotation
        # yaw = np.arctan2(
        #     2.0 * (q.w * q.z + q.x * q.y),
        #     1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        # )
        return (0.0, 0.0, 0.0)

    def _ensure_grid(self, scan_frame, scan_stamp):
        """Cache rigid sensor geometry; resolve it again if either frame changes."""
        frames = (self._odom_child_frame, scan_frame)
        if self._grid_frames == frames:
            return True
        lidar_pose = self._lookup_lidar_pose(scan_frame, scan_stamp)
        if lidar_pose is None:
            return False
        # Do not leave a previous drive command active during JAX compilation.
        self._publish_stop(self.get_clock().now())
        self.grid = og.OccupancyGrid.create(lidar_pose=lidar_pose, **self._grid_settings)
        self._warm_up()
        # Compilation pauses are not steady-state control-loop measurements.
        self._control_latencies.clear()
        self._control_periods.clear()
        self._last_control_start = None
        self._stats_window_start = None
        self._grid_frames = frames
        return True

    def control_callback(self):
        now = self.get_clock().now()
        observation = self._prepare_control_observation(now)
        if observation is None:
            return
        control_start = time.perf_counter()
        if self.tracker is None:
            result = control_step(
                self.control_state, observation, self.course, self.grid,
                self.config, export_path=self.path_publisher is not None,
                export_top_paths=getattr(self, 'top_paths_publisher', None) is not None)
            self._publish_control_result(result, observation, now)
        else:
            self._track_and_publish(observation, now)
        self._log_control_stats(control_start)

    def _track_and_publish(self, observation, now):
        """Plan on GPU, track on CPU, then publish one command."""
        result, nominal = tracking_control_step(
            self.control_state, observation, self.course, self.grid,
            self.config,
            export_top_paths=getattr(self, 'top_paths_publisher', None) is not None)
        self._publish_top_paths(result.top_paths, now)
        self.control_state, self.grid = result.state, result.grid
        states, controls = jax.device_get((nominal, result.state.U))
        try:
            tracked = self.tracker.solve(
                observation.x0, states, controls, self._last_tracker_action)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f'Tracker rejected command: {exc}', throttle_duration_sec=1.0)
            self._publish_stop(now)
            return
        self._last_tracker_action = tracked[0].copy()
        self._publish_drive(tracked[0], observation.x0, now)
        if self.path_publisher is not None:
            self.path_publisher.history.try_record(states[:, :2], now.nanoseconds)

    def _prepare_control_observation(self, now):
        """Validate sensors and compute host-side timing/reset policy."""
        odom = self._latest_odom
        scan = self._latest_scan

        if odom is None:
            return

        now_sec = now.nanoseconds * 1e-9
        x0 = self._observed_vehicle_state(odom)
        if not np.isfinite(x0).all():
            self.get_logger().error('Non-finite odometry; stopping', throttle_duration_sec=1.0)
            self._publish_stop(now)
            return

        if scan is None:
            self.get_logger().warning(
                'No scan received yet; publishing stop command',
                throttle_duration_sec=1.0,
            )
            self._publish_stop(now)
            return

        ranges, angle_min, angle_increment, range_min, range_max, scan_frame, scan_stamp = scan
        if not self._ensure_grid(scan_frame, scan_stamp):
            self._publish_stop(now)
            return

        elapsed = (
            self.config.DT
            if self._last_control_time is None
            else now_sec - self._last_control_time
        )
        self._last_control_time = now_sec
        elapsed = float(np.clip(elapsed, 0.0, self.config.DT))

        reset = (
            self.last_x0 is not None
            and np.hypot(*(x0[:2] - self.last_x0[:2])) > 1.0
        )
        self.last_x0 = x0.copy()
        return ControlObservation(
            x0.astype(np.float32), ranges, np.float32(elapsed), np.bool_(reset))

    def _observed_vehicle_state(self, odom):
        """Use latest published setpoints, not command state from an older odom tick."""
        if self.config.DYNAMICS == 'kinematic':
            return odom
        return np.concatenate((odom, self._last_drive_command))

    def _publish_control_result(self, result, observation, now):
        """Keep persistent arrays resident; read back only publication outputs."""
        self._publish_top_paths(result.top_paths, now)
        self.control_state = result.state
        self.grid = result.grid
        action, feasible = jax.device_get((result.action, result.feasible))
        if not feasible:
            self.get_logger().warning(
                'No model-feasible MPPI trajectory; stopping', throttle_duration_sec=1.0)
            self._publish_stop(now)
            return
        self._publish_drive(action, observation.x0, now)
        if self.path_publisher is not None:
            path_xy = jax.device_get(result.path_xy)
            self.path_publisher.history.try_record(path_xy, now.nanoseconds)

    def _publish_top_paths(self, top_paths, now):
        """Read back selected diagnostics, then publish rank-matched messages."""
        if top_paths is None:
            return
        paths = jax.device_get(top_paths)
        self.top_paths_publisher.publish(self._top_path_markers(paths, now))
        self.top_costs_publisher.publish(self._top_path_costs(paths, now))

    def _top_path_markers(self, paths, now):
        markers = MarkerArray()
        for rank, xy in enumerate(paths.xy):
            marker = Marker()
            marker.header.frame_id = self.get_parameter('path_frame').value
            marker.header.stamp = now.to_msg()
            marker.ns = 'top_paths'
            marker.id = rank
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.025
            marker.color.r = float(rank / max(1, len(paths.xy) - 1))
            marker.color.g = 1.0 - marker.color.r
            marker.color.a = 0.7
            marker.lifetime.sec = 1
            marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in xy]
            markers.markers.append(marker)
        return markers

    def _top_path_costs(self, paths, now):
        records = []
        for rank, total in enumerate(paths.totals):
            record = dict(rank=rank, total=float(total),
                          **{name: float(values[rank])
                             for name, values in paths.components.items()})
            # Strict JSON: non-finite diagnostics become null, not NaN literals.
            records.append({key: value if np.isfinite(value) else None
                            for key, value in record.items()})
        message = String()
        message.data = json.dumps(dict(stamp_ns=now.nanoseconds,
                                       frame_id=self.get_parameter('path_frame').value,
                                       paths=records), allow_nan=False)
        return message

    def _publish_drive(self, action, x0, now):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = now.to_msg()
        drive_msg.drive.steering_angle = float(action[1])

        previous_speed = x0[7] if self.config.DYNAMICS == 'dynamic' else x0[3]
        target_speed = previous_speed + float(action[0]) * self.config.DT
        target_speed = np.clip(
            target_speed,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
        )

        drive_msg.drive.speed = float(target_speed)
        self._last_drive_command = np.array([float(action[1]), float(target_speed)])
        self.drive_publisher.publish(drive_msg)

    def _log_control_stats(self, start):
        """Wall-clock hot-path timings, synchronized by the selected-action readback.

        Includes drive publication and optional selected-XY readback/copy. Excludes input
        checks, TF lookup, warmup, skipped ticks, and this summary's logging cost.
        """
        end = time.perf_counter()
        self._control_latencies.append(end - start)
        if self._last_control_start is not None:
            self._control_periods.append(start - self._last_control_start)
        self._last_control_start = start
        if self._stats_window_start is None:
            self._stats_window_start = start
        if end - self._stats_window_start < 1.0:
            return

        latency_ms = np.asarray(self._control_latencies) * 1e3
        period_ms = np.asarray(self._control_periods) * 1e3
        overruns = np.count_nonzero(latency_ms > self.config.DT * 1e3)
        period_stats = (
            f'{period_ms.mean():.2f}/{np.percentile(period_ms, 95):.2f}/{period_ms.max():.2f}'
            if period_ms.size else 'n/a'
        )
        self.get_logger().info(
            f'Control loop: n={latency_ms.size}, '
            f'latency ms mean/p95/max={latency_ms.mean():.2f}/'
            f'{np.percentile(latency_ms, 95):.2f}/{latency_ms.max():.2f}, '
            f'period ms mean/p95/max={period_stats}, '
            f'latency > {self.config.DT * 1e3:.1f} ms: {overruns}/{latency_ms.size}, '
            f'publish_selected_paths={self.path_publisher is not None}'
        )
        self._control_latencies.clear()
        self._control_periods.clear()
        self._stats_window_start = end

    def _publish_stop(self, stamp):
        self._last_tracker_action = np.zeros(2)
        self._last_drive_command = np.zeros(2)
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp.to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(msg)

    def destroy_node(self):
        """Release the optional publisher process before shutting down ROS."""
        if self.path_publisher is not None:
            self.path_publisher.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    print("MPPI Initialized")
    mppi_node = MPPI()
    try:
        rclpy.spin(mppi_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        mppi_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
