#!/usr/bin/env python3
from dataclasses import replace
from controller_config import MPPIConfig
from functools import partial
from typing import NamedTuple
import time

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from utils import calc_interpolated_ref_trajectory_gpu, shift_sequence
import occupancy_grid as og

import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from ament_index_python.packages import get_package_prefix, get_package_share_directory


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


def cost(X, U, ref, cfg, grid):
    error = X[1:] - ref[1:]

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
    obs_block_cost = jnp.where(~obs_blocked, blocked_penalty, 0.0)

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

    return state_cost + input_cost + input_difference_cost + obs_block_cost


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


def mppi_update(key, U, U_std, x0, ref, cfg, grid):
    U_samples = sample_controls(key, U, U_std, cfg)

    # 2. Simulate: roll out all T sequences from x0
    X_samples = rollout(x0, U_samples, cfg)

    C = jax.vmap(cost, in_axes=(0, 0, None, None, None))(
        X_samples, U_samples, ref, cfg, grid
    )
    S = jnp.flip(jnp.cumsum(jnp.flip(C, axis=1), axis=1), axis=1)

    # 4. Weight: normalize S[:, t] to [0, 1] over the samples, then softmax
    S_min = jnp.min(S, axis=0, keepdims=True)
    S_max = jnp.max(S, axis=0, keepdims=True)
    S_normalized = (S - S_min) / (S_max - S_min + 1e-6)
    weights = jax.nn.softmax(-S_normalized / cfg.LAMBDA, axis=0)

    # 5. Update: U_new[t] = sum_k w[k, t] * u_k[t]
    U_new = jnp.sum(weights[:, :, jnp.newaxis] * U_samples, axis=0)

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

    return U_new, U_std_new, X_samples, weights


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


class Visualization(NamedTuple):
    reference: jax.Array
    samples: jax.Array
    weights: jax.Array
    plan: jax.Array


class ControlResult(NamedTuple):
    state: ControllerState
    grid: og.OccupancyGrid
    action: jax.Array
    visualization: Visualization | None


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
        v0=x0[3], max_accel=cfg.MAX_ACCEL)


def optimize_controls(state, x0, reference, grid, cfg):
    """Own sampling RNG advancement and fixed/adaptive covariance policy."""
    key, sample_key = jax.random.split(state.key)
    U, updated_std, samples, weights = mppi_update(
        sample_key, state.U, state.U_std, x0, reference, cfg, grid)
    U_std = updated_std if cfg.COV_UPDATE else state.U_std
    return OptimizationResult(ControllerState(key, U, U_std), samples, weights)


def select_control_outputs(solution, x0, reference, grid, cfg, emit_viz):
    """Select action and bounded debug payload before crossing to the host."""
    visualization = None
    if emit_viz:
        visualization = Visualization(
            reference, solution.samples[:50], solution.weights[:50, 0],
            simulate(x0, solution.state.U, cfg))
    return ControlResult(
        solution.state, grid, solution.state.U[0], visualization)


@partial(jax.jit, static_argnames=("cfg", "emit_viz"))
def control_step(state, observation, course, grid, cfg, *, emit_viz=False):
    """One compiled dispatch; helpers preserve domain responsibility boundaries."""
    state = prepare_warm_start(state, observation, cfg)
    grid = grid.from_scan(observation.ranges)
    reference = build_reference(observation.x0, course, cfg)
    solution = optimize_controls(state, observation.x0, reference, grid, cfg)
    return select_control_outputs(
        solution, observation.x0, reference, grid, cfg, emit_viz)


@partial(jax.jit, static_argnames=("cfg", "emit_viz"))
def tracking_control_step(state, observation, course, grid, cfg, *, emit_viz=False):
    """Keep optimization on device; export only selected nominal, never sample batch."""
    result = control_step(
        state, observation, course, grid, cfg, emit_viz=emit_viz)
    nominal = simulate(observation.x0, result.state.U, cfg)
    return result, nominal


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
        for field in ('MAX_SPEED', 'LAMBDA', 'INTEGRATOR', 'SAMPLING', 'COV_UPDATE'):
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
        self._last_visualization_time = 0.0
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
            num_beams=819,
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
            track + '.csv'
        )
        self.waypoints = np.loadtxt(
            waypoint_path, delimiter=',', skiprows=1
        )
        self.waypoints[:, 3] = np.minimum(self.waypoints[:, 3], self.config.MAX_SPEED)
        d = np.diff(self.waypoints[:, :2], axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        yaw = np.append(yaw, yaw[-1])
        # Fixed course geometry: compute once, upload once.
        self.course = jax.device_put(np.column_stack((
            self.waypoints[:, :2], yaw, self.waypoints[:, 3],
        )).astype(np.float32), self.device)

        self.declare_parameter('visualize', True)
        self.visualize = self.get_parameter('visualize').value
        if self.visualize:
            self.marker_publisher = self.create_publisher(MarkerArray, '/mppi/markers', 1)

        self.get_logger().info(f'MPPI initialized on {track}; waiting for LiDAR extrinsics')

    def _create_tracker(self):
        """Optional host solver; default MPPI path needs no acados installation."""
        tracker = self.get_parameter('tracker').value
        if tracker == 'none':
            return None
        if tracker != 'acados':
            raise ValueError('tracker must be none or acados')
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
        compute = tracking_control_step if self.tracker is not None else control_step
        for emit_viz in ((False, True) if self.visualize else (False,)):
            result = compute(
                self.control_state, observation, self.course, self.grid,
                self.config, emit_viz=emit_viz)
            jax.block_until_ready(result)
        self.get_logger().info(f'MPPI ready: {self.config}')

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        velocity = msg.twist.twist.linear

        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        speed = np.hypot(velocity.x, velocity.y)

        state = np.array(
            [position.x, position.y, yaw, speed],
            dtype=np.float64,
        )
        self._latest_odom = state
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
        if not self._odom_child_frame or not scan_frame:
            self.get_logger().warning('Missing odometry child or scan frame; stopping',
                                      throttle_duration_sec=1.0)
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self._odom_child_frame, scan_frame, Time.from_msg(scan_stamp))
        except TransformException as exc:
            self.get_logger().warning(f'LiDAR extrinsics unavailable; stopping: {exc}',
                                      throttle_duration_sec=1.0)
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (t.x, t.y, yaw)

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
                self.config, emit_viz=self._visualization_due(now))
            self._publish_control_result(result, observation, now)
        else:
            self._track_and_publish(observation, now)
        self._log_control_stats(control_start)

    def _track_and_publish(self, observation, now):
        """Plan on GPU, track on CPU, then publish one command."""
        result, nominal = tracking_control_step(
            self.control_state, observation, self.course, self.grid,
            self.config, emit_viz=self._visualization_due(now))
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
        if result.visualization is not None:
            self._last_visualization_time = now.nanoseconds * 1e-9
            self.publish_markers(jax.device_get(result.visualization))

    def _visualization_due(self, now):
        return (self.visualize
                and now.nanoseconds * 1e-9 - self._last_visualization_time >= 0.2)

    def _prepare_control_observation(self, now):
        """Validate sensors and compute host-side timing/reset policy."""
        odom = self._latest_odom
        scan = self._latest_scan

        if odom is None:
            return

        now_sec = now.nanoseconds * 1e-9
        x0 = odom

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

    def _publish_control_result(self, result, observation, now):
        """Keep persistent arrays resident; read back only publication outputs."""
        self.control_state = result.state
        self.grid = result.grid
        action, visualization = jax.device_get((result.action, result.visualization))
        self._publish_drive(action, observation.x0, now)
        if visualization is not None:
            self._last_visualization_time = now.nanoseconds * 1e-9
            self.publish_markers(visualization)

    def _publish_drive(self, action, x0, now):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = now.to_msg()
        drive_msg.drive.steering_angle = float(action[1])

        target_speed = x0[3] + float(action[0]) * self.config.DT
        target_speed = np.clip(
            target_speed,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
        )

        drive_msg.drive.speed = float(target_speed)
        self.drive_publisher.publish(drive_msg)

    def _log_control_stats(self, start):
        """Wall-clock hot-path timings, synchronized by the selected-action readback.

        Includes ROS drive publication and optional visualization. Excludes input
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
            f'visualize={self.visualize}'
        )
        self._control_latencies.clear()
        self._control_periods.clear()
        self._stats_window_start = end

    def _publish_stop(self, stamp):
        self._last_tracker_action = np.zeros(2)
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp.to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(msg)

    def publish_markers(self, visualization):
        """Publish host-only reference, optimized plan, and sampled rollouts."""
        ref, X_samples, w, plan = visualization
        w = w / (w.max() + 1e-12)

        def line(i, points, r, g, b, width):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'mppi'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.scale.x = width
            marker.color.r, marker.color.g, marker.color.b = float(r), float(g), float(b)
            marker.color.a = 1.0
            marker.points = [Point(x=float(p[0]), y=float(p[1])) for p in points]
            return marker

        markers = MarkerArray()
        markers.markers.append(line(0, ref, 0.0, 1.0, 0.0, 0.05))
        markers.markers.append(line(1, plan, 1.0, 0.0, 0.0, 0.05))
        for k in range(len(X_samples)):
            markers.markers.append(
                line(2 + k, X_samples[k], 1.0 - w[k], 1.0 - w[k], 1.0, 0.01))
        self.marker_publisher.publish(markers)


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
