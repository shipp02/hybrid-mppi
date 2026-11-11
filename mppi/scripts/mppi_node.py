#!/usr/bin/env python3
from dataclasses import dataclass, replace
from functools import partial
import functools

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from utils import calc_interpolated_ref_trajectory
import occupancy_grid as og

import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from ament_index_python.packages import get_package_share_directory


@dataclass(frozen=True)
class MPPIConfig:
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 20  # horizon length
    DT: float = 0.05  # time step [s]
    INTEGRATOR: str = "rk4"  # "euler" or "rk4" (bonus)
    SAMPLING: str = "truncated"  # "clip", or bonus: "truncated" (normal) or "uniform"
    COV_UPDATE: bool = True  # bonus: adapt the sampling std of each step

    # ---------------------------------------------------
    # TODO: you may need to tune the following parameters
    LAMBDA: float = 0.1  # temperature
    U_STD: tuple = (1.5, 0.15)  # std of the sampled inputs [a, delta]
    NOISE_RHO: float = 0.9  # correlation of the sampled noise from one step to the next
    U_STD_MIN: tuple = (0.3, 0.03)  # covariance update: floor of the std
    COV_ALPHA: float = 0.5  # covariance update: blend of the new std
    # cost() weights the position error along the reference (lag) and across it (lateral)
    Q: tuple = (1.0, 20.0, 2.0, 1.0)  # state error cost [lag, lateral, yaw, v]
    Qf: tuple = (2.0, 40.0, 4.0, 1.0)  # final state error cost [lag, lateral, yaw, v]
    R: tuple = (0.01, 0.01)  # input cost [a, delta]
    Rd: tuple = (0.01, 1.0)  # input difference cost [a, delta]
    # ---------------------------------------------------

    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # minimum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_SPEED: float = 0.0  # minimum speed [m/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]


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


def cost(X, U, ref, cfg, grid_cfg, grid_precomps, grid_state):
    error = X[1:] - ref[1:]

    yaw_error = jnp.arctan2(
        jnp.sin(error[:, 2]),
        jnp.cos(error[:, 2])
    )

    error = error.at[:, 2].set(yaw_error)
    
    # Map rollout trajectory into local car frame for collision/distance evaluation
    local_traj = trajectory_to_car_frame(X[1:], X[0, 0], X[0, 1], X[0, 2])
    
    # Evaluate obstacle proximity using the precomputed JAX occupancy grid functions
    obs_blocked = og.is_free_poses(grid_cfg, grid_state, local_traj)
    obs_dist = og.find_distances(grid_cfg, grid_state, local_traj)
    
    # Example penalty weights for obstacles and collisions
    # obs_cost = jnp.sum(jnp.where(obs_dist < grid_cfg.inflation_radius, 1e1, 0.0))
    obs_cost = 0.0
    obs_block_cost = jnp.sum(jnp.where(~obs_blocked, 1e1, 0.0))

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

    return state_cost + input_cost + input_difference_cost + obs_cost + obs_block_cost


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


def mppi_update(key, U, U_std, x0, ref, cfg, grid_cfg, grid_precomps, grid_state):
    U_samples = sample_controls(key, U, U_std, cfg)

    # 2. Simulate: roll out all T sequences from x0
    X_samples = rollout(x0, U_samples, cfg)

    C = jax.vmap(cost, in_axes=(0, 0, None, None, None, None, None))(
        X_samples, U_samples, ref, cfg, grid_cfg, grid_precomps, grid_state
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


@partial(jax.jit, static_argnames=("cfg", "grid_cfg", "grid_precomps"))
def mppi_step(key, U, x0, ref, cfg, grid_cfg, grid_precomps, grid_state):
    U_std = jnp.broadcast_to(jnp.array(cfg.U_STD), (cfg.H, cfg.NU))
    U_new, _, X_samples, weights = mppi_update(key, U, U_std, x0, ref, cfg, grid_cfg, grid_precomps, grid_state)
    return U_new, X_samples, weights


@partial(jax.jit, static_argnames=("cfg", "grid_cfg", "grid_precomps"))
def mppi_step_cov(key, U, U_std, x0, ref, cfg, grid_cfg, grid_precomps, grid_state):
    return mppi_update(key, U, U_std, x0, ref, cfg, grid_cfg, grid_precomps, grid_state)


class MPPI(Node):
    """
    Implement Kinematic MPPI on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mppi_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'levine')
        self.declare_parameter('odom_topic', '')
        track = self.get_parameter('track').value
        
        overrides = {}
        for field in ('MAX_SPEED', 'LAMBDA', 'INTEGRATOR', 'SAMPLING', 'COV_UPDATE'):
            default = getattr(MPPIConfig, field)
            self.declare_parameter(field.lower(), default)
            overrides[field] = type(default)(self.get_parameter(field.lower()).value)
        self.config = replace(MPPIConfig(), **overrides)

        self.U = jnp.zeros((self.config.H, self.config.NU))
        self.U_std = jnp.broadcast_to(
            jnp.array(self.config.U_STD), (self.config.H, self.config.NU))
        self.key = jax.random.PRNGKey(0)
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

        # Initialize grid configuration and precomputations
        self.grid_cfg = og.GridConfig(
            width=3.0,
            height=7.0,
            middle=(1.5, 1.0),
            resolution=0.05,
            inflation_radius=0.15,
            num_beams=819,
            distance_cap=0.25,
        )
        self.grid_precomps = None
        self._grid_frames = None

        rows_n, cols_n = self.grid_cfg.shape
        cells = jnp.full((rows_n, cols_n), og.UNKNOWN, dtype=jnp.uint8)
        free_dist = jnp.full((rows_n, cols_n), 1.0, dtype=jnp.float32)
        self.grid_state = og.GridState(cells=cells, free_distance=free_dist)

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

        self.declare_parameter('visualize', True)
        self.visualize = self.get_parameter('visualize').value
        if self.visualize:
            self.marker_publisher = self.create_publisher(MarkerArray, '/mppi/markers', 1)

        self.get_logger().info(f'MPPI initialized on {track}; waiting for LiDAR extrinsics')

    def _warm_up(self):
        """Compile only after the actual sensor geometry is available."""
        dummy_ranges = jnp.zeros(self.grid_cfg.num_beams)
        dummy_grid_state = og.update(
            self.grid_cfg, self.grid_precomps, dummy_ranges)
        mppi_step(self.key, self.U, np.zeros(self.config.NX),
                  np.zeros((self.config.H + 1, self.config.NX)), self.config, self.grid_cfg, self.grid_precomps, dummy_grid_state)
        mppi_step_cov(self.key, self.U, self.U_std, np.zeros(self.config.NX),
                      np.zeros((self.config.H + 1, self.config.NX)), self.config, self.grid_cfg, self.grid_precomps, dummy_grid_state)
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
        self._odom_child_frame = msg.child_frame_id

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

    def _ensure_grid_precomputations(self, scan_frame, scan_stamp):
        """Cache rigid sensor geometry; resolve it again if either frame changes."""
        frames = (self._odom_child_frame, scan_frame)
        if self._grid_frames == frames:
            return True
        lidar_pose = self._lookup_lidar_pose(scan_frame, scan_stamp)
        if lidar_pose is None:
            return False
        # Do not leave a previous drive command active during JAX compilation.
        self._publish_stop(self.get_clock().now())
        self.grid_precomps = og.GridPrecomputations(self.grid_cfg, lidar_pose)
        self._warm_up()
        self._grid_frames = frames
        return True

    def control_callback(self):
        odom = self._latest_odom
        scan = self._latest_scan

        if odom is None:
            return

        now = self.get_clock().now()
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

        if not self._ensure_grid_precomputations(scan_frame, scan_stamp):
            self._publish_stop(now)
            return

        grid_state = og.update(self.grid_cfg, self.grid_precomps, jnp.array(ranges))
        self.grid_state = grid_state

        d = np.diff(self.waypoints[:, :2], axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        yaw = np.append(yaw, yaw[-1])
        ref = self.calc_ref_trajectory(
            x0,
            self.waypoints[:, 0],
            self.waypoints[:, 1],
            yaw,
            self.waypoints[:, 3],
        )

        elapsed = (
            self.config.DT
            if self._last_control_time is None
            else now_sec - self._last_control_time
        )
        self._last_control_time = now_sec
        elapsed = float(np.clip(elapsed, 0.0, self.config.DT))

        if (
            self.last_x0 is not None
            and np.hypot(*(x0[:2] - self.last_x0[:2])) > 1.0
        ):
            self.U = jnp.zeros((self.config.H, self.config.NU))
            self.U_std = jnp.broadcast_to(
                jnp.array(self.config.U_STD),
                (self.config.H, self.config.NU),
            )

        self.last_x0 = x0.copy()
        self.U = self.shift(self.U, elapsed, np.asarray(self.U[-1]))

        if self.config.COV_UPDATE:
            self.U_std = self.shift(
                self.U_std,
                elapsed,
                np.array(self.config.U_STD),
            )

        self.key, subkey = jax.random.split(self.key)

        if self.config.COV_UPDATE:
            self.U, self.U_std, X_samples, weights = mppi_step_cov(
                subkey, self.U, self.U_std, x0, ref, self.config, self.grid_cfg, self.grid_precomps, self.grid_state
            )
        else:
            self.U, X_samples, weights = mppi_step(
                subkey, self.U, x0, ref, self.config, self.grid_cfg, self.grid_precomps, self.grid_state
            )

        # TODO: publish drive message: steering delta_0, speed v + a_0 * DT
        #       (float() turns a JAX array into a number)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = now.to_msg()
        drive_msg.drive.steering_angle = float(self.U[0, 1])

        target_speed = x0[3] + float(self.U[0, 0]) * self.config.DT
        target_speed = np.clip(
            target_speed,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
        )

        drive_msg.drive.speed = float(target_speed)
        self.drive_publisher.publish(drive_msg)

        if self.visualize and now_sec - self._last_visualization_time >= 0.2:
            self._last_visualization_time = now_sec
            self.publish_markers(x0, ref, X_samples, weights)

    def _publish_stop(self, stamp):
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp.to_msg()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(msg)

    def publish_markers(self, x0, ref, X_samples, weights, n_samples=50):
        """
        Debugging only: the reference (green), the plan of self.U (red) and
        n_samples sampled rollouts, from white (low weight) to blue (high weight)
        """
        X_samples = np.asarray(X_samples[:n_samples])
        w = np.asarray(weights[:n_samples, 0])
        w = w / (w.max() + 1e-12)
        plan = np.asarray(simulate(jnp.asarray(x0), self.U, self.config))

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

    def shift(self, seq, elapsed, fill):
        """
        Warm start: seq[k] is planned for time k * DT, so after `elapsed` seconds
        the new seq[k] is the old one at k * DT + elapsed, linearly interpolated.
        With elapsed = DT, this drops the first step and appends `fill`.
        :param seq: sequence over the horizon, shape (H, NU)
        :param fill: value after the end of the horizon, shape (NU,)
        """
        seq = np.asarray(seq)
        grid = np.arange(self.config.H) * self.config.DT
        return jnp.array(np.column_stack([
            np.interp(grid + elapsed, grid, seq[:, j], right=fill[j])
            for j in range(seq.shape[1])
        ]))

    def calc_ref_trajectory(self, x0, cx, cy, cyaw, cv):
        """
        Calculate interpolated reference trajectory for H steps: [x, y, yaw, v]

        :param x0: current state [p_x, p_y, yaw, v]
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param cv: speed profile
        :return: reference trajectory ref (H+1 x NX)
        """
        return calc_interpolated_ref_trajectory(
            x0[0], x0[1],
            cx, cy, cyaw, cv,
            self.config.DT,
            self.config.H,
            v0=x0[3],
            max_accel=self.config.MAX_ACCEL
        )


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
