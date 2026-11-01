#!/usr/bin/env python3
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from utils import calc_interpolated_ref_trajectory

# TODO CHECK: include needed ROS msg type headers and libraries
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory


@dataclass(frozen=True)
class MPPIConfig:
    # frozen so jax.jit can take it as a static argument: use tuples, not arrays
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


def cost(X, U, ref, cfg):
    """
    Cost of each step of one trajectory (README Section III.a): c[t] is the cost
    of applying u_t and reaching x_{t+1}
        c[t] = (x_{t+1} - ref_{t+1})^T Q (x_{t+1} - ref_{t+1})   (Qf for the last step)
             + u_t^T R u_t
             + (u_t - u_{t-1})^T Rd (u_t - u_{t-1})              (from t = 1)
    :param X: states x_0 ... x_H, shape (H+1, NX)
    :param U: inputs u_0 ... u_{H-1}, shape (H, NU)
    :param ref: reference states, shape (H+1, NX)
    :return: c, shape (H,)
    """
    # TODO: Q, Qf, R and Rd are diagonal: e^T Q e = sum(jnp.array(cfg.Q) * e**2)
    # TODO: wrap the heading error to [-pi, pi]
    # TODO: add your own penalties if you want, e.g. for going over cfg.MAX_SPEED
    error = X[1:] - ref[1:]

    yaw_error = jnp.arctan2(
        jnp.sin(error[:, 2]),
        jnp.cos(error[:, 2])
    )

    error = error.at[:, 2].set(yaw_error)

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

    return state_cost + input_cost + input_difference_cost


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


def mppi_update(key, U, U_std, x0, ref, cfg):
    """
    Steps 1 to 5 of MPPI (README Section IV), shared by mppi_step and mppi_step_cov
    :return: U_new (H, NU), U_std_new (H, NU), X_samples (T, H+1, NX), weights (T, H)
    """
    # 1. Sample: T control sequences around U, inside the input bounds
    U_samples = sample_controls(key, U, U_std, cfg)

    # 2. Simulate: roll out all T sequences from x0
    X_samples = rollout(x0, U_samples, cfg)

    # 3. Score: the cost of each step of each sample, then the cost-to-go
    #    S[k, t] = c[k, t] + ... + c[k, H-1]
    C = jax.vmap(cost, in_axes=(0, 0, None, None))(
        X_samples, U_samples, ref, cfg
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
def mppi_step(key, U, x0, ref, cfg):
    """
    One MPPI update (README Section IV) with the fixed std cfg.U_STD
    :param key: JAX random key, a new one at every call
    :param U: mean control sequence, already warm started, shape (H, NU)
    :param x0: current state, shape (NX,)
    :param ref: reference states, shape (H+1, NX)
    :param cfg: MPPIConfig
    :return: the new mean control sequence, shape (H, NU),
             the sampled trajectories (for visualization), shape (T, H+1, NX),
             and the weights of the samples at each step, shape (T, H)
    """
    U_std = jnp.broadcast_to(jnp.array(cfg.U_STD), (cfg.H, cfg.NU))
    U_new, _, X_samples, weights = mppi_update(key, U, U_std, x0, ref, cfg)
    return U_new, X_samples, weights


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step_cov(key, U, U_std, x0, ref, cfg):
    """
    Bonus: one MPPI update that also updates the sampling std of each step
    :param U_std: std of each input at each step, already warm started, shape (H, NU)
    :return: U_new (H, NU), U_std_new (H, NU), X_samples (T, H+1, NX), weights (T, H)
    """
    return mppi_update(key, U, U_std, x0, ref, cfg)


class MPPI(Node):
    """
    Implement Kinematic MPPI on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mppi_node')
        # launch/levine_launch.py sets track:=levine, launch/spielberg_launch.py track:=spielberg
        self.declare_parameter('track', 'levine')
        track = self.get_parameter('track').value

        # TODO: create ROS subscribers and publishers
        #       use MPPI as a tracker (similar to pure pursuit)
        # queue of 1: always run MPPI from the newest pose, never a stale one
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            1
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # TODO: load the waypoints of `track` from the package's installed waypoints/ folder:
        #       os.path.join(get_package_share_directory('mppi'), 'waypoints', <your csv>)
        #       (ament_index_python.packages). A path like /home/you/... only exists on your laptop.
        waypoint_path = os.path.join(
            get_package_share_directory('mppi'),
            'waypoints',
            track + '.csv'
        )
        # columns x, y, yaw, v
        self.waypoints = np.loadtxt(
            waypoint_path,
            delimiter=',',
            skiprows=1
        )

        # per-track overrides of MPPIConfig from the launch file, e.g. 'max_speed': 6.0
        overrides = {}
        for field in ('MAX_SPEED', 'LAMBDA', 'INTEGRATOR', 'SAMPLING', 'COV_UPDATE'):
            default = getattr(MPPIConfig, field)
            self.declare_parameter(field.lower(), default)
            overrides[field] = type(default)(self.get_parameter(field.lower()).value)
        self.config = replace(MPPIConfig(), **overrides)
        # the reference must not ask for more than the car is allowed to do
        self.waypoints[:, 3] = np.minimum(self.waypoints[:, 3], self.config.MAX_SPEED)

        self.U = jnp.zeros((self.config.H, self.config.NU))  # mean control sequence
        self.U_std = jnp.broadcast_to(  # sampling std of each step (covariance update)
            jnp.array(self.config.U_STD), (self.config.H, self.config.NU))
        self.key = jax.random.PRNGKey(0)
        self.last_stamp = None  # stamp of the last odometry, for the warm start
        self.last_x0 = None  # state at the last odometry

        # debugging only: sampled rollouts and the plan in RViz, on /mppi/markers
        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter('visualize').value
        if self.visualize:
            self.marker_publisher = self.create_publisher(MarkerArray, '/mppi/markers', 1)

        # jax.jit compiles mppi_step the first time it is called, which takes a
        # few seconds: call it once here so the car does not wait for it
        mppi_step(self.key, self.U, np.zeros(self.config.NX),
                  np.zeros((self.config.H + 1, self.config.NX)), self.config)
        mppi_step_cov(self.key, self.U, self.U_std, np.zeros(self.config.NX),
                      np.zeros((self.config.H + 1, self.config.NX)), self.config)
        self.get_logger().info(f'MPPI ready on {track}: {self.config}')

    def odom_callback(self, odom_msg):
        # TODO: extract the state x0 = [p_x, p_y, yaw, v] from the ROS msg
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        velocity = odom_msg.twist.twist.linear

        yaw = np.arctan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)
        )

        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)

        x0 = np.array([
            position.x,
            position.y,
            yaw,
            speed
        ])

        # TODO: Calculate the reference trajectory for the next H steps
        #       with the current vehicle state.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref = self.calc_ref_trajectory(
            x0,
            self.waypoints[:, 0],
            self.waypoints[:, 1],
            self.waypoints[:, 2],
            self.waypoints[:, 3]
        )

        # TODO: warm start: the first input of self.U was applied at the last
        #       step, so shift self.U one step forward (drop the first input
        #       and repeat the last one)
        #       Odometry comes much faster than DT (250 Hz in the simulator), so
        #       shift by the time since the last callback instead of a whole step
        stamp = odom_msg.header.stamp.sec + 1e-9 * odom_msg.header.stamp.nanosec
        elapsed = self.config.DT if self.last_stamp is None else stamp - self.last_stamp
        elapsed = float(np.clip(elapsed, 0.0, self.config.DT))
        self.last_stamp = stamp
        if self.last_x0 is not None and np.hypot(*(x0[:2] - self.last_x0[:2])) > 1.0:
            # the car jumped (new start pose): the old plan is meaningless here
            self.U = jnp.zeros((self.config.H, self.config.NU))
            self.U_std = jnp.broadcast_to(
                jnp.array(self.config.U_STD), (self.config.H, self.config.NU))
        self.last_x0 = x0
        self.U = self.shift(self.U, elapsed, np.asarray(self.U[-1]))

        if self.config.COV_UPDATE:
            # shift the std like U; the new last step starts from the prior std
            self.U_std = self.shift(self.U_std, elapsed, np.array(self.config.U_STD))

        # a new random key at every step, so MPPI draws new samples
        self.key, subkey = jax.random.split(self.key)
        # TODO: run MPPI, keep the new mean control sequence in self.U
        if self.config.COV_UPDATE:
            self.U, self.U_std, X_samples, weights = mppi_step_cov(
                subkey,
                self.U,
                self.U_std,
                x0,
                ref,
                self.config
            )
        else:
            self.U, X_samples, weights = mppi_step(
                subkey,
                self.U,
                x0,
                ref,
                self.config
            )

        # TODO: publish drive message: steering delta_0, speed v + a_0 * DT
        #       (float() turns a JAX array into a number)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(self.U[0, 1])

        target_speed = x0[3] + float(self.U[0, 0]) * self.config.DT
        target_speed = np.clip(
            target_speed,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED
        )

        drive_msg.drive.speed = float(target_speed)
        self.drive_publisher.publish(drive_msg)

        # TODO (optional, debugging only): visualize the reference, the trajectory
        #       of self.U and a few sampled trajectories, colored by their weight
        if self.visualize:
            self.publish_markers(x0, ref, X_samples, weights)

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
        pass  # Ctrl-C: stop quietly
    finally:
        mppi_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()