#!/usr/bin/env python3
from dataclasses import dataclass
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
from ament_index_python.packages import get_package_share_directory


@dataclass(frozen=True)
class MPPIConfig:
    # frozen so jax.jit can take it as a static argument: use tuples, not arrays
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 20  # horizon length
    DT: float = 0.05  # time step [s]
    INTEGRATOR: str = "euler"  # "euler" or "rk4" (bonus)

    # ---------------------------------------------------
    # TODO: you may need to tune the following parameters
    LAMBDA: float = 0.01  # temperature
    U_STD: tuple = (1.0, 0.1)  # std of the sampled inputs [a, delta]
    Q: tuple = (5.0, 5.0, 1.0, 1.0)  # state error cost [x, y, yaw, v]
    Qf: tuple = (5.0, 5.0, 1.0, 1.0)  # final state error cost [x, y, yaw, v]
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
        return step_rk4(x, u, cfg)
    return step_euler(x, u, cfg)


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


@partial(jax.jit, static_argnames=("cfg",))
def mppi_step(key, U, x0, ref, cfg):
    """
    One MPPI update (README Section IV)
    :param key: JAX random key, a new one at every call
    :param U: mean control sequence, already warm started, shape (H, NU)
    :param x0: current state, shape (NX,)
    :param ref: reference states, shape (H+1, NX)
    :param cfg: MPPIConfig
    :return: the new mean control sequence, shape (H, NU),
             the sampled trajectories (for visualization), shape (T, H+1, NX),
             and the weights of the samples at each step, shape (T, H)
    For the covariance update (bonus), you may change the inputs and outputs.
    """
    # --------------------------------------------------------
    # TODO 1. Sample: T control sequences around U with noise of std cfg.U_STD
    #         (jax.random.normal(key, shape)), clipped to the input bounds.
    #         Bonus: sample from a truncated distribution instead of clipping
    noise = jax.random.normal(
        key,
        (cfg.T, cfg.H, cfg.NU)
    )
    noise = noise * jnp.array(cfg.U_STD)

    U_samples = U[jnp.newaxis, :, :] + noise

    U_samples = U_samples.at[:, :, 0].set(
        jnp.clip(U_samples[:, :, 0], cfg.MIN_ACCEL, cfg.MAX_ACCEL)
    )
    U_samples = U_samples.at[:, :, 1].set(
        jnp.clip(U_samples[:, :, 1], cfg.MIN_STEER, cfg.MAX_STEER)
    )

    # TODO 2. Simulate: roll out all T sequences from x0
    X_samples = rollout(x0, U_samples, cfg)

    # TODO 3. Score: the cost of each step of each sample, c of shape (T, H)
    #         (vectorize cost() with jax.vmap), then the cost-to-go
    #         S[k, t] = c[k, t] + ... + c[k, H-1] (cumsum of the reversed steps)
    C = jax.vmap(cost, in_axes=(0, 0, None, None))(
        X_samples, U_samples, ref, cfg
    )

    S = jnp.flip(
        jnp.cumsum(jnp.flip(C, axis=1), axis=1),
        axis=1
    )

    # TODO 4. Weight: for each step t, normalize S[:, t] to [0, 1] over the
    #         samples, (S - min) / (max - min + 1e-6), then take its softmax
    #         over the samples, w[:, t] = softmax(-S[:, t] / cfg.LAMBDA)
    S_min = jnp.min(S, axis=0, keepdims=True)
    S_max = jnp.max(S, axis=0, keepdims=True)

    S_normalized = (S - S_min) / (S_max - S_min + 1e-6)

    weights = jax.nn.softmax(
        -S_normalized / cfg.LAMBDA,
        axis=0
    )

    # TODO 5. Update: U_new[t] = sum over the samples k of w[k, t] * u_k[t]
    #         Bonus: also update the covariance
    U_new = jnp.sum(
        weights[:, :, jnp.newaxis] * U_samples,
        axis=0
    )

    # --------------------------------------------------------
    return U_new, X_samples, weights


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
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
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
        self.waypoints = np.loadtxt(
            waypoint_path,
            delimiter=',',
            skiprows=1
        )

        self.config = MPPIConfig()
        self.U = jnp.zeros((self.config.H, self.config.NU))  # mean control sequence
        self.key = jax.random.PRNGKey(0)

        # jax.jit compiles mppi_step the first time it is called, which takes a
        # few seconds: call it once here so the car does not wait for it
        mppi_step(self.key, self.U, np.zeros(self.config.NX),
                  np.zeros((self.config.H + 1, self.config.NX)), self.config)

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
        self.U = jnp.concatenate(
            (self.U[1:], self.U[-1:]),
            axis=0
        )

        # a new random key at every step, so MPPI draws new samples
        self.key, subkey = jax.random.split(self.key)
        # TODO: run MPPI, keep the new mean control sequence in self.U
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
        pass

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
            self.config.H
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