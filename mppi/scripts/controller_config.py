"""Shared controller defaults for ROS execution and offline solver generation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MPPIConfig:
    NX: int = 4  # length of state vector: x = [p_x, p_y, yaw, v]
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 40  # horizon length
    DT: float = 0.05  # time step [s]
    INTEGRATOR: str = "rk4"  # "euler" or "rk4" (bonus)
    SAMPLING: str = "uniform"  # "clip", or bonus: "truncated" (normal) or "uniform"
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
