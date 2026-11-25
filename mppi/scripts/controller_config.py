"""Shared controller defaults for ROS execution and offline solver generation."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MPPIConfig:
    NX: int = 4  # derived from DYNAMICS: 4 kinematic, 8 dynamic states
    DYNAMICS: str = "kinematic"  # opt-in "dynamic"; direct MPPI only
    NU: int = 2  # length of input vector: u = [a, delta]
    T: int = 1024  # number of sampled control sequences
    H: int = 40  # horizon length
    DT: float = 0.050  # time step [s]
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
    Qf: tuple = (2.0, 40.0, 4.0, 1.0)
    R: tuple = (0.01, 1.0)  # input cost [a, delta]
    Rd: tuple = (0.01, 1.0)  # input difference cost [a, delta]
    # ---------------------------------------------------

    WB: float = 0.33  # Wheelbase [m]
    STEERING_CURVATURE_GAIN: float = 1.0  # Applied to tan(steering) / WB in direct kinematic MPPI
    STEERING_DELAY_S: float = 0.0  # Steering dead time for direct kinematic MPPI
    MIN_STEER: float = -0.35  # minimum steering angle [rad]
    MAX_STEER: float = 0.35  # maximum steering angle [rad]
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_SPEED: float = 0.0  # minimum speed [m/s]
    MAX_SPEED: float = 2.0  # maximum speed [m/s]
    MAX_SPEED_LEAD: float = 1.0  # kinematic: setpoint may exceed measured speed by this [m/s]

    # Obstacles (kinematic)
    BODY_OFFSETS: tuple = (0.0,)  # points along the body checked for collision [m ahead of the pose]
    STOP_MARGIN: float = 0.3  # kinematic: a colliding plan brakes to stop this far before contact [m]
    NEAR_OBSTACLE_SPEED: float = 0.0  # speed cap at the inflation edge, 0 = off [m/s]
    NEAR_OBSTACLE_WEIGHT: float = 0.0  # cost of exceeding that cap

    # Dynamic bicycle starting estimates, NOT identified real-car parameters.
    MASS: float = 3.5  # kg
    YAW_INERTIA: float = 0.05  # kg m^2
    FRONT_LENGTH: float = 0.16  # COM to front axle [m]; rear length = WB - this
    TIRE_MU: float = 0.8
    FRONT_STIFFNESS: float = 20.0  # axle cornering stiffness [N/rad]
    REAR_STIFFNESS: float = 20.0
    FRONT_DRIVE_FRACTION: float = 0.5  # longitudinal force split, not torque control
    SPEED_RESPONSE_TIME: float = 0.25  # closed-loop speed-setpoint response [s]
    MAX_STEER_RATE: float = 3.14  # command slew [rad/s]
    DYNAMIC_SUBSTEPS: int = 10  # small steps for tire/lateral dynamics
    BLEND_LOW_SPEED: float = 0.5  # kinematic below this forward speed [m/s]
    BLEND_HIGH_SPEED: float = 1.5  # dynamic above this speed [m/s]
    LATERAL_RESPONSE_TIME: float = 0.1  # low-speed lateral/yaw relaxation [s]
    MAX_SIDESLIP: float = 0.6  # bounded slip allowed, not a slip reward [rad]
    MAX_YAW_RATE: float = 5.0  # rad/s
    ENVELOPE_WEIGHT: float = 100.0

    @property
    def steering_delay_steps(self):
        return math.floor(self.STEERING_DELAY_S / self.DT + 0.5)

    def __post_init__(self):
        """Reject unusable settings before tracing any rollout."""
        if self.DYNAMICS not in ('kinematic', 'dynamic'):
            raise ValueError('dynamics must be kinematic or dynamic')
        if self.H < 1 or self.T < 1:
            raise ValueError('horizon and num_samples must be positive')
        if not math.isfinite(self.MAX_SPEED_LEAD) or self.MAX_SPEED_LEAD <= 0:
            raise ValueError('max_speed_lead must be finite and positive')
        if not self.BODY_OFFSETS or not all(math.isfinite(d) for d in self.BODY_OFFSETS):
            raise ValueError('body_offsets must be a non-empty list of finite numbers')
        if not math.isfinite(self.STOP_MARGIN) or self.STOP_MARGIN < 0:
            raise ValueError('stop_margin must be finite and nonnegative')
        if self.NEAR_OBSTACLE_SPEED < 0 or self.NEAR_OBSTACLE_WEIGHT < 0:
            raise ValueError('near_obstacle_speed and near_obstacle_weight must be nonnegative')
        if not math.isfinite(self.STEERING_CURVATURE_GAIN) or self.STEERING_CURVATURE_GAIN <= 0:
            raise ValueError('steering_curvature_gain must be finite and positive')
        if self.STEERING_CURVATURE_GAIN != 1.0 and self.DYNAMICS != 'kinematic':
            raise ValueError('steering_curvature_gain requires kinematic MPPI')
        if not math.isfinite(self.STEERING_DELAY_S) or self.STEERING_DELAY_S < 0:
            raise ValueError('steering_delay_s must be finite and nonnegative')
        if self.STEERING_DELAY_S:
            if self.DYNAMICS != 'kinematic':
                raise ValueError('steering_delay_s requires kinematic MPPI')
            if self.STEERING_DELAY_S >= self.H * self.DT or self.steering_delay_steps >= self.H:
                raise ValueError('steering delay must be shorter than the rollout horizon')
        object.__setattr__(self, 'NX', 8 if self.DYNAMICS == 'dynamic' else 4)
        if self.DYNAMICS == 'kinematic':
            return
        positive = ('MASS', 'YAW_INERTIA', 'WB', 'TIRE_MU', 'FRONT_STIFFNESS',
                    'REAR_STIFFNESS', 'SPEED_RESPONSE_TIME', 'MAX_STEER_RATE',
                    'BLEND_LOW_SPEED', 'LATERAL_RESPONSE_TIME', 'MAX_SIDESLIP',
                    'MAX_YAW_RATE', 'ENVELOPE_WEIGHT', 'DT', 'MAX_SPEED')
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0
               for name in positive):
            raise ValueError('dynamic model scales must be finite and positive')
        if not 0 < self.FRONT_LENGTH < self.WB:
            raise ValueError('front_length must lie between zero and wheelbase')
        if not 0 <= self.FRONT_DRIVE_FRACTION <= 1:
            raise ValueError('front_drive_fraction must lie in [0, 1]')
        if not math.isfinite(self.BLEND_HIGH_SPEED) or self.BLEND_HIGH_SPEED <= self.BLEND_LOW_SPEED:
            raise ValueError('blend_high_speed must exceed blend_low_speed')
        if not isinstance(self.DYNAMIC_SUBSTEPS, int) or self.DYNAMIC_SUBSTEPS < 1:
            raise ValueError('dynamic_substeps must be a positive integer')
        if (self.MIN_SPEED != 0 or not math.isfinite(self.MIN_ACCEL)
                or not math.isfinite(self.MAX_ACCEL) or not self.MIN_ACCEL < 0 < self.MAX_ACCEL):
            raise ValueError('dynamic model requires forward-only speed and signed acceleration')
        if not -math.pi / 2 < self.MIN_STEER < 0 < self.MAX_STEER < math.pi / 2:
            raise ValueError('dynamic steering bounds must straddle zero inside (-pi/2, pi/2)')
        if self.H < 1 or self.T < 1:
            raise ValueError('dynamic horizon and sample count must be positive')
