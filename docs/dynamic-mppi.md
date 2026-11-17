# Direct slip-aware MPPI (experimental)

Dynamic mode generates and checks trajectories inside MPPI, then publishes the
first command directly. It does not use or change the acados tracker. The default
remains `dynamics:=kinematic`; combining `dynamic` with `tracker:=acados` is rejected.

This is a nominal-model feasibility check, not a real-car safety guarantee or a
validated drift controller. Start in simulation at reduced speed. Tire and actuator
defaults are starting estimates, not identified parameters.

## Build and run on the VM

Source your ROS environment, then build from your colcon workspace root:

```bash
colcon build --packages-select mppi --symlink-install --cmake-args -DMPPI_BUILD_ACADOS=OFF
source install/setup.bash
```

Start the simulator separately. Stop any other controller publishing `/drive`.
For an initial Levine simulation run:

```bash
ros2 run mppi mppi_node.py --ros-args \
  -p track:=levine \
  -p odom_topic:=/ego_racecar/odom \
  -p dynamics:=dynamic -p tracker:=none \
  -p use_cpu:=true -p horizon:=20 -p max_speed:=2.0 \
  -p visualize:=true
```

Use `use_cpu:=false` when the VM has a working JAX CUDA backend. GPU selection
never silently falls back to CPU. Compare reported loop latency with the 50 ms
control period; CPU execution is not established to meet this deadline. Match
`use_sim_time` to the simulator's clock configuration: odometry, scan stamps and
the controller must share a clock. Wait for compilation and `MPPI ready` before
judging steady-state behavior.

The existing configurable launch also accepts dynamic mode:

```bash
ros2 launch mppi car_launch.py \
  odom_topic:=/ego_racecar/odom track:=levine \
  dynamics:=dynamic tracker:=none use_cpu:=true horizon:=20 max_speed:=2.0
```

Despite its name, this launch can target simulator odometry. Graded Levine and
Spielberg launches remain unchanged and use kinematic mode.

## State, frames and command semantics

Dynamic rollout state is:

```text
[COM x, COM y, body yaw, body vx, body vy, yaw rate,
 previous steering command, previous speed setpoint]
```

Odometry must describe the vehicle COM, with signed longitudinal/lateral velocity
in the body frame (`+x` forward, `+y` left) and positive counterclockwise yaw rate.
A nonempty `child_frame_id` is required and is also used for LiDAR extrinsics.
A frame name alone cannot verify these semantics: check the simulator bridge or
localization source. World-frame twist or rear-axle pose needs conversion upstream;
this controller does not perform that conversion or estimate missing lateral speed.

Inputs remain two-dimensional: speed-setpoint slew and steering command. Unlike
kinematic mode, published speed advances the previous **speed setpoint**, not
measured speed. The rollout models the same command history and approximates the
closed-loop speed response with `speed_response_time`. Steering-command slew is
bounded from the last published command; unmeasured steering servo lag is not modeled.
Only this controller may own the command stream while this history is in use.

## Model and trajectory selection

- Six physical states use substepped RK4, regardless of the kinematic `integrator`
  setting. Low forward speed blends toward kinematic lateral/yaw relaxation.
- Axle lateral forces use saturating cornering-stiffness curves. Longitudinal
  demand consumes each axle's friction-circle capacity, with static axle loads.
- Tracking compares velocity **course** with path tangent. Body heading can differ
  from travel direction; slip inside the configured envelope is not penalized
  toward zero or rewarded for its own sake.
- Samples and the weighted command sequence are projected onto command bounds and
  steering slew limits. The warm-start nominal is retained as one sample.
- The weighted nominal is simulated and checked again. If infeasible, MPPI selects
  the lowest-total-cost feasible sample; if none exists, it checks a braking,
  steering-centering candidate. If that also fails, publication requests stop.
- Feasibility checks cover finite values, forward motion, speed/sideslip/yaw-rate
  envelopes and command bounds/slew only. Collision costs retain the existing
  `grid.is_free()` point queries and inflation behavior. No extra footprint,
  grid-boundary or hard collision-feasibility checks are added.
- The stop path sends zero speed and steering; it is not a validated recovery
  maneuver from an established drift. Sensor-age and plan-expiry checks are not
  enforced. Cached sensor data can continue to drive planning if updates stop.

A model-feasible trajectory is not guaranteed collision-free: collision avoidance
remains a cost, as in the original controller. There is no continuous swept-volume
check, terminal recoverability proof, uncertainty ensemble, load transfer model or
guaranteed recursive feasibility.

## Parameters to identify before increasing speed

Dynamic parameters are ROS parameters using lowercase names from
`mppi/scripts/controller_config.py`. Restart the node after changing them.

| Concern | Parameters |
| --- | --- |
| Geometry and inertia | `mass`, `yaw_inertia`, `wb`, `front_length` (rear length is `wb - front_length`) |
| Grip and tires | `tire_mu`, `front_stiffness`, `rear_stiffness`, `front_drive_fraction` |
| Actuation | `speed_response_time`, `max_steer_rate` |
| Low-speed numerics | `blend_low_speed`, `blend_high_speed`, `lateral_response_time`, `dynamic_substeps` |
| Operating envelope | `max_speed`, `max_sideslip`, `max_yaw_rate`, `envelope_weight` |

Cornering stiffness is axle force per slip angle (N/rad), not a simulator's
normalized tire coefficient. Do not copy coefficients without checking units.
Record actual speed response, signed lateral velocity, measured yaw rate, commands,
cross-track error and loop timing. Test entry, sustained turns and exit/recovery
across plausible grip and delay ranges, not only one simulator configuration.

## Local validation

From the repository root, with the project's Python environment:

```bash
PYTHONPATH=mppi/test ../../venv/bin/python -m unittest -v \
  test_dynamic_bicycle test_dynamic_mppi test_control_step test_generated_tracker
```

The command above passed 32 tests after removing added collision checks and their
related tests.
An earlier broader local selection passed 49 tests, including GPU-residency
contracts executed on JAX CPU. Full discovery also found existing acados mock,
missing debug-module and occupancy-grid test failures; these reproduced against
the pre-integration controller. No VM laps, CUDA timing or real-car tests have
been performed for this change.
