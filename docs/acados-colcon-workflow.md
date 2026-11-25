# Build the acados tracker with colcon

## Layout

- `tools/generate_acados_tracker.py`: offline C/JSON generator; no compilation.
- `mppi/generated/acados_tracker/`: committed generated sources and formulation.
- `mppi/scripts/controller_config.py`: shared controller/generator defaults.
- `MPPI_BUILD_ACADOS=ON`: compile and install the native solver with the ROS package.

The checked-in formulation uses 40 steps, a 0.05 s timestep, RK4, a 6 m/s speed
bound, and a steering-rate limit of pi rad/s. This permits 6 m/s operation and
is not a low-speed testing configuration.

## Generate on the development machine

Use acados v0.6.0 and CasADi 3.7.2. The selected Python must have
`acados_template` and CasADi installed. The acados template renderer must be
available; ROS and JAX are not needed for generation.

```bash
export ACADOS_SOURCE_DIR="$HOME/acados"  # adjust to your installation
python tools/generate_acados_tracker.py
```

To change the horizon or speed bound:

```bash
python tools/generate_acados_tracker.py --horizon 40 --max-speed 1.0
```

Commit the resulting C, headers, and JSON, not object files or native libraries.
Regenerate and rebuild whenever the formulation changes. Generated JSON paths
are portable; the runtime loader relocates them to the VM installation.

## Build on the VM

Build/install matching acados v0.6.0 for the VM architecture at `~/acados`.
Install `acados_template` and CasADi into the Python environment used by ROS;
native C compilation does not install those Python dependencies.

Source your ROS distribution, then:

```bash
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd ~/sim_ws
colcon build --packages-select hybrid_mppi --cmake-args \
  -DMPPI_BUILD_ACADOS=ON \
  -DACADOS_ROOT="$ACADOS_SOURCE_DIR"
source install/setup.bash
```

Colcon compiles generated C natively and installs the library and JSON under
`lib/mppi/acados_tracker/`. The node discovers this directory automatically.
There is no runtime generation and no need to copy JSON or use the generated
Makefile. acados remains an external dependency; colcon does not download it.
`MPPI_BUILD_ACADOS` defaults OFF for ordinary MPPI builds without acados.

## Run

For the checked-in 40-step, 6 m/s formulation:

```bash
ros2 launch hybrid_mppi car_launch.py \
  odom_topic:=/ego_racecar/odom \
  tracker:=acados use_cpu:=true horizon:=40 max_speed:=6.0
```

For a 1 m/s test, first generate with `--max-speed 1.0`, commit and rebuild, then
launch with `max_speed:=1.0`. Runtime horizon, limits, and dynamics must match the
compiled formulation; the loader rejects incompatible settings.

Select waypoints matching the simulator map (`track:=levine` by default).
The simulator must provide `/scan`, `/drive`, odometry, and the required LiDAR TF.
The tracker currently has no post-solve collision validation. Solver errors and
invalid controls still trigger a stop.

If the installed script's Python shebang cannot import `acados_template`, install
it into that Python environment or invoke the installed node explicitly with a
ROS-compatible venv Python. Activating a venv alone does not change the shebang.
An explicit `tracker_prebuilt_dir` remains available for external solver builds.
