# Car runs: recording, fetching, replay

## Record

`launch/car_launch.py` records every run. Recording starts with the controller and
stops on Ctrl-C or when `mppi_node` exits; the MCAP file is closed cleanly either way.

```bash
ros2 launch hybrid_mppi car_launch.py run_label:=levine_2ms   # record:=false to skip
```

Each run is a folder in `~/f1tenth_ws/run_logs/hybrid_mppi/<date>_<label>/`:

| File | Content |
|---|---|
| `bag/` | MCAP bag: TF, map, scan, odometry, PF pose, drive/teleop commands, MPPI visualization and `/mppi/metrics/*` |
| `car_config.yaml` | the parameter file used |
| `launch_args.yaml` | launch arguments given on the command line (`track`, `max_speed`, `use_cpu`) |
| `version.txt` | hybrid-mppi commit and local changes |

## Fetch to the laptop

```bash
src/hybrid-mppi/tools/fetch_car_run.sh          # all finished runs, then delete them on the car
src/hybrid-mppi/tools/fetch_car_run.sh --list   # what is on the car
src/hybrid-mppi/tools/fetch_car_run.sh --keep   # copy without deleting
```

Runs land in `<workspace>/run_logs/hybrid_mppi/`. A run is deleted on the car only
after an rsync checksum comparison shows no difference. Runs still being recorded are
skipped. The script stops before a copy would leave less than 1 GB free.

## Replay

Open `bag/*.mcap` in Foxglove (desktop app or app.foxglove.dev, "Open local file") and
import `tools/foxglove/mppi_car_layout.json`: map view on the left; speed (route
target, commanded, measured) and latency (compute, loop period, pose age, scan age)
plots on the right.

For ROS tools, `tools/replay_car_run.sh <run folder>` plays the bag on
`ROS_DOMAIN_ID` 42, localhost only, with `/drive`, `/teleop` and `/ackermann_cmd`
renamed under `/replay/`, and starts foxglove_bridge on `ws://localhost:8765`. It
refuses domain 10, which the car uses.

## Live metrics

`mppi_node` publishes `std_msgs/Float32` on `/mppi/metrics/<name>` every control tick:

| Name | Meaning |
|---|---|
| `route_speed` | waypoint speed nearest to the car [m/s] |
| `commanded_speed` | speed sent on `/drive` [m/s] |
| `measured_speed` | speed from localization [m/s] |
| `compute_ms` | control step time, from observation to published command |
| `period_ms` | time between control steps (target `dt`, 50 ms) |
| `odom_age_ms` | now minus the pose message stamp (localization latency) |
| `scan_age_ms` | now minus the latest scan stamp |
