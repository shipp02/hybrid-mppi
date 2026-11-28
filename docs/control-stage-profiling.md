# Control-stage profiling

Set `profile_control_stages: true` under `mppi_node.ros__parameters` in
`config/car_config.yaml`, rebuild/source the ROS package, and restart
`car_launch.py`. The car YAML enables profiling; the generic ROS default is off.
No acados regeneration is needed for profiling.

Alongside the existing control-loop summary, once per second the node logs:

```text
Control stages ms mean/p95/max (nested stages inclusive): stage[n=count]=mean/p95/max; ...
```

Each stage has its own sample count. Failed solves and optional diagnostics can
produce different counts. Buffers retain at most 256 samples per stage per log
window. Statistics are host wall-clock milliseconds unless labeled native.

## Hybrid stages

| Stage | Includes |
| --- | --- |
| `host.delay_prediction` | Predict planning state under the previous command |
| `gpu.plan_complete` | Dispatch and completion of fused grid update, reference, three-mode MPPI, nominal rollout, and optional corridor |
| `host.grid_debug` | Optional occupancy-grid readback/shared-memory export |
| `host.plan_readback` | Nominal, controls, targets, and corridor device-to-host readback |
| `host.corridor_debug` | Rate-limited shared-memory corridor export, not sidecar rendering |
| `mpc.total` | Entire tracker call, including Python setup and extraction |
| `mpc.prepare_reference` | Validation, yaw alignment, and augmented warm-start construction |
| `mpc.load_reference_and_limits` | Per-stage iterate, references, and bounds loading |
| `mpc.load_corridor_parameters_weights` | Per-stage corridor/model parameters and cost weights |
| `mpc.solve_total` | All RTI calls plus native-stat retrieval |
| `mpc.solve_1`, `mpc.solve_2`, ... | Wall-clock duration of each solver call |
| `mpc.read_native_stats` | Diagnostic statistics retrieval after each solver call |
| `mpc.native_time_tot_sum` | Sum of native acados total times across this tick's RTI calls |
| `mpc.native_time_qp_sum` | Sum of native QP times across this tick's RTI calls |
| `mpc.native_time_lin_sum` | Sum of native linearization times across this tick's RTI calls |
| `mpc.read_controls` | Control extraction, finite/bounds/slew checks |
| `host.feedback_upload` | Last-action update and completed smoothed-plan upload |
| `host.drive_publish` | Drive message construction/publication |
| `host.path_debug` | Selected-path/top-path diagnostic export, not sidecar rendering |
| `host.metrics` | Speed/latency/selection metric publication |
| `host.stop_publish` | Stop publication after a tracker rejection |

Without a tracker, `gpu.direct_complete` measures the completed fused controller
and `host.direct_readback_and_publish` measures result handling/publication.

## Interpretation

- Do not add every reported number: parent stages include their children, and
  native timings overlap wall-clock solver timings. `mpc.total` is the useful
  end-to-end MPC figure; `mpc.native_time_tot_sum` isolates native solver work.
- Native timings are summed after every RTI call, not taken only from the last
  call. Unsupported native-stat fields are omitted without rejecting controls.
- GPU stages explicitly wait for completion. They include dispatch/wait overhead,
  not just device execution. Readback is timed separately after that wait.
- The fused GPU graph remains intact. This profiler does not independently time
  occupancy update versus MPPI versus corridor inside that graph.
- Input validation, sensor preparation, and warmup before the existing control
  timer's start remain outside these stages. Summary logging is also excluded.
- Profiling adds synchronization points and statistics retrieval, so it perturbs
  latency. Set `profile_control_stages: false` and restart to remove those waits
  and native-stat queries after diagnosing the bottleneck.
