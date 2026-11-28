"""Host-side bicycle tracker. Optional dependencies load only when enabled."""
from pathlib import Path

import numpy as np
from stage_profiler import StageProfiler


def _bicycle_step(ca, x, u, cfg, parameters=None):
    """Match MPPI's calibrated, bounded-acceleration kinematic integration.

    Acceleration is limited once at the step's initial speed, before Euler/RK4.
    The map is piecewise smooth at saturation; speed bounds remain OCP constraints.
    """
    if parameters is None:
        speed_min, speed_max = cfg.MIN_SPEED, cfg.MAX_SPEED
        curvature_gain, steering_offset = cfg.STEERING_CURVATURE_GAIN, cfg.STEERING_OFFSET
        accel_min, accel_max, wheelbase = cfg.MIN_ACCEL, cfg.MAX_ACCEL, cfg.WB
    else:
        speed_min, speed_max = parameters[0], parameters[1]
        curvature_gain, steering_offset = parameters[2], parameters[3]
        accel_min, accel_max, wheelbase = parameters[4], parameters[5], parameters[6]
    lower = ca.fmin(accel_max, (speed_min - x[3]) / cfg.DT)
    upper = ca.fmax(accel_min, (speed_max - x[3]) / cfg.DT)
    accel = ca.fmin(ca.fmax(u[0], accel_min), accel_max)
    accel = ca.fmin(ca.fmax(accel, lower), upper)

    def flow(state):
        return ca.vertcat(state[3] * ca.cos(state[2]),
                          state[3] * ca.sin(state[2]),
                          state[3] * ca.tan(u[1] + steering_offset)
                          * curvature_gain / wheelbase, accel)

    k1 = flow(x)
    if cfg.INTEGRATOR == 'euler':
        return x + cfg.DT * k1
    k2 = flow(x + cfg.DT * k1 / 2)
    k3 = flow(x + cfg.DT * k2 / 2)
    k4 = flow(x + cfg.DT * k3)
    return x + cfg.DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# NONLINEAR_LS weights. Stage y = [x, y, yaw, v | a, delta | da, ddelta | x-cx, y-cy];
# the corridor-center terms are weighted only in corridor mode (eq. 29's Q).
STAGE_WEIGHTS = (13.5, 13.5, 13.0, 5.5, 0.01, 1.0, 0.01, 10.0, 0.0, 0.0)
TERMINAL_WEIGHTS = (13.5, 13.5, 13.0, 5.5, 0.0, 0.0)
# Inactive corridor: a 10 m ball around the warm start keeps (x-cx)^2+(y-cy)^2-r^2
# near -100, inside a moderate lower bound (huge bounds wreck HPIPM's scaling)
INACTIVE_RADIUS = 10.0
CORRIDOR_LOWER = -150.0  # lower bound of (x-cx)^2 + (y-cy)^2 - r^2
CORRIDOR_SLACK = (1e2, 1e4)  # L1 and L2 penalty on corridor violation


def dynamics_parameters(cfg):
    """Runtime model values in the generated solver's parameter order."""
    return np.array([cfg.MIN_SPEED, cfg.MAX_SPEED,
                     cfg.STEERING_CURVATURE_GAIN, cfg.STEERING_OFFSET,
                     cfg.MIN_ACCEL, cfg.MAX_ACCEL, cfg.WB])


def build_ocp(cfg, max_steer_rate, export_directory):
    """Generate discrete nonlinear OCP; acados differentiates it online for RTI."""
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    model = AcadosModel()
    model.name = 'mppi_bicycle_tracker'
    # Last input is state, so first-stage slew includes last published command.
    model.x = ca.SX.sym('x', 6)
    model.u = ca.SX.sym('u', 2)
    # Runtime parameters: [cx, cy, r] followed by dynamics_parameters(cfg).
    model.p = ca.SX.sym('p', 10)
    model.disc_dyn_expr = ca.vertcat(
        _bicycle_step(ca, model.x[:4], model.u, cfg, model.p[3:]), model.u)
    offset = model.x[:2] - model.p[:2]
    ball = ca.sumsqr(offset) - model.p[2] ** 2
    model.cost_y_expr = ca.vertcat(
        model.x[:4], model.u, model.u - model.x[4:], offset)
    model.cost_y_expr_e = ca.vertcat(model.x[:4], offset)
    model.con_h_expr = ca.vertcat(model.u[1] - model.x[5], ball)
    model.con_h_expr_e = ball
    # acados skips con_h_expr at stage 0, so slew from the last command needs its own
    model.con_h_expr_0 = model.u[1] - model.x[5]

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = cfg.H
    ocp.solver_options.tf = cfg.H * cfg.DT
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = min(cfg.H, 10)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # State order everywhere: [x, y, yaw, v]. Position weights are world-frame.
    ocp.cost.W = np.diag(STAGE_WEIGHTS)
    ocp.cost.W_e = np.diag(TERMINAL_WEIGHTS)
    ocp.cost.yref = np.zeros(len(STAGE_WEIGHTS))
    ocp.cost.yref_e = np.zeros(len(TERMINAL_WEIGHTS))
    ocp.parameter_values = np.r_[0.0, 0.0, INACTIVE_RADIUS, dynamics_parameters(cfg)]
    ocp.constraints.x0 = np.zeros(6)
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([cfg.MIN_ACCEL, cfg.MIN_STEER])
    ocp.constraints.ubu = np.array([cfg.MAX_ACCEL, cfg.MAX_STEER])
    ocp.constraints.idxbx = np.array([3])
    ocp.constraints.lbx = np.array([cfg.MIN_SPEED])
    ocp.constraints.ubx = np.array([cfg.MAX_SPEED])
    ocp.constraints.idxbx_e = np.array([3])
    ocp.constraints.lbx_e = np.array([cfg.MIN_SPEED])
    ocp.constraints.ubx_e = np.array([cfg.MAX_SPEED])
    slew = max_steer_rate * cfg.DT
    ocp.constraints.lh = np.array([-slew, CORRIDOR_LOWER])
    ocp.constraints.uh = np.array([slew, 0.0])
    ocp.constraints.lh_0 = np.array([-slew])
    ocp.constraints.uh_0 = np.array([slew])
    ocp.constraints.lh_e = np.array([CORRIDOR_LOWER])
    ocp.constraints.uh_e = np.array([0.0])
    # Soft corridor: a hard ball can make the real-time QP infeasible
    l1, l2 = CORRIDOR_SLACK
    ocp.constraints.idxsh = np.array([1])
    ocp.constraints.idxsh_e = np.array([0])
    for suffix in ('', '_e'):
        for name, value in (('zl', l1), ('zu', l1), ('Zl', l2), ('Zu', l2)):
            setattr(ocp.cost, name + suffix, np.array([value]))
    ocp.code_export_directory = str(Path(export_directory).resolve())
    return ocp


def _align_yaw(states, yaw0):
    """Continuous yaw, shifted by whole turns to sit next to the measured yaw."""
    states = states.copy()
    states[:, 2] = np.unwrap(states[:, 2])
    states[:, 2] += 2 * np.pi * np.round((yaw0 - states[0, 2]) / (2 * np.pi))
    return states


def prepare_reference(x0, states, controls, previous_control, cfg, reference=None):
    """Validate MPPI payload; return iterate and tracking targets with aligned yaw.

    The MPPI nominal seeds the RTI iterate. Targets are the sampled spline
    reference when given, otherwise the nominal itself.
    """
    x0 = np.asarray(x0, dtype=float)
    states = np.array(states, dtype=float, copy=True)
    controls = np.asarray(controls, dtype=float)
    previous_control = np.asarray(previous_control, dtype=float)
    targets = states if reference is None else np.array(reference, dtype=float)
    if (x0.shape != (4,) or states.shape != (cfg.H + 1, 4)
            or targets.shape != (cfg.H + 1, 4)
            or controls.shape != (cfg.H, 2) or previous_control.shape != (2,)):
        raise ValueError('Invalid tracker state/control horizon dimensions')
    if not all(np.isfinite(value).all()
               for value in (x0, states, targets, controls, previous_control)):
        raise ValueError('Non-finite tracker input')
    states = _align_yaw(states, x0[2])
    targets = _align_yaw(targets, x0[2])
    augmented = np.column_stack((states, np.vstack((previous_control, controls))))
    initial = np.concatenate((x0, previous_control))
    augmented[0] = initial
    return initial, augmented, controls, targets


class AcadosTracker:
    """RTI iterations, relinearized at fresh MPPI nominal on each control tick."""

    profiler = StageProfiler()  # disabled unless the host attaches its profiler

    def __init__(self, cfg, *, max_steer_rate=np.pi, export_directory=None,
                 generate=True):
        try:
            from acados_template import AcadosOcpSolver
        except ImportError as exc:
            raise RuntimeError(
                'acados tracker requires built acados, acados_template and casadi; '
                'see docs/acados-tracker.md') from exc
        if cfg.H < 1 or cfg.DT <= 0 or max_steer_rate <= 0:
            raise ValueError('Tracker horizon, timestep and steering rate must be positive')
        self.cfg = cfg
        self.max_steer_rate = max_steer_rate
        default_directory = Path(__file__).resolve().parents[3] / 'acados_generated' / 'tracker'
        directory = Path(export_directory or default_directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        if generate:
            ocp = build_ocp(cfg, max_steer_rate, directory)
            self.solver = AcadosOcpSolver(ocp, json_file=str(directory / 'acados_ocp.json'))
        else:
            ocp = self._load_formulation(directory)
            self.solver = AcadosOcpSolver(
                ocp, generate=False, build=False, check_reuse_possible=False)

    def _load_formulation(self, directory):
        """Check compiled structure/dynamics; numerical limits load at runtime."""
        import os
        from acados_template import AcadosOcp

        ocp = AcadosOcp.from_json(str(directory / 'acados_ocp.json'))
        expected = build_ocp(self.cfg, self.max_steer_rate, directory)
        if (ocp.solver_options.N_horizon != self.cfg.H
                or not np.isclose(ocp.solver_options.tf, self.cfg.H * self.cfg.DT)
                or ocp.model.x.numel() != 6 or ocp.model.u.numel() != 2
                or ocp.model.p.numel() != 10):
            raise ValueError('Prebuilt solver horizon, timestep or dimensions do not match')
        # Compare discrete dynamics at turns and speed saturation boundaries.
        import casadi as ca
        loaded_step = ca.Function('loaded_step', [ocp.model.x, ocp.model.u, ocp.model.p],
                                  [ocp.model.disc_dyn_expr])
        expected_step = ca.Function(
            'expected_step', [expected.model.x, expected.model.u, expected.model.p],
            [expected.model.disc_dyn_expr])
        probes = ((0., 0., 0., 0.4), (0.3, 2., 0.1, 0.4),
                  (-0.7, 4., -0.2, 0.4),
                  (0.2, self.cfg.MAX_SPEED, 0.1, self.cfg.MAX_ACCEL),
                  (-0.2, self.cfg.MIN_SPEED, -0.1, self.cfg.MIN_ACCEL))
        for yaw, speed, steer, accel in probes:
            state = [0., 0., yaw, speed, 0., 0.]
            control = [accel, steer]
            if not np.allclose(loaded_step(state, control, expected.parameter_values),
                               expected_step(state, control, expected.parameter_values),
                               rtol=1e-10, atol=1e-12):
                raise ValueError('Prebuilt bicycle dynamics do not match runtime configuration')
        acados_root = Path(os.environ['ACADOS_SOURCE_DIR']).resolve()
        ocp.code_gen_options.code_export_directory = str(directory)
        ocp.code_gen_options.acados_include_path = str(acados_root / 'include')
        ocp.code_gen_options.acados_lib_path = str(acados_root / 'lib')
        return ocp

    def _load_reference(self, initial, nominal, controls, targets):
        """Set RTI iterate and tracking targets, including measured initial state."""
        for stage in range(self.cfg.H):
            self.solver.set(stage, 'x', nominal[stage])
            self.solver.set(stage, 'u', controls[stage])
            self.solver.set(stage, 'yref', np.concatenate(
                (targets[stage], controls[stage], np.zeros(4))))
        self.solver.set(self.cfg.H, 'x', nominal[-1])
        self.solver.set(self.cfg.H, 'yref', np.concatenate((targets[-1], np.zeros(2))))
        self.solver.set(0, 'lbx', initial)
        self.solver.set(0, 'ubx', initial)
        self._load_limits(initial[3])

    def _load_limits(self, initial_speed):
        """Override reusable input, slew and speed bounds from runtime config."""
        lower_u = np.array([self.cfg.MIN_ACCEL, self.cfg.MIN_STEER])
        upper_u = np.array([self.cfg.MAX_ACCEL, self.cfg.MAX_STEER])
        slew = self.max_steer_rate * self.cfg.DT
        for stage in range(self.cfg.H):
            self.solver.constraints_set(stage, 'lbu', lower_u)
            self.solver.constraints_set(stage, 'ubu', upper_u)
            lower_h = [-slew] if stage == 0 else [-slew, CORRIDOR_LOWER]
            upper_h = [slew] if stage == 0 else [slew, 0.0]
            self.solver.constraints_set(stage, 'lh', np.array(lower_h))
            self.solver.constraints_set(stage, 'uh', np.array(upper_h))
        self.solver.constraints_set(self.cfg.H, 'lh', np.array([CORRIDOR_LOWER]))
        self.solver.constraints_set(self.cfg.H, 'uh', np.array([0.0]))
        # Permit bounded recovery after noisy/out-of-range odometry.
        for stage in range(1, self.cfg.H + 1):
            lower = min(self.cfg.MIN_SPEED,
                        initial_speed + stage * self.cfg.DT * self.cfg.MAX_ACCEL)
            upper = max(self.cfg.MAX_SPEED,
                        initial_speed + stage * self.cfg.DT * self.cfg.MIN_ACCEL)
            self.solver.set(stage, 'lbx', np.array([lower]))
            self.solver.set(stage, 'ubx', np.array([upper]))

    def _load_corridor(self, corridor, nominal):
        """Ball per stage and its center weight; None leaves the ball inactive."""
        if corridor is None:
            balls = np.column_stack((nominal[:, :2], np.full(self.cfg.H + 1, INACTIVE_RADIUS)))
            weight = 0.0
        else:
            centers, radii = (np.asarray(value, dtype=float) for value in corridor)
            if centers.shape != (self.cfg.H + 1, 2) or radii.shape != (self.cfg.H + 1,):
                raise ValueError('Invalid corridor horizon dimensions')
            if not (np.isfinite(centers).all() and np.isfinite(radii).all()):
                raise ValueError('Non-finite corridor')
            balls = np.column_stack((centers, radii))
            weight = self.cfg.CORRIDOR_Q
        stage = np.array(STAGE_WEIGHTS[:-2])
        terminal = np.array(TERMINAL_WEIGHTS[:-2])
        if corridor is not None:
            # MPPI owns path selection; MPC smooths controls within its corridor.
            stage[:3] = 0.0
            terminal[:3] = 0.0
            stage[-1] = self.cfg.CORRIDOR_STEER_RATE_WEIGHT  # steering change
        stage_w = np.diag(np.r_[stage, weight, weight])
        terminal_w = np.diag(np.r_[terminal, weight, weight])
        dynamics = np.tile(dynamics_parameters(self.cfg), (self.cfg.H + 1, 1))
        parameters = np.column_stack((balls, dynamics))
        for stage in range(self.cfg.H + 1):
            self.solver.set(stage, 'p', parameters[stage])
            self.solver.cost_set(stage, 'W', terminal_w if stage == self.cfg.H else stage_w)

    def _read_controls(self, previous_control):
        """Reject unusable RTI output rather than retaining last drive command."""
        controls = np.array([self.solver.get(k, 'u') for k in range(self.cfg.H)])
        lower = np.array([self.cfg.MIN_ACCEL, self.cfg.MIN_STEER])
        upper = np.array([self.cfg.MAX_ACCEL, self.cfg.MAX_STEER])
        if (not np.isfinite(controls).all() or np.any(controls < lower - 1e-5)
                or np.any(controls > upper + 1e-5)):
            raise RuntimeError('acados returned invalid or out-of-bounds controls')
        steering = np.r_[previous_control[1], controls[:, 1]]
        if np.any(np.abs(np.diff(steering)) > self.max_steer_rate * self.cfg.DT + 1e-5):
            raise RuntimeError('acados steering slew limit violated')
        return np.clip(controls, lower, upper)

    def solve(self, x0, states, controls, previous_control, reference=None, corridor=None):
        """RTI from the MPPI warm start. corridor = (centers (H+1, 2), radii (H+1,)) or None.

        With a corridor, CORRIDOR_SQP_ITERS iterations: the first linearizes the
        ball at its center (zero gradient), the next ones enforce it.
        """
        with self.profiler.measure('mpc.prepare_reference'):
            initial, nominal, controls, targets = prepare_reference(
                x0, states, controls, previous_control, self.cfg, reference)
        with self.profiler.measure('mpc.load_reference_and_limits'):
            self._load_reference(initial, nominal, controls, targets)
        with self.profiler.measure('mpc.load_corridor_parameters_weights'):
            self._load_corridor(corridor, nominal)
        with self.profiler.measure('mpc.solve_total'):
            self._solve_iterations(1 if corridor is None else self.cfg.CORRIDOR_SQP_ITERS)
        with self.profiler.measure('mpc.read_controls'):
            return self._read_controls(previous_control)

    def _solve_iterations(self, count):
        """Time each RTI call; accumulate native stats across all calls this tick."""
        native = dict.fromkeys(('time_tot', 'time_qp', 'time_lin'), 0.0)
        try:
            for iteration in range(count):
                with self.profiler.measure(f'mpc.solve_{iteration + 1}'):
                    status = self.solver.solve()
                if self.profiler.enabled:
                    with self.profiler.measure('mpc.read_native_stats'):
                        self._accumulate_native_timings(native)
                if status != 0:
                    raise RuntimeError(f'acados solve failed: status {status}')
        finally:
            if self.profiler.enabled:
                for name, seconds in native.items():
                    if seconds is not None:
                        self.profiler.record(f'mpc.native_{name}_sum', seconds)

    def _accumulate_native_timings(self, totals):
        """Unavailable diagnostic fields must never reject a drive command."""
        for name, total in totals.items():
            if total is None:
                continue
            try:
                value = float(self.solver.get_stats(name))
                totals[name] = total + value if np.isfinite(value) and value >= 0. else None
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                totals[name] = None
