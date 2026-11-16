"""Host-side bicycle tracker. Optional dependencies load only when enabled."""
from pathlib import Path

import numpy as np


def _bicycle_step(ca, x, u, cfg):
    """Match MPPI's smooth bicycle integration (speed limits are constraints)."""
    def flow(state):
        return ca.vertcat(state[3] * ca.cos(state[2]),
                          state[3] * ca.sin(state[2]),
                          state[3] * ca.tan(u[1]) / cfg.WB, u[0])

    k1 = flow(x)
    if cfg.INTEGRATOR == 'euler':
        return x + cfg.DT * k1
    k2 = flow(x + cfg.DT * k1 / 2)
    k3 = flow(x + cfg.DT * k2 / 2)
    k4 = flow(x + cfg.DT * k3)
    return x + cfg.DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def build_ocp(cfg, max_steer_rate, export_directory):
    """Generate discrete nonlinear OCP; acados differentiates it online for RTI."""
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    model = AcadosModel()
    model.name = 'mppi_bicycle_tracker'
    # Last input is state, so first-stage slew includes last published command.
    model.x = ca.SX.sym('x', 6)
    model.u = ca.SX.sym('u', 2)
    model.disc_dyn_expr = ca.vertcat(
        _bicycle_step(ca, model.x[:4], model.u, cfg), model.u)
    model.cost_y_expr = ca.vertcat(
        model.x[:4], model.u, model.u - model.x[4:])
    model.cost_y_expr_e = model.x[:4]
    model.con_h_expr = model.u[1] - model.x[5]

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
    ocp.cost.W = np.diag([13.5, 13.5, 13.0, 5.5, 0.01, 1.0, 0.01, 100.0])
    ocp.cost.W_e = np.diag([13.5, 13.5, 13.0, 5.5])
    ocp.cost.yref = np.zeros(8)
    ocp.cost.yref_e = np.zeros(4)
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
    ocp.constraints.lh = np.array([-max_steer_rate * cfg.DT])
    ocp.constraints.uh = np.array([max_steer_rate * cfg.DT])
    ocp.code_export_directory = str(Path(export_directory).resolve())
    return ocp


def prepare_reference(x0, states, controls, previous_control, cfg):
    """Validate MPPI payload and align continuous yaw with measured quaternion yaw."""
    x0 = np.asarray(x0, dtype=float)
    states = np.array(states, dtype=float, copy=True)
    controls = np.asarray(controls, dtype=float)
    previous_control = np.asarray(previous_control, dtype=float)
    if (x0.shape != (4,) or states.shape != (cfg.H + 1, 4)
            or controls.shape != (cfg.H, 2) or previous_control.shape != (2,)):
        raise ValueError('Invalid tracker state/control horizon dimensions')
    if not all(np.isfinite(value).all()
               for value in (x0, states, controls, previous_control)):
        raise ValueError('Non-finite tracker input')
    states[:, 2] = np.unwrap(states[:, 2])
    states[:, 2] += 2 * np.pi * np.round((x0[2] - states[0, 2]) / (2 * np.pi))
    augmented = np.column_stack((states, np.vstack((previous_control, controls))))
    initial = np.concatenate((x0, previous_control))
    augmented[0] = initial
    return initial, augmented, controls


class AcadosTracker:
    """Single RTI step, relinearized at fresh MPPI nominal on each control tick."""

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
        """Relocate saved formulation and fail closed on incompatible runtime limits."""
        import os
        from acados_template import AcadosOcp

        ocp = AcadosOcp.from_json(str(directory / 'acados_ocp.json'))
        expected = build_ocp(self.cfg, self.max_steer_rate, directory)
        if (ocp.solver_options.N_horizon != self.cfg.H
                or not np.isclose(ocp.solver_options.tf, self.cfg.H * self.cfg.DT)
                or ocp.model.x.numel() != 6 or ocp.model.u.numel() != 2):
            raise ValueError('Prebuilt solver horizon, timestep or dimensions do not match')
        for field in ('lbu', 'ubu', 'lh', 'uh', 'lbx', 'ubx', 'lbx_e', 'ubx_e'):
            if not np.allclose(getattr(ocp.constraints, field),
                               getattr(expected.constraints, field)):
                raise ValueError(f'Prebuilt solver limits do not match: {field}')
        # Compare smooth discrete dynamics at multiple operating points, including turns.
        import casadi as ca
        loaded_step = ca.Function('loaded_step', [ocp.model.x, ocp.model.u],
                                  [ocp.model.disc_dyn_expr])
        expected_step = ca.Function('expected_step', [expected.model.x, expected.model.u],
                                    [expected.model.disc_dyn_expr])
        for yaw, speed, steer in ((0., 0., 0.), (0.3, 2., 0.1), (-0.7, 4., -0.2)):
            state = [0., 0., yaw, speed, 0., 0.]
            control = [0.4, steer]
            if not np.allclose(loaded_step(state, control), expected_step(state, control),
                               rtol=1e-10, atol=1e-12):
                raise ValueError('Prebuilt bicycle dynamics do not match runtime configuration')
        acados_root = Path(os.environ['ACADOS_SOURCE_DIR']).resolve()
        ocp.code_gen_options.code_export_directory = str(directory)
        ocp.code_gen_options.acados_include_path = str(acados_root / 'include')
        ocp.code_gen_options.acados_lib_path = str(acados_root / 'lib')
        return ocp

    def _load_reference(self, initial, nominal, controls):
        """Set RTI iterate and tracking targets, including measured initial state."""
        for stage in range(self.cfg.H):
            self.solver.set(stage, 'x', nominal[stage])
            self.solver.set(stage, 'u', controls[stage])
            self.solver.set(stage, 'yref', np.concatenate(
                (nominal[stage, :4], controls[stage], np.zeros(2))))
        self.solver.set(self.cfg.H, 'x', nominal[-1])
        self.solver.set(self.cfg.H, 'yref', nominal[-1, :4])
        self.solver.set(0, 'lbx', initial)
        self.solver.set(0, 'ubx', initial)
        # Permit bounded deceleration back into limits after noisy/out-of-range odometry.
        for stage in range(1, self.cfg.H + 1):
            lower = min(self.cfg.MIN_SPEED,
                        initial[3] + stage * self.cfg.DT * self.cfg.MAX_ACCEL)
            upper = max(self.cfg.MAX_SPEED,
                        initial[3] + stage * self.cfg.DT * self.cfg.MIN_ACCEL)
            self.solver.set(stage, 'lbx', np.array([lower]))
            self.solver.set(stage, 'ubx', np.array([upper]))

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

    def solve(self, x0, states, controls, previous_control):
        initial, nominal, controls = prepare_reference(
            x0, states, controls, previous_control, self.cfg)
        self._load_reference(initial, nominal, controls)
        status = self.solver.solve()
        if status != 0:
            raise RuntimeError(f'acados solve failed: status {status}')
        return self._read_controls(previous_control)
