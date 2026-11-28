"""Committed generated-solver contract; runnable without ROS, JAX, or acados."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
GENERATED = PACKAGE / 'generated/acados_tracker'


def load_defaults():
    spec = importlib.util.spec_from_file_location(
        '_generated_tracker_config', PACKAGE / 'scripts/controller_config.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MPPIConfig()


class GeneratedTrackerTest(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((GENERATED / 'acados_ocp.json').read_text())

    def test_sources_match_formulation_name(self):
        name = self.payload['name']
        self.assertEqual(list(GENERATED.glob('acados_solver_*.c')),
                         [GENERATED / f'acados_solver_{name}.c'])
        self.assertTrue((GENERATED / f'acados_solver_{name}.h').is_file())
        for component in ('model', 'cost', 'constraints'):
            self.assertTrue(list(GENERATED.glob(f'*_{component}/*.c')))

    def test_formulation_has_no_generator_machine_paths(self):
        options = self.payload['code_gen_options']
        self.assertEqual(options['code_export_directory'], '.')
        self.assertEqual(options['json_file'], 'acados_ocp.json')
        self.assertEqual(options['acados_lib_path'], '${ACADOS_SOURCE_DIR}/lib')
        self.assertEqual(options['cython_include_dirs'], [])

    def test_dynamics_and_limits_contract(self):
        cfg = load_defaults()
        options = self.payload['solver_options']
        dims = self.payload['dims']
        self.assertEqual(dims['nx'], cfg.NX + cfg.NU)
        self.assertEqual(dims['nu'], cfg.NU)
        self.assertAlmostEqual(options['tf'] / options['N_horizon'], cfg.DT)
        # Numerical limits are generation-time defaults, not a reuse contract.
        constraints = self.payload['constraints']
        self.assertEqual(len(constraints['lbu']), cfg.NU)
        self.assertEqual(len(constraints['ubu']), cfg.NU)
        self.assertTrue(all(lo <= hi for lo, hi in
                            zip(constraints['lbu'], constraints['ubu'])))
        self.assertEqual(len(constraints['lbx']), 1)
        self.assertEqual(len(constraints['ubx']), 1)
        parameters = self.payload['parameter_values']
        self.assertEqual(len(parameters), 10)
        self.assertLess(parameters[3], parameters[4])  # speed limits
        self.assertLess(parameters[7], parameters[8])  # acceleration limits
        self.assertGreater(parameters[9], 0.)  # wheelbase

    def test_first_step_steering_slew_is_constrained(self):
        constraints = self.payload['constraints']
        self.assertEqual(self.payload['dims']['nh_0'], 1)
        self.assertEqual(constraints['lh_0'], constraints['lh'][:1])
        self.assertEqual(constraints['uh_0'], constraints['uh'][:1])

    def test_corridor_ball_is_a_soft_parametric_constraint(self):
        dims, constraints = self.payload['dims'], self.payload['constraints']
        # Corridor, speed limits, steering calibration, accel limits, wheelbase.
        self.assertEqual(dims['np'], 10)
        self.assertEqual((dims['nh'], dims['nh_e'], dims['nsh'], dims['nsh_e']), (2, 1, 1, 1))
        self.assertEqual(constraints['uh'][1], 0.0)
        # a moderate lower bound: huge bounds wreck the interior-point QP's scaling
        self.assertGreaterEqual(constraints['lh'][1], -1e3)


if __name__ == '__main__':
    unittest.main()
