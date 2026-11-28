"""Discrete MPPI/MPC agreement, including saturation and runtime calibration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import jax.numpy as jnp
import numpy as np

from test_control_step import controller as c

CASADI_PYTHON = os.environ.get('CASADI_PYTHON')
if CASADI_PYTHON is None and importlib.util.find_spec('casadi') is not None:
    CASADI_PYTHON = sys.executable


class KinematicLimitsTest(unittest.TestCase):
    def test_accelerating_at_max_speed_does_not_overstate_distance(self):
        for integrator in ('euler', 'rk4'):
            cfg = c.MPPIConfig(INTEGRATOR=integrator)
            actual = c.step(jnp.array([0., 0., 0., 2.]), jnp.array([3., 0.]), cfg)
            np.testing.assert_allclose(actual, [.1, 0., 0., 2.], atol=1e-6)

    def test_braking_at_rest_does_not_move_backwards(self):
        actual = c.step(jnp.zeros(4), jnp.array([-3., .2]), c.MPPIConfig())
        np.testing.assert_array_equal(actual, 0.)

    def test_near_limits_bounds_acceleration_before_integration(self):
        cfg = c.MPPIConfig()
        for speed, accel in ((1.99, 3.), (.01, -3.)):
            initial = jnp.array([0., 0., 0., speed])
            actual = c.step(initial, jnp.array([accel, 0.]), cfg)
            self.assertGreaterEqual(float(actual[3]), -1e-6)
            self.assertLessEqual(float(actual[3]), cfg.MAX_SPEED + 1e-6)
            expected_distance = .5 * (speed + float(actual[3])) * cfg.DT
            self.assertAlmostEqual(float(actual[0]), expected_distance, places=6)

    def test_noisy_out_of_bounds_speed_recovers_with_bounded_acceleration(self):
        cfg = c.MPPIConfig()
        for speed, expected in ((2.5, 2.35), (-.5, -.35)):
            actual = c.step(jnp.array([0., 0., 0., speed]), jnp.zeros(2), cfg)
            self.assertAlmostEqual(float(actual[3]), expected, places=6)

    def test_delay_prediction_uses_steering_calibration(self):
        cfg = c.MPPIConfig(DELAY_COMPENSATION=.1, STEERING_CURVATURE_GAIN=1.1,
                           STEERING_OFFSET=.02)
        predicted = c.compensate_delay(np.array([0., 0., 0., 1.]), [0., 0.], cfg)
        expected_yaw = np.tan(.02) * 1.1 / cfg.WB * .1
        self.assertAlmostEqual(predicted[2], expected_yaw)

    @unittest.skipUnless(CASADI_PYTHON, 'CasADi unavailable; set CASADI_PYTHON')
    def test_jax_matches_parameterized_casadi_at_boundaries_and_turns(self):
        cases = []
        for integrator in ('euler', 'rk4'):
            for maximum, gain, offset in ((2., 1., 0.), (6., 1.09245, .02)):
                config = dict(INTEGRATOR=integrator, MAX_SPEED=maximum,
                              STEERING_CURVATURE_GAIN=gain, STEERING_OFFSET=offset,
                              MIN_ACCEL=-2., MAX_ACCEL=1.5, WB=.35)
                for speed, accel in ((0., -3.), (.01, -3.), (1., .4),
                                     (maximum - .01, 3.), (maximum, 3.),
                                     (maximum + .5, 0.), (-.5, 0.)):
                    cases.append(dict(config=config, state=[1., -2., .7, speed],
                                      control=[accel, -.2]))
        script = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import casadi as ca
from acados_tracker import _bicycle_step
from controller_config import MPPIConfig
out = []
for case in json.load(sys.stdin):
    config = case['config']
    cfg = MPPIConfig(INTEGRATOR=config['INTEGRATOR'])
    x, u, p = ca.SX.sym('x', 4), ca.SX.sym('u', 2), ca.SX.sym('p', 7)
    step = _bicycle_step(ca, x, u, cfg, p)
    fn = ca.Function('step', [x, u, p], [step])
    parameters = [0., config['MAX_SPEED'], config['STEERING_CURVATURE_GAIN'],
                  config['STEERING_OFFSET'], config['MIN_ACCEL'], config['MAX_ACCEL'],
                  config['WB']]
    out.append(list(map(float, fn(case['state'], case['control'], parameters).full().ravel())))
print(json.dumps(out))
'''
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        result = subprocess.run([CASADI_PYTHON, '-c', script, str(scripts)],
                                input=json.dumps(cases), text=True, capture_output=True, check=True)
        for case, expected in zip(cases, json.loads(result.stdout)):
            actual = c.step(jnp.array(case['state']), jnp.array(case['control']),
                            c.MPPIConfig(**case['config']))
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
