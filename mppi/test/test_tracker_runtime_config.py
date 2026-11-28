"""Prebuilt tracker reuse: only compiled structure/dynamics require regeneration."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from acados_tracker import AcadosTracker
from controller_config import MPPIConfig
from stage_profiler import StageProfiler

GENERATED = Path(__file__).resolve().parents[1] / 'generated/acados_tracker'
HAS_ACADOS = (importlib.util.find_spec('acados_template') is not None
              and bool(os.environ.get('ACADOS_SOURCE_DIR')))
NATIVE_DIR = os.environ.get('TRACKER_NATIVE_DIR')


@unittest.skipUnless(HAS_ACADOS, 'Requires acados_template and ACADOS_SOURCE_DIR')
class TrackerRuntimeConfigTest(unittest.TestCase):
    def config(self):
        return MPPIConfig(MIN_SPEED=.1, MAX_SPEED=8., MIN_ACCEL=-4., MAX_ACCEL=2.,
                          MIN_STEER=-.2, MAX_STEER=.25, STEERING_CURVATURE_GAIN=1.1,
                          STEERING_OFFSET=.01, WB=.4)

    def load(self, cfg):
        tracker = object.__new__(AcadosTracker)
        tracker.cfg, tracker.max_steer_rate = cfg, .8
        return tracker._load_formulation(GENERATED)

    def test_runtime_limits_and_calibration_need_no_regeneration(self):
        # Both wider and narrower limits are legal; every bound loads at solve time.
        ocp = self.load(self.config())
        self.assertEqual(ocp.model.p.numel(), 10)

    def test_changed_horizon_or_timestep_is_rejected(self):
        for override in (dict(H=41), dict(DT=.1)):
            with self.assertRaisesRegex(ValueError, 'horizon, timestep or dimensions'):
                self.load(replace(self.config(), **override))

    def test_changed_compiled_integrator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dynamics do not match'):
            self.load(replace(self.config(), INTEGRATOR='euler'))

    @unittest.skipUnless(NATIVE_DIR, 'Set TRACKER_NATIVE_DIR to compiled solver directory')
    def test_native_solver_reuses_code_with_different_limits_and_wheelbase(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory(prefix='tracker-runtime-test-') as directory:
            root = Path(directory)
            shutil.copy(GENERATED / 'acados_ocp.json', root / 'acados_ocp.json')
            libraries = list(Path(NATIVE_DIR).glob('libacados_ocp_solver_*'))
            self.assertTrue(libraries)
            for library in libraries:
                shutil.copy(library, root / library.name)
            tracker = AcadosTracker(cfg, max_steer_rate=.8, export_directory=root, generate=False)
            tracker.profiler = StageProfiler(True)
            states = np.zeros((cfg.H + 1, 4))
            states[:, 0], states[:, 3] = np.arange(cfg.H + 1) * cfg.DT, 1.
            controls = np.tile([0., -cfg.STEERING_OFFSET], (cfg.H, 1))
            corridor = states[:, :2], np.full(cfg.H + 1, .3)
            result = tracker.solve(states[0], states, controls, controls[0],
                                   reference=states, corridor=corridor)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result, controls, atol=1e-5)
            predicted = np.array([tracker.solver.get(k, 'x') for k in range(cfg.H + 1)])
            np.testing.assert_allclose(predicted[:, 3], 1., atol=1e-5)
            timings = tracker.profiler.drain()
            self.assertGreater(timings['mpc.native_time_tot_sum'][1], 0.)
            for iteration in range(1, cfg.CORRIDOR_SQP_ITERS + 1):
                self.assertIn(f'mpc.solve_{iteration}', timings)


if __name__ == '__main__':
    unittest.main()
