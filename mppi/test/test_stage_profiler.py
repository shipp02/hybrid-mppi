"""Stage timing is bounded, opt-in, and never changes solver acceptance."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_control_step import controller as c
from acados_tracker import AcadosTracker
from stage_profiler import StageProfiler


class StageProfilerTest(unittest.TestCase):
    def test_disabled_profiler_does_not_read_clock_or_wait(self):
        profiler = StageProfiler()
        with patch('stage_profiler.time.perf_counter') as clock, \
                patch('jax.block_until_ready') as wait:
            with profiler.measure('stage'):
                profiler.wait(object())
            profiler.record('native', .001)
        clock.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(profiler.drain(), {})

    def test_measured_seconds_become_milliseconds_even_after_exception(self):
        profiler = StageProfiler(True)
        with patch('stage_profiler.time.perf_counter', side_effect=[1., 1.002]):
            with self.assertRaisesRegex(ValueError, 'failed'):
                with profiler.measure('stage'):
                    raise ValueError('failed')
        count, mean, p95, maximum = profiler.drain()['stage']
        self.assertEqual(count, 1)
        np.testing.assert_allclose([mean, p95, maximum], 2.)
        self.assertEqual(profiler.drain(), {})

    def test_buffers_are_bounded_and_invalid_stats_are_ignored(self):
        profiler = StageProfiler(True, capacity=2)
        for value in (.001, .002, .004, np.nan, -.1):
            profiler.record('stage', value)
        count, mean, p95, maximum = profiler.drain()['stage']
        self.assertEqual(count, 2)
        np.testing.assert_allclose([mean, p95, maximum], [3., 3.9, 4.])

    def test_enabled_wait_and_summary_log(self):
        profiler = StageProfiler(True)
        values = object()
        with patch('jax.block_until_ready') as wait:
            profiler.wait(values)
        wait.assert_called_once_with(values)
        profiler.record('gpu.plan_complete', .01)
        logger = Mock()
        profiler.log_summary(logger)
        self.assertIn('gpu.plan_complete[n=1]=10.00/10.00/10.00', logger.info.call_args.args[0])
        self.assertIn('inclusive', logger.info.call_args.args[0])
        profiler.log_summary(logger)
        logger.info.assert_called_once()


class TrackerStageProfilerTest(unittest.TestCase):
    def tracker(self, enabled=True):
        tracker = object.__new__(AcadosTracker)
        tracker.cfg = c.MPPIConfig(H=2, CORRIDOR_SQP_ITERS=3)
        tracker.profiler = StageProfiler(enabled)
        tracker.solver = Mock()
        tracker.solver.solve.return_value = 0
        tracker.solver.get_stats.side_effect = lambda name: {
            'time_tot': .002, 'time_qp': .001, 'time_lin': .0005}[name]
        tracker._load_reference = Mock()
        tracker._load_corridor = Mock()
        tracker._read_controls = Mock(return_value=np.zeros((2, 2)))
        return tracker

    def solve(self, tracker, corridor=True):
        return tracker.solve(np.zeros(4), np.zeros((3, 4)), np.zeros((2, 2)), np.zeros(2),
                             corridor=(np.zeros((3, 2)), np.ones(3)) if corridor else None)

    def test_native_stats_sum_all_iterations_not_only_last(self):
        tracker = self.tracker()
        self.solve(tracker)
        summary = tracker.profiler.drain()
        self.assertEqual(tracker.solver.solve.call_count, 3)
        self.assertEqual(tracker.solver.get_stats.call_count, 9)
        for iteration in (1, 2, 3):
            self.assertEqual(summary[f'mpc.solve_{iteration}'][0], 1)
        self.assertAlmostEqual(summary['mpc.native_time_tot_sum'][1], 6.)
        self.assertAlmostEqual(summary['mpc.native_time_qp_sum'][1], 3.)
        self.assertAlmostEqual(summary['mpc.native_time_lin_sum'][1], 1.5)
        for stage in ('prepare_reference', 'load_reference_and_limits',
                      'load_corridor_parameters_weights', 'solve_total', 'read_controls'):
            self.assertIn(f'mpc.{stage}', summary)

    def test_non_corridor_mode_times_one_iteration(self):
        tracker = self.tracker()
        self.solve(tracker, corridor=False)
        summary = tracker.profiler.drain()
        self.assertEqual(tracker.solver.solve.call_count, 1)
        self.assertAlmostEqual(summary['mpc.native_time_tot_sum'][1], 2.)

    def test_disabled_profiling_does_not_fetch_native_stats(self):
        tracker = self.tracker(enabled=False)
        self.solve(tracker)
        tracker.solver.get_stats.assert_not_called()
        self.assertEqual(tracker.profiler.drain(), {})

    def test_unavailable_native_stat_does_not_reject_controls(self):
        tracker = self.tracker()
        tracker.solver.get_stats.side_effect = RuntimeError('unsupported field')
        np.testing.assert_array_equal(self.solve(tracker), 0.)
        summary = tracker.profiler.drain()
        self.assertIn('mpc.solve_total', summary)
        self.assertNotIn('mpc.native_time_tot_sum', summary)

    def test_failure_still_records_solve_time_and_rejects_controls(self):
        tracker = self.tracker()
        tracker.solver.solve.return_value = 3
        with self.assertRaisesRegex(RuntimeError, 'acados solve failed: status 3'):
            self.solve(tracker)
        tracker._read_controls.assert_not_called()
        self.assertIn('mpc.solve_1', tracker.profiler.drain())


if __name__ == '__main__':
    unittest.main()
