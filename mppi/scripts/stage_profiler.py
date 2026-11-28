"""Bounded host-side timing for diagnosing control-loop latency.

Nested stages are inclusive, not additive. GPU waits happen only when profiling
is enabled, so completed work can be separated from host readback. Native solver
statistics are also inclusive and must not be added to wall-clock stages.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
import time

import numpy as np


class StageProfiler:
    def __init__(self, enabled=False, capacity=256):
        self.enabled = enabled
        self._samples = defaultdict(lambda: deque(maxlen=capacity))

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    def record(self, name, seconds):
        if self.enabled and np.isfinite(seconds) and seconds >= 0.:
            self._samples[name].append(float(seconds))

    def wait(self, values):
        """Do not introduce synchronization in the normal, unprofiled path."""
        if self.enabled:
            import jax
            jax.block_until_ready(values)

    def drain(self):
        """Return per-stage count/mean/p95/max in ms and start a fresh window."""
        summary = {}
        for name, values in self._samples.items():
            if values:
                milliseconds = np.asarray(values) * 1e3
                summary[name] = (len(values), float(milliseconds.mean()),
                                 float(np.percentile(milliseconds, 95)), float(milliseconds.max()))
        self._samples.clear()
        return summary

    def log_summary(self, logger):
        summary = self.drain()
        if not summary:
            return
        stages = '; '.join(f'{name}[n={count}]={mean:.2f}/{p95:.2f}/{maximum:.2f}'
                           for name, (count, mean, p95, maximum) in summary.items())
        logger.info(f'Control stages ms mean/p95/max (nested stages inclusive): {stages}')
