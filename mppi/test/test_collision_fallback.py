"""Colliding weighted mean is replaced by the best collision-free sample (kinematic)."""
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller as c  # noqa: E402


class CollisionFallbackTest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(H=20, T=8)
        grid = c.og.OccupancyGrid.create(
            width=3.0, height=7.0, middle=(1.5, 1.0), resolution=0.05, num_beams=811,
            inflation_radius=0.22, distance_cap=0.47)
        ranges = np.full(811, 10.0, np.float32)
        ranges[400:411] = 2.0                       # small object 2 m straight ahead
        self.grid = grid.from_scan(jnp.asarray(ranges))
        self.x0 = jnp.array([0.0, 0.0, 0.0, 3.0])
        self.straight = jnp.zeros((self.cfg.H, 2))
        self.swerve = jnp.zeros((self.cfg.H, 2)).at[:, 1].set(0.35)

    def collides(self, U):
        X = c.simulate(self.x0, U, self.cfg)
        local = c.trajectory_to_car_frame(X[1:], 0.0, 0.0, 0.0)
        return (~self.grid.is_free(local)).astype(jnp.float32)

    def select(self, U, samples, costs):
        obstacle = jax.vmap(self.collides)(samples)
        return c.select_collision_free_controls(
            U, samples, jnp.asarray(costs), obstacle, self.x0, self.cfg, self.grid)

    def test_scenario_is_meaningful(self):
        self.assertGreater(float(self.collides(self.straight).sum()), 0)
        self.assertEqual(float(self.collides(self.swerve).sum()), 0)

    def test_colliding_mean_takes_the_clear_sample(self):
        samples = jnp.stack((self.straight, self.swerve))
        chosen = self.select(self.straight, samples, np.ones((2, self.cfg.H)))
        np.testing.assert_allclose(chosen, self.swerve)

    def test_clear_mean_is_kept(self):
        samples = jnp.stack((self.straight, self.swerve))
        chosen = self.select(self.swerve * 0.9, samples, np.ones((2, self.cfg.H)))
        np.testing.assert_allclose(chosen, self.swerve * 0.9)

    def test_mean_is_kept_when_every_sample_collides(self):
        samples = jnp.stack((self.straight, self.straight))
        chosen = self.select(self.straight, samples, np.ones((2, self.cfg.H)))
        np.testing.assert_allclose(chosen, self.straight)


if __name__ == '__main__':
    unittest.main()
