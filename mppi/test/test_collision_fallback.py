"""Passing an obstacle on one side when the weighted mean would collide (kinematic)."""
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_step import controller as c  # noqa: E402


class PassOnOneSideTest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(H=20, T=8)
        grid = c.og.OccupancyGrid.create(
            width=3.0, height=7.0, middle=(1.5, 1.0), resolution=0.05, num_beams=811,
            inflation_radius=0.22, distance_cap=0.47)
        ranges = np.full(811, 10.0, np.float32)
        ranges[400:411] = 2.0                       # small object 2 m straight ahead
        self.grid = grid.from_scan(jnp.asarray(ranges))
        self.x0 = jnp.array([0.0, 0.0, 0.0, 3.0])
        steps = np.arange(self.cfg.H + 1) * self.cfg.DT * 3.0
        self.ref = jnp.asarray(np.column_stack((steps, 0 * steps, 0 * steps, 0 * steps + 3.0)))
        self.straight = self.steer(0.0)

    def steer(self, angle):
        return jnp.zeros((self.cfg.H, 2)).at[:, 1].set(angle)

    def choose(self, mean, samples, costs):
        X = c.rollout(self.x0, samples, self.cfg)
        obstacle = jax.vmap(lambda Xs: (~self.grid.is_free(
            c.trajectory_to_car_frame(Xs[1:], 0.0, 0.0, 0.0))).astype(jnp.float32))(X)
        C = jnp.broadcast_to(jnp.asarray(costs, jnp.float32)[:, None], obstacle.shape) / self.cfg.H
        S = jnp.flip(jnp.cumsum(jnp.flip(C, axis=1), axis=1), axis=1)
        weights = c.mppi_weights(S, self.cfg)
        U, _, self.fallback_used = c.pass_on_one_side(mean, weights, samples, X, S, C, obstacle, self.ref,
                                  self.x0, self.cfg, self.grid)
        return U

    def side_of(self, U):
        X = c.simulate(self.x0, U, self.cfg)
        return float(c.reference_side(X[None], self.ref)[0])

    def test_scenario_is_meaningful(self):
        self.assertTrue(bool(c.plan_collides(self.straight, self.x0, self.cfg, self.grid)))
        self.assertFalse(bool(c.plan_collides(self.steer(0.35), self.x0, self.cfg, self.grid)))
        self.assertEqual(self.side_of(self.steer(0.35)), 1.0)
        self.assertEqual(self.side_of(self.steer(-0.35)), -1.0)

    def test_colliding_mean_passes_on_the_side_of_the_best_clear_sample(self):
        samples = jnp.stack((self.straight, self.steer(0.35), self.steer(0.30), self.steer(-0.35)))
        U = self.choose(self.straight, samples, [1.0, 2.0, 2.5, 3.0])  # left is cheapest clear
        self.assertFalse(bool(c.plan_collides(U, self.x0, self.cfg, self.grid)))
        self.assertEqual(self.side_of(U), 1.0)
        self.assertTrue(bool(self.fallback_used))
        # an average of the two left samples, not the right one
        self.assertTrue(0.30 <= float(U[0, 1]) <= 0.35)

    def test_clear_mean_is_kept(self):
        samples = jnp.stack((self.straight, self.steer(0.35)))
        U = self.choose(self.steer(0.32), samples, [1.0, 2.0])
        np.testing.assert_allclose(U, self.steer(0.32))
        self.assertFalse(bool(self.fallback_used))

    def test_mean_is_kept_when_every_sample_collides(self):
        samples = jnp.stack((self.straight, self.straight))
        U = self.choose(self.straight, samples, [1.0, 1.0])
        np.testing.assert_allclose(U, self.straight)
        self.assertFalse(bool(self.fallback_used))

    def test_colliding_plan_brakes_before_contact(self):
        U = c.brake_before_collision(self.straight, self.x0, self.cfg, self.grid)
        # 3 m/s, object 2 m ahead: limited to sqrt(2 * 3 * (~1.85 - 0.3)) ~ 3.0 m/s or less
        self.assertTrue(bool(jnp.all(U[:, 0] <= 0.0)))
        stopped = c.brake_before_collision(self.straight.at[:, 0].set(3.0),
                                           self.x0.at[3].set(0.0), self.cfg, self.grid)
        # from rest the car may still crawl forward
        self.assertTrue(bool(jnp.all(stopped[:, 0] > 0.0)))
        np.testing.assert_allclose(U[:, 1], self.straight[:, 1])

    def test_clear_plan_is_not_braked(self):
        U = self.steer(0.35)
        np.testing.assert_allclose(c.brake_before_collision(U, self.x0, self.cfg, self.grid), U)


    def test_masked_weights_ignore_excluded_samples(self):
        S = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        w = c.mppi_weights(S, self.cfg, jnp.array([False, True, True]))
        np.testing.assert_allclose(w[0], [0.0, 0.0])
        np.testing.assert_allclose(jnp.sum(w, axis=0), [1.0, 1.0], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
