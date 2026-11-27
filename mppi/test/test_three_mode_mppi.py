"""Three proposals share one rollout budget and never average across modes."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from test_control_step import controller as c


class ThreeModeMPPItest(unittest.TestCase):
    def setUp(self):
        self.cfg = c.MPPIConfig(H=12, T=10, STEER_BIAS=.2, BIAS_DURATION=.3)
        self.U = jnp.zeros((self.cfg.H, 2)).at[:, 1].set(.04)
        self.std = jnp.broadcast_to(jnp.array(self.cfg.U_STD), self.U.shape)
        self.key = jax.random.PRNGKey(19)
        self.x0 = jnp.array([0., 0., 0., 1.])
        grid = c.og.OccupancyGrid.create(
            width=6., height=8., middle=(3., 2.), resolution=.1,
            inflation_radius=.2, distance_cap=.4, num_beams=21)
        self.grid = c.replace(
            grid, log_odds=jnp.full(grid.spec.shape, c.og.L_FREE),
            free_distance=jnp.full(grid.spec.shape, .4))
        self.ref = c.simulate(self.x0, self.U, self.cfg)

    def test_bias_is_symmetric_and_returns_to_nominal_not_countersteering(self):
        proposals = c.build_mode_proposals(self.U, self.cfg)
        np.testing.assert_array_equal(proposals[0], self.U)
        np.testing.assert_array_equal(proposals[:, :, 0], np.zeros((3, self.cfg.H)))
        left = np.asarray(proposals[1, :, 1] - self.U[:, 1])
        right = np.asarray(proposals[2, :, 1] - self.U[:, 1])
        np.testing.assert_allclose(left, -right, atol=1e-7)
        self.assertTrue(np.all(left >= 0.))
        self.assertGreater(left.max(), 0.)
        np.testing.assert_array_equal(left[6:], 0.)
        np.testing.assert_array_equal(proposals[:, 6:, 1], np.full((3, 6), .04, np.float32))

    def test_zero_bias_and_zero_duration_retain_nominal(self):
        for cfg in (c.replace(self.cfg, STEER_BIAS=0.),
                    c.replace(self.cfg, BIAS_DURATION=0.)):
            np.testing.assert_array_equal(
                c.build_mode_proposals(self.U, cfg), np.broadcast_to(self.U, (3, *self.U.shape)))

    def test_proposals_respect_steering_bounds(self):
        cfg = c.replace(self.cfg, STEER_BIAS=1.)
        proposals = np.asarray(c.build_mode_proposals(self.U, cfg))
        self.assertTrue(np.all(proposals[:, :, 1] <= cfg.MAX_STEER))
        self.assertTrue(np.all(proposals[:, :, 1] >= cfg.MIN_STEER))

    def test_batches_preserve_budget_and_each_seed(self):
        proposals = c.build_mode_proposals(self.U, self.cfg)
        original = c.sample_rollout_controls
        with patch.object(c, 'sample_rollout_controls', wraps=original) as sample:
            updates = c.optimize_mppi_modes(
                self.key, proposals, self.std, self.x0, self.ref, self.cfg, self.grid)
        self.assertEqual([call.args[4].T for call in sample.call_args_list], [4, 3, 3])
        self.assertEqual(sum(update.trajectories.shape[0] for update in updates), self.cfg.T)
        for mode, update in enumerate(updates):
            expected = c.simulate(self.x0, proposals[mode], self.cfg)
            np.testing.assert_allclose(update.trajectories[0], expected, atol=1e-6)
            np.testing.assert_allclose(update.weights.sum(axis=0), 1., atol=1e-6)

    def test_selected_mean_covariance_and_weights_are_not_mixed(self):
        proposals = c.build_mode_proposals(self.U, self.cfg)
        updates = c.optimize_mppi_modes(
            self.key, proposals, self.std, self.x0, self.ref, self.cfg, self.grid)
        for selected in range(3):
            flagged = tuple(update._replace(fallback_used=jnp.asarray(mode == 1))
                            for mode, update in enumerate(updates))
            U, std, X, weights, selection, top = c.combine_mode_outputs(
                flagged, jnp.asarray(selected), True)
            np.testing.assert_array_equal(
                selection, [selected == 0, selected == 1, selected == 2, selected == 1])
            np.testing.assert_array_equal(U, updates[selected].controls)
            np.testing.assert_array_equal(std, updates[selected].std)
            self.assertEqual(X.shape, (self.cfg.T, self.cfg.H + 1, self.cfg.NX))
            self.assertEqual(top.xy.shape[0], self.cfg.T)
            offset = 0
            for mode, update in enumerate(updates):
                count = update.weights.shape[0]
                expected = update.weights if mode == selected else np.zeros_like(update.weights)
                np.testing.assert_array_equal(weights[offset:offset + count], expected)
                offset += count

    def test_plan_selection_uses_raw_reevaluated_cost(self):
        proposals = c.build_mode_proposals(self.U, self.cfg)
        updates = c.optimize_mppi_modes(
            self.key, proposals, self.std, self.x0, self.ref, self.cfg, self.grid)
        for winner in range(3):
            # Stored sample costs must not determine whole-plan scores.
            candidates = tuple(update._replace(
                controls=self.U if mode == winner else self.U.at[:, 0].set(2.),
                costs=jnp.full_like(update.costs, -1000. if mode != winner else 1000.))
                for mode, update in enumerate(updates))
            selected = c.select_mode_plan(candidates, self.x0, self.ref, self.cfg, self.grid)
            self.assertEqual(int(selected), winner)

    def test_symmetric_obstacle_selects_one_side_not_average(self):
        cfg = c.replace(self.cfg, H=20, T=12, STEER_BIAS=.3, BIAS_DURATION=.6)
        U = jnp.zeros((cfg.H, 2))
        std = jnp.full_like(U, .001)
        x0 = self.x0.at[3].set(2.)
        ref = c.simulate(x0, U, cfg)
        rows, cols = jnp.indices(self.grid.spec.shape)
        forward = (rows + .5) * self.grid.spec.resolution - self.grid.spec.middle[1]
        left = (cols + .5) * self.grid.spec.resolution - self.grid.spec.middle[0]
        distance = jnp.minimum(jnp.hypot(forward - 1.2, left), self.grid.spec.distance_cap)
        grid = c.replace(self.grid, free_distance=distance)
        self.assertTrue(bool(c.plan_collides(U, x0, cfg, grid)))
        selected, _, _, weights, selection = c.mppi_step_cov(self.key, U, std, x0, ref, cfg, grid)
        self.assertEqual(selection.shape, (4,))
        self.assertFalse(bool(c.plan_collides(selected, x0, cfg, grid)))
        active_modes = np.asarray(weights).reshape(3, 4, cfg.H).sum(axis=(1, 2)) > 0
        self.assertEqual(active_modes.sum(), 1)
        self.assertFalse(active_modes[0])

    def test_compiled_update_requires_no_host_transfers(self):
        args = self.key, self.U, self.std, self.x0, self.ref, self.cfg, self.grid
        expected = c.mppi_step_cov(*args)
        jax.block_until_ready(expected)
        with jax.transfer_guard('disallow'):
            actual = c.mppi_step_cov(*args)
            jax.block_until_ready(actual)
        self.assertEqual(actual[2].shape[0], self.cfg.T)
        self.assertEqual(actual[0].devices(), self.U.devices())
        self.assertEqual(actual[4].devices(), self.U.devices())
        self.assertEqual(actual[4].shape, (4,))
        self.assertEqual(int(actual[4][:3].sum()), 1)
        np.testing.assert_allclose(actual[3].sum(axis=0), 1., atol=1e-6)

    def test_dynamic_fallback_flag_reports_mean_rejection(self):
        cfg = c.replace(self.cfg, DYNAMICS='dynamic', H=6)
        x0 = jnp.array([0., 0., 0., 2., 0., 0., 0., 2.])
        good = jnp.zeros((cfg.H, cfg.NU))
        bad = good.at[:, 1].set(.4)
        samples = jnp.stack((bad, good))
        trajectories = c.rollout(x0, samples, cfg)
        costs = jnp.zeros((2, cfg.H))
        for mean, expected_fallback in ((bad, True), (good, False)):
            selected, fallback = c.select_feasible_controls(
                mean, samples, trajectories, costs, x0, cfg)
            np.testing.assert_array_equal(selected, good)
            self.assertEqual(bool(fallback), expected_fallback)

    def test_small_budget_is_rejected_at_optimizer_entry(self):
        with self.assertRaisesRegex(ValueError, 'num_samples >= 3'):
            c.mppi_update(self.key, self.U, self.std, self.x0, self.ref,
                          c.replace(self.cfg, T=2), self.grid)


if __name__ == '__main__':
    unittest.main()
