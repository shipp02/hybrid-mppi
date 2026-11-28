"""Compatibility between the Mira tracker and the resident log-odds controller."""
from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from test_control_step import controller as c


class MiraRebaseTest(unittest.TestCase):
    def test_raw_waypoint_arrays_keep_linear_reference(self):
        x = jnp.linspace(0., 5., 20)
        course = jnp.column_stack((x, x * 0, x * 0, x * 0 + 1.))
        cfg = c.MPPIConfig(H=3)
        actual = c.build_reference(jnp.array([0., 0., 0., 1.]), course, cfg)
        self.assertEqual(actual.shape, (4, 4))
        np.testing.assert_allclose(actual[:, 1:3], 0.)

    def test_collision_queries_use_measured_grid_pose(self):
        grid = c.og.OccupancyGrid.create()
        grid = replace(grid, pose=jnp.array([2., 3., np.pi / 2]), has_pose=jnp.bool_(True))
        trajectory = jnp.array([[2., 4., np.pi / 2, 1.]])
        local = c.trajectory_to_grid_frame(trajectory, jnp.zeros(4), grid)
        np.testing.assert_allclose(local[0, :3], [1., 0., 0.], atol=1e-6)

    def test_tracking_scan_stays_at_measured_pose(self):
        cfg = c.MPPIConfig(H=3, T=4, REFERENCE='linear')
        grid = c.og.OccupancyGrid.create(
            width=1., height=2., resolution=.2, num_beams=21,
            inflation_radius=.1, distance_cap=.2)
        state = c.ControllerState(
            jax.random.PRNGKey(0), jnp.zeros((3, 2)), jnp.ones((3, 2)))
        observation = c.ControlObservation(
            jnp.array([.2, 0., 0., 1.]), jnp.full(21, 3.), .05, False)
        x = jnp.linspace(0., 5., 20)
        course = jnp.column_stack((x, x * 0, x * 0, x * 0 + 1.))
        measured = jnp.array([0., 0., 0.])
        result, _ = c.tracking_control_step(
            state, observation, course, grid, cfg, scan_pose=measured)
        np.testing.assert_array_equal(result.grid.pose, measured)


if __name__ == '__main__':
    unittest.main()
