"""CPU-only classification/clustering tests, independent of ROS and JAX."""
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from map_obstacles import (ClusterConfig, MapMatcher, STATIC, UNMAPPED, UNCERTAIN,
                           cluster_scan, scan_points)


class MapMatchingTests(unittest.TestCase):
    def setUp(self):
        self.cells = np.zeros((10, 10), dtype=np.int8)
        self.cells[:, 0] = 100
        self.cells[8, 8] = -1

    def test_wall_free_unknown_outside_and_invalid(self):
        matcher = MapMatcher(self.cells, 0.1, (0, 0, 0), allowance=0.04)
        labels = matcher.classify(np.array([[0.05, .25], [.15, .25], [.55, .25],
                                           [.85, .85], [1.5, .25], [np.nan, 0]]))
        np.testing.assert_array_equal(labels, [STATIC, STATIC, UNMAPPED,
                                               UNCERTAIN, UNCERTAIN, UNCERTAIN])

    def test_rotated_translated_origin(self):
        matcher = MapMatcher(self.cells, 0.1, (2, 3, np.pi / 2), allowance=0)
        labels = matcher.classify(np.array([[1.75, 3.05], [1.75, 3.55]]))
        np.testing.assert_array_equal(labels, [STATIC, UNMAPPED])

    def test_empty_occupied_mask_has_no_phantom_wall(self):
        matcher = MapMatcher(np.zeros((3, 3)), 1, (0, 0, 0))
        self.assertEqual(matcher.classify(np.array([[.5, .5]]))[0], UNMAPPED)

    def test_unknown_near_wall_can_match(self):
        self.cells[2, 1] = -1
        matcher = MapMatcher(self.cells, .1, (0, 0, 0), allowance=.04)
        self.assertEqual(matcher.classify(np.array([[.15, .25]]))[0], STATIC)

    def test_threshold_and_map_replacement(self):
        cells = np.array([[0, 64, 65]])
        matcher = MapMatcher(cells, 1, (0, 0, 0), allowance=0)
        np.testing.assert_array_equal(matcher.classify(np.array([[.5, .5], [2.5, .5]])),
                                      [UNMAPPED, STATIC])

    def test_invalid_configuration(self):
        for resolution in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                MapMatcher(self.cells, resolution, (0, 0, 0))


class ClusteringTests(unittest.TestCase):
    def groups(self, ranges, cfg=None, labels=None, increment=.01):
        ranges = np.asarray(ranges, dtype=float)
        points, valid = scan_points(ranges, 0, increment, .1, 10)
        if labels is None:
            labels = np.full(len(ranges), UNMAPPED)
        groups = cluster_scan(points, ranges, labels, valid, increment,
                              cfg or ClusterConfig(min_points=1))
        return [g.tolist() for g in groups]

    def test_two_nans_bridge_without_fabricated_points(self):
        self.assertEqual(self.groups([2, 2, np.nan, np.nan, 2, 2]), [[0, 1, 4, 5]])

    def test_excess_gap_splits(self):
        self.assertEqual(self.groups([2, np.nan, np.nan, np.nan, 2]), [[0], [4]])

    def test_inf_splits_by_default_and_can_be_enabled(self):
        self.assertEqual(self.groups([2, np.inf, 2]), [[0], [2]])
        cfg = ClusterConfig(min_points=1, bridge_inf=True)
        self.assertEqual(self.groups([2, np.inf, 2], cfg), [[0, 2]])

    def test_nan_bridging_can_be_disabled(self):
        cfg = ClusterConfig(min_points=1, bridge_nan=False)
        self.assertEqual(self.groups([2, np.nan, 2], cfg), [[0], [2]])

    def test_valid_static_and_uncertain_hits_split(self):
        for middle in (STATIC, UNCERTAIN):
            self.assertEqual(self.groups([2, 2, 2], labels=[UNMAPPED, middle, UNMAPPED]),
                             [[0], [2]])

    def test_invalid_finite_and_negative_inf_split(self):
        for middle in (0, .01, 11, -np.inf):
            self.assertEqual(self.groups([2, middle, 2]), [[0], [2]])

    def test_large_physical_gap_splits(self):
        self.assertEqual(self.groups([2, np.nan, 3]), [[0], [2]])

    def test_angle_cap_splits(self):
        cfg = ClusterConfig(min_points=1, max_angle_gap=.015)
        self.assertEqual(self.groups([2, np.nan, 2], cfg), [[0], [2]])

    def test_min_points_and_edges(self):
        cfg = ClusterConfig(min_points=3)
        self.assertEqual(self.groups([np.nan, 2, 2, 2, np.nan], cfg), [[1, 2, 3]])
        self.assertEqual(self.groups([2, 2], cfg), [])
        self.assertEqual(self.groups([]), [])
        self.assertEqual(self.groups([np.nan, np.inf]), [])

    def test_reversed_scan(self):
        self.assertEqual(self.groups([2, np.nan, 2], increment=-.01), [[0, 2]])

    def test_hard_distance_cap(self):
        cfg = ClusterConfig(min_points=1, base_gap=2, max_link_distance=.1)
        self.assertEqual(self.groups([2, 2.2], cfg), [[0], [1]])

    def test_invalid_cluster_settings(self):
        for kwargs in ({'max_invalid_gap': -1}, {'min_points': 0}, {'base_gap': np.nan}):
            with self.assertRaises(ValueError):
                ClusterConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
