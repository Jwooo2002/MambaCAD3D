"""Analytic CT phantoms test physical geometry independently of any CAD label."""

import unittest
import numpy as np

from scripts.physical_patches import make_patches, physical_to_indices, sample_planes, window_starts


class PhysicalPatchTests(unittest.TestCase):
    def test_direction_origin_and_anisotropic_spacing(self):
        direction = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        index = np.array([[4., 5, 6], [1.5, 2.5, 3.5]])
        origin, spacing = [10, -20, 30], [0.4, 0.7, 1.2]
        physical = np.asarray(origin) + (index * spacing) @ direction.T
        np.testing.assert_allclose(physical_to_indices(physical, origin, spacing, direction), index)

    def test_trilinear_planes_match_analytic_physical_field(self):
        zz, yy, xx = np.indices((20, 30, 40))
        # Physical x=.4*i, y=.7*j, z=1.2*k; exact for linear interpolation.
        volume = (2 * xx * .4 + 3 * yy * .7 - zz * 1.2).astype(np.float32)
        centers = np.array([[8., 10.5, 9.6], [8, 10.5, 10.8]])
        u, v = np.tile([1., 0, 0], (2, 1)), np.tile([0., 1, 0], (2, 1))
        planes, outside = sample_planes(volume, centers, u, v, [0, 0, 0], [.4, .7, 1.2], np.eye(3), 4., 4)
        offsets = np.array([-1.5, -.5, .5, 1.5])
        expected = 2 * (8 + offsets[None, :]) + 3 * (10.5 + offsets[:, None])
        np.testing.assert_allclose(planes[0], expected - 9.6, atol=1e-5)
        np.testing.assert_allclose(planes[1], expected - 10.8, atol=1e-5)
        self.assertEqual(outside, 0)

    def test_tail_window_is_unique_and_short_sequence_is_padded(self):
        self.assertEqual(window_starts(128), [0, 32, 64])
        self.assertEqual(window_starts(100), [0, 32, 36])
        self.assertEqual(window_starts(15), [0])
        protocol = {"depth": 64, "stride": 32, "longitudinal_spacing_mm": .5,
                    "cross_section_fov_mm": 4., "cross_section_pixels": 4}
        path = np.array([[5., 5., 2.], [5., 5., 7.]])
        volume = np.indices((16, 16, 16))[0].astype(np.float32)
        patches, rows, paths, geometry, qc = make_patches(volume, [path, path], [0]*3, [1]*3, np.eye(3), protocol)
        self.assertEqual(patches.shape, (1, 64, 4, 4))
        self.assertEqual(rows[0]["valid_depth"], 11)
        self.assertEqual(qc["duplicate_windows_removed"], 1)
        self.assertEqual(qc["padded_patch_count"], 1)
        np.testing.assert_allclose(patches[0, 10:], 7.)
        self.assertLessEqual(qc["resampled_adjacent_chord_max_mm"], .5 + 1e-12)


if __name__ == "__main__":
    unittest.main()
