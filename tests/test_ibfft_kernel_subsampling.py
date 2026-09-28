import unittest

import numpy as np

from ibumap.kernels.cpu.ibfft import _build_umap_kernel


class IBFFTKernelSubsamplingTest(unittest.TestCase):
    def setUp(self):
        self.h = 0.25
        indices = np.arange(8, dtype=np.float64)
        self.dist_matrix = (
            indices[:, None] * indices[:, None]
            + indices[None, :] * indices[None, :]
        ) * (self.h * self.h)
        self.params = {
            "umap_a": 1.7,
            "umap_b": 0.9,
            "umap_gamma": 1.0,
            "umap_epsilon": 0.1,
            "ibfft_kernel_clip": 1e6,
        }

    def _build(self, mode=None, radius=2, points=4, **overrides):
        params = {**self.params, **overrides}
        return _build_umap_kernel(
            self.dist_matrix,
            self.h,
            params["umap_a"],
            params["umap_b"],
            params["umap_gamma"],
            params["umap_epsilon"],
            params["ibfft_kernel_clip"],
            mode,
            radius,
            points,
        )

    def test_default_matches_original_point_sampled_formula(self):
        expected = (2.0 * self.params["umap_gamma"] * self.params["umap_b"]) / (
            (self.params["umap_epsilon"] + self.dist_matrix)
            * (
                self.params["umap_a"]
                * np.power(self.dist_matrix, self.params["umap_b"])
                + 1
            )
        )
        expected = np.clip(
            expected,
            -self.params["ibfft_kernel_clip"],
            self.params["ibfft_kernel_clip"],
        )

        np.testing.assert_array_equal(self._build(), expected)
        np.testing.assert_array_equal(self._build(radius=7, points=8), expected)

    def test_radius_zero_changes_only_origin(self):
        point_sampled = self._build()
        subsampled = self._build(mode="near_origin", radius=0, points=4)
        changed = point_sampled != subsampled

        self.assertTrue(changed[0, 0])
        self.assertEqual(np.count_nonzero(changed), 1)

    def test_radius_two_leaves_far_offsets_unchanged(self):
        point_sampled = self._build()
        subsampled = self._build(mode="near_origin", radius=2, points=4)
        rows, cols = np.indices(point_sampled.shape)
        far = np.maximum(rows, cols) > 2

        np.testing.assert_array_equal(subsampled[far], point_sampled[far])
        self.assertTrue(np.any(subsampled[~far] != point_sampled[~far]))

    def test_reasonable_settings_are_finite(self):
        for epsilon in (1e-3, 1e-4, 1e-5):
            for kernel_clip in (1.0, 2.0, 4.0):
                for points in (2, 4, 8):
                    with self.subTest(
                        epsilon=epsilon,
                        kernel_clip=kernel_clip,
                        points=points,
                    ):
                        kernel = self._build(
                            mode="near_origin",
                            radius=2,
                            points=points,
                            umap_epsilon=epsilon,
                            ibfft_kernel_clip=kernel_clip,
                        )
                        self.assertTrue(np.isfinite(kernel).all())

    def test_invalid_subsampling_parameters_are_rejected(self):
        for kwargs in (
            {"mode": "all"},
            {"radius": -1},
            {"points": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self._build(**kwargs)


if __name__ == "__main__":
    unittest.main()
