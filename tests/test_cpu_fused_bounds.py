import unittest
from unittest import mock

import numpy as np

from ibumap.kernels.cpu import ibfft


def _kernel_kwargs():
    return {
        "n_interpolation_points": 1,
        "intervals_per_integer": 1.0,
        "min_num_intervals": 32,
        "gamma": 2.0,
        "paraFactor": 2.0,
        "n_boxes_per_dim": 1.0,
        "kernel_method": "UMAP_kernel",
        "umap_a": 1.57,
        "umap_b": 0.895,
        "umap_gamma": 1.0,
        "umap_epsilon": 1e-3,
        "ibfft_kernel_clip": 4.0,
    }


class CPUFusedBoundsTest(unittest.TestCase):
    def test_reduction_matches_numpy_and_reuses_workspace(self):
        rng = np.random.default_rng(17)
        for dtype in (np.float32, np.float64):
            values = rng.normal(size=(200_003, 2)).astype(dtype)
            workspace = {}

            first = ibfft._fused_bounds_finite(values, workspace)
            scratch_ids = {
                key: id(value)
                for key, value in workspace.items()
                if key.startswith(ibfft._CPU_BOUNDS_WORKSPACE_PREFIX)
            }
            second = ibfft._fused_bounds_finite(values, workspace)

            self.assertEqual(first[:3], (values.min(), values.max(), 0))
            self.assertEqual(second, first)
            self.assertEqual(
                scratch_ids,
                {
                    key: id(value)
                    for key, value in workspace.items()
                    if key.startswith(ibfft._CPU_BOUNDS_WORKSPACE_PREFIX)
                },
            )

    def test_reduction_counts_all_nonfinite_values(self):
        values = np.array(
            [[-4.0, np.nan], [np.inf, 8.0], [-np.inf, 2.0]],
            dtype=np.float32,
        )

        minimum, maximum, bad, _ = ibfft._fused_bounds_finite(values)

        self.assertEqual(minimum, np.float32(-4.0))
        self.assertEqual(maximum, np.float32(8.0))
        self.assertEqual(bad, 3)

    def test_large_contiguous_route_is_reported_and_retains_scratch(self):
        embedding = np.random.default_rng(7).uniform(
            0.0, 10.0, size=(128, 2)
        ).astype(np.float32)
        diagnostics = {}
        workspace = {}

        with mock.patch.object(ibfft, "_CPU_FUSED_BOUNDS_MIN_POINTS", 1):
            force = ibfft.ibFFT_repulsive_sampling(
                embedding,
                _workspace=workspace,
                p2m_diagnostics=diagnostics,
                **_kernel_kwargs(),
            )

        self.assertEqual(force.shape, embedding.shape)
        self.assertTrue(np.all(np.isfinite(force)))
        self.assertTrue(diagnostics["bounds_finite_fused_reduction"])
        self.assertEqual(diagnostics["bounds_finite_backend"], "fixed_chunk_numba")
        self.assertIn(
            f"{ibfft._CPU_BOUNDS_WORKSPACE_PREFIX}min",
            workspace,
        )

    def test_fused_route_preserves_full_kernel_output_with_and_without_sampling(self):
        embedding = np.random.default_rng(8).uniform(
            0.0, 10.0, size=(128, 2)
        ).astype(np.float32)
        probabilities = np.full(embedding.shape[0], 0.75)

        for sampled in (False, True):
            call_kwargs = _kernel_kwargs()
            if sampled:
                call_kwargs.update(
                    probabilities=probabilities,
                    random_state=np.random.RandomState(19),
                )
            with mock.patch.object(
                ibfft, "_CPU_FUSED_BOUNDS_MIN_POINTS", embedding.shape[0] + 1
            ):
                baseline = ibfft.ibFFT_repulsive_sampling(
                    embedding.copy(),
                    _workspace={},
                    **call_kwargs,
                )

            if sampled:
                call_kwargs["random_state"] = np.random.RandomState(19)
            with mock.patch.object(ibfft, "_CPU_FUSED_BOUNDS_MIN_POINTS", 1):
                candidate = ibfft.ibFFT_repulsive_sampling(
                    embedding.copy(),
                    _workspace={},
                    **call_kwargs,
                )

            np.testing.assert_array_equal(candidate, baseline)

    def test_noncontiguous_input_keeps_numpy_fallback(self):
        embedding = np.random.default_rng(9).uniform(
            0.0, 10.0, size=(128, 2)
        ).astype(np.float32)[:, ::-1]
        diagnostics = {}

        with mock.patch.object(ibfft, "_CPU_FUSED_BOUNDS_MIN_POINTS", 1):
            ibfft.ibFFT_repulsive_sampling(
                embedding,
                p2m_diagnostics=diagnostics,
                **_kernel_kwargs(),
            )

        self.assertFalse(diagnostics["bounds_finite_fused_reduction"])
        self.assertEqual(diagnostics["bounds_finite_backend"], "numpy")

    def test_minimal_policy_does_not_retain_bounds_scratch(self):
        embedding = np.random.default_rng(11).uniform(
            0.0, 10.0, size=(128, 2)
        ).astype(np.float32)
        workspace = {}

        with mock.patch.object(ibfft, "_CPU_FUSED_BOUNDS_MIN_POINTS", 1):
            ibfft.ibFFT_repulsive_sampling(
                embedding,
                _workspace=workspace,
                workspace_policy="minimal",
                **_kernel_kwargs(),
            )

        self.assertFalse(
            any(
                key.startswith(ibfft._CPU_BOUNDS_WORKSPACE_PREFIX)
                for key in workspace
            )
        )


if __name__ == "__main__":
    unittest.main()
