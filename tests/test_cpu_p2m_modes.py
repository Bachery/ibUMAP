import unittest

import numpy as np

from ibumap.kernels.cpu import ibfft


def _kwargs():
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


class CPUP2MModesTest(unittest.TestCase):
    def setUp(self):
        self.embedding = np.random.default_rng(20260702).uniform(
            -3.0, 8.0, size=(4096, 2)
        ).astype(np.float32)
        ibfft._clear_fft_kernel_cache()

    def tearDown(self):
        ibfft._clear_fft_kernel_cache()

    def _run(self, mode, **kwargs):
        diagnostics = {}
        force = ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            p2m_mode=mode,
            p2m_diagnostics=diagnostics,
            **kwargs,
            **_kwargs(),
        )
        return force.copy(), diagnostics

    def test_segmented_matches_serial_bitwise(self):
        serial, _ = self._run("serial")
        segmented, diagnostics = self._run("segmented")

        np.testing.assert_array_equal(segmented, serial)
        self.assertEqual(diagnostics["resolved_mode"], "segmented")
        self.assertGreaterEqual(diagnostics["time_s"], 0.0)

    def test_cpu_auto_preserves_benchmarked_serial_default(self):
        _, diagnostics = self._run("auto", deterministic=True)
        self.assertEqual(diagnostics["resolved_mode"], "serial")
        self.assertEqual(diagnostics["reason"], "cpu_serial_benchmark_default")

    def test_deterministic_auto_uses_serial_when_budget_is_insufficient(self):
        serial, _ = self._run("serial")
        fallback, diagnostics = self._run(
            "auto", deterministic=True, workspace_limit_bytes=1
        )

        np.testing.assert_array_equal(fallback, serial)
        self.assertEqual(diagnostics["resolved_mode"], "serial")
        self.assertEqual(diagnostics["reason"], "cpu_serial_benchmark_default")

    def test_minimal_segmented_workspace_is_transient(self):
        workspace = {}
        ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            p2m_mode="segmented",
            workspace_policy="minimal",
            _workspace=workspace,
            **_kwargs(),
        )
        self.assertFalse(any(key.startswith("p2m.") for key in workspace))

    def test_ibfft_reports_exclusive_repulsion_subtimings(self):
        timing = {}
        p2m = {}

        force = ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            p2m_mode="serial",
            p2m_diagnostics=p2m,
            timing_diagnostics=timing,
            **_kwargs(),
        )

        exclusive_keys = (
            "setup_time_s",
            "kernel_build_time_s",
            "kernel_fft_time_s",
            "point_setup_time_s",
            "mesh_clear_time_s",
            "p2m_time_s",
            "fft_plan_build_time_s",
            "fft_forward_time_s",
            "fft_multiply_time_s",
            "fft_inverse_time_s",
            "m2p_time_s",
            "sampling_restore_time_s",
            "ibfft_other_time_s",
        )
        self.assertTrue(np.isfinite(force).all())
        for key in exclusive_keys:
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)
        self.assertAlmostEqual(timing["p2m_time_s"], p2m["time_s"], places=12)
        self.assertAlmostEqual(
            sum(timing[key] for key in exclusive_keys),
            timing["ibfft_total_time_s"],
            places=9,
        )

    @unittest.skipIf(
        ibfft._native_p2m_atomic is None,
        "optional CPU atomic extension is unavailable",
    )
    def test_atomic_extension_produces_finite_force(self):
        force, diagnostics = self._run("atomic")
        self.assertTrue(np.isfinite(force).all())
        self.assertEqual(diagnostics["resolved_mode"], "atomic")


if __name__ == "__main__":
    unittest.main()
