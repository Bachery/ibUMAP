import unittest

import numpy as np

from ibumap.kernels.cpu import ibfft
from ibumap.kernels.fft_grid import FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE
from ibumap.runtime import CPUWorkspace


class IBFFTKernelCacheTest(unittest.TestCase):
    def setUp(self):
        ibfft._clear_fft_kernel_cache()
        rng = np.random.default_rng(7)
        self.embedding = rng.uniform(0.0, 10.0, size=(64, 2)).astype(np.float32)
        self.kwargs = {
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
            "umap_epsilon": 0.001,
            "ibfft_kernel_clip": 4.0,
        }

    def tearDown(self):
        ibfft._clear_fft_kernel_cache()

    def test_different_box_widths_do_not_share_a_kernel(self):
        first = ibfft.ibFFT_repulsive_sampling(self.embedding, **self.kwargs)
        scaled = (self.embedding * np.float32(1.08)).astype(np.float32)
        cached = ibfft.ibFFT_repulsive_sampling(scaled, **self.kwargs).copy()

        self.assertEqual(len(ibfft._fft_kernel_cache), 2)

        ibfft._clear_fft_kernel_cache()
        fresh = ibfft.ibFFT_repulsive_sampling(scaled, **self.kwargs)
        np.testing.assert_array_equal(cached, fresh)
        self.assertTrue(np.isfinite(first).all())

    def test_nearby_spans_share_quantized_kernel_without_shrinking_grid(self):
        workspace = CPUWorkspace()
        first_diagnostics = {}
        second_diagnostics = {}
        ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            _workspace=workspace.buffers,
            fft_kernel_cache=workspace.fft_kernel_cache,
            fft_kernel_cache_policy="byte_lru",
            fft_kernel_cache_limit_bytes=1024 * 1024,
            p2m_diagnostics=first_diagnostics,
            **self.kwargs,
        )
        scaled = (self.embedding * np.float32(0.999)).astype(np.float32)
        cached = ibfft.ibFFT_repulsive_sampling(
            scaled,
            _workspace=workspace.buffers,
            fft_kernel_cache=workspace.fft_kernel_cache,
            fft_kernel_cache_policy="byte_lru",
            fft_kernel_cache_limit_bytes=1024 * 1024,
            p2m_diagnostics=second_diagnostics,
            **self.kwargs,
        ).copy()

        self.assertFalse(first_diagnostics["kernel_cache_hit"])
        self.assertTrue(second_diagnostics["kernel_cache_hit"])
        self.assertEqual(
            first_diagnostics["grid_quantization_level"],
            second_diagnostics["grid_quantization_level"],
        )
        self.assertEqual(len(workspace.fft_kernel_cache), 1)
        for diagnostics in (first_diagnostics, second_diagnostics):
            raw_width = diagnostics["grid_raw_box_width"]
            box_width = diagnostics["grid_box_width"]
            self.assertEqual(diagnostics["kernel_cache_policy"], "byte_lru")
            self.assertEqual(diagnostics["kernel_cache_scope"], "workspace")
            self.assertLessEqual(
                diagnostics["kernel_cache_bytes"],
                diagnostics["kernel_cache_limit_bytes"],
            )
            self.assertGreater(diagnostics["kernel_cache_entry_bytes"], 0)
            self.assertGreaterEqual(box_width, raw_width)
            self.assertLessEqual(
                box_width / raw_width,
                2.0 ** (1.0 / FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE) + 1e-6,
            )
        self.assertEqual(first_diagnostics["kernel_cache_miss_type"], "compulsory")
        self.assertIsNone(second_diagnostics["kernel_cache_miss_type"])

        ibfft._clear_fft_kernel_cache()
        fresh = ibfft.ibFFT_repulsive_sampling(scaled, **self.kwargs)
        np.testing.assert_array_equal(cached, fresh)

    def test_cache_is_bounded_and_uses_lru_eviction(self):
        for index in range(ibfft._FFT_KERNEL_CACHE_MAXSIZE + 2):
            scaled = self.embedding * np.float32(1.0 + 0.03 * index)
            ibfft.ibFFT_repulsive_sampling(scaled, **self.kwargs)

        self.assertEqual(
            len(ibfft._fft_kernel_cache),
            ibfft._FFT_KERNEL_CACHE_MAXSIZE,
        )

    def test_integer_and_float_box_scales_use_same_adaptive_grid(self):
        integer_kwargs = dict(self.kwargs, n_boxes_per_dim=1)
        float_kwargs = dict(self.kwargs, n_boxes_per_dim=1.0)

        integer_force = ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            **integer_kwargs,
        ).copy()
        float_force = ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            **float_kwargs,
        ).copy()

        np.testing.assert_array_equal(integer_force, float_force)

    def test_nonfinite_embedding_is_rejected_at_entry(self):
        embedding = self.embedding.copy()
        embedding[3, 1] = np.nan

        with self.assertRaisesRegex(FloatingPointError, "ibFFT input embedding"):
            ibfft.ibFFT_repulsive_sampling(embedding, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
