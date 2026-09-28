from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CUDAP2MSourceContractTest(unittest.TestCase):
    def test_segmented_layout_uses_stable_single_key_argsort(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "cupy_ibfft.py"
        ).read_text(encoding="utf-8")
        layout_source = source.split("def stable_segment_layout", 1)[1]
        layout_source = layout_source.split("_ALLOWED_N_BOXES_PER_DIM", 1)[0]
        self.assertIn("cupy.argsort(cell_ids, kind='stable')", layout_source)
        self.assertIn("cupy.not_equal(", layout_source)
        self.assertIn("cupy.flatnonzero(boundary)", layout_source)
        self.assertIn("if n_points == 0:", layout_source)
        self.assertNotIn("cupy.arange", layout_source)
        self.assertNotIn("cupy.stack", layout_source)
        self.assertNotIn("cupy.lexsort", layout_source)
        self.assertNotIn("cupy.concatenate", layout_source)

    def test_raw_cuda_exports_all_p2m_modes(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        for kernel in (
            "Compute_mat_w_p1_cu",
            "Compute_mat_w_p1_block_atomic_cu",
            "Compute_mat_w_p1_segmented_cu",
            "Compute_mat_w_segmented_cu",
        ):
            self.assertIn(f'extern "C" __global__ void {kernel}', source)

    def test_segmented_kernel_has_no_atomic_accumulation(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        segmented = source.split("Compute_mat_w_p1_segmented_cu", 1)[1]
        segmented = segmented.split('extern "C" __global__ void', 1)[0]
        self.assertNotIn("atomicAdd", segmented)

        generic = source.split("Compute_mat_w_segmented_cu", 1)[1]
        generic = generic.split('extern "C" __global__ void', 1)[0]
        self.assertNotIn("atomicAdd", generic)
        self.assertIn("interpolated_values", generic)
        self.assertIn("nodes_per_box", generic)

    def test_ibfft_exposes_consecutive_cuda_substage_events(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "cupy_ibfft.py"
        ).read_text(encoding="utf-8")
        self.assertIn("timing_event_pairs = None", source)
        self.assertIn("def _timing_mark(name):", source)
        for stage in (
            "setup_time_s",
            "kernel_build_time_s",
            "kernel_fft_time_s",
            "point_setup_time_s",
            "mesh_clear_time_s",
            "p2m_time_s",
            "fft_forward_time_s",
            "fft_multiply_time_s",
            "fft_inverse_time_s",
            "m2p_time_s",
            "sampling_restore_time_s",
        ):
            self.assertIn(f"_timing_mark('{stage}')", source)

    def test_cuda_ibfft_quantizes_grid_and_caches_kernel_spectrum(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "cupy_ibfft.py"
        ).read_text(encoding="utf-8")
        self.assertIn("quantize_fft_box_width", source)
        self.assertIn("fft_kernel_cache", source)
        self.assertIn("kernel_cache_hit", source)
        self.assertIn("_get_cached_fft_kernel", source)
        self.assertIn("_store_fft_kernel", source)
        self.assertIn("FFTKernelLRUCache", source)
        self.assertIn("fft_kernel_cache_limit_bytes", source)
        self.assertIn("DEFAULT_CUDA_FFT_KERNEL_CACHE_MAX_ENTRIES", source)
        self.assertIn("resolved_fft_kernel_cache_max_entries", source)
        self.assertIn("kernel_cache_miss_type", source)
        self.assertIn("kernel_cache_evictions", source)
        self.assertIn("kernel_cache_bytes", source)

    def test_cuda_exports_fused_m2p_update_kernels(self):
        source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        self.assertIn('extern "C" __global__ void NegF_p1_fused_update_cu', source)
        self.assertIn(
            'extern "C" __global__ void PotentialsQij_fused_update_cu', source
        )
        self.assertIn("apply_fused_m2p_update", source)

    def test_cuda_bounds_and_finite_check_use_fused_reduction(self):
        python_source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "cupy_ibfft.py"
        ).read_text(encoding="utf-8")
        cuda_source = (
            ROOT
            / "src"
            / "ibumap"
            / "kernels"
            / "gpu"
            / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")

        self.assertIn("BoundsFinitePartials_cu", python_source)
        self.assertIn("BoundsFiniteFinalize_cu", python_source)
        self.assertIn("bounds_finite_fused_reduction", python_source)
        self.assertIn('extern "C" __global__ void BoundsFinitePartials_cu', cuda_source)
        self.assertIn('extern "C" __global__ void BoundsFiniteFinalize_cu', cuda_source)
        self.assertNotIn("cupy.count_nonzero(~cupy.isfinite(Y)).item()", python_source)
        self.assertNotIn("max_coord = Y.max()", python_source)
        self.assertNotIn("min_coord = Y.min()", python_source)
        self.assertNotIn("min_coord = cupy.float32(min_coord_value)", python_source)
        self.assertIn(
            "min_coord = cupy.asarray(min_coord_value, dtype=cupy.float32)",
            python_source,
        )


if __name__ == "__main__":
    unittest.main()
