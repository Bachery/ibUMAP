from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.kernels.cpu import ibfft  # noqa: E402


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _force_kwargs(p: int) -> dict:
    return {
        "n_interpolation_points": p,
        "intervals_per_integer": 1.0,
        "min_num_intervals": 32,
        "gamma": 2.0,
        "paraFactor": 2.0,
        "probabilities": None,
        "n_boxes_per_dim": 1.0,
        "kernel_method": "UMAP_kernel",
        "umap_a": 1.57,
        "umap_b": 0.895,
        "umap_gamma": 1.0,
        "umap_epsilon": 1e-3,
        "ibfft_kernel_clip": 4.0,
    }


@unittest.skipUnless(_cuda_available(), "cupy/CUDA GPU is unavailable")
class CUDAIBFFTContractTest(unittest.TestCase):
    def test_deterministic_workspace_reuses_plans_and_fft_buffers(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = (
            np.random.default_rng(20260806)
            .uniform(-3.7, 8.9, size=(512, 2))
            .astype(np.float32)
        )
        kwargs = _force_kwargs(1)
        baseline = cp.asnumpy(
            ibFFT_repulsive_sampling_GPU(
                cp.asarray(embedding),
                deterministic=False,
                p2m_mode="segmented",
                workspace=GPUWorkspace(),
                workspace_policy="performance",
                **kwargs,
            )
        )
        workspace = GPUWorkspace()
        outputs = []
        diagnostics = []
        for _ in range(2):
            current_diagnostics = {}
            outputs.append(
                cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        deterministic=True,
                        p2m_mode="segmented",
                        workspace=workspace,
                        workspace_policy="performance",
                        p2m_diagnostics=current_diagnostics,
                        **kwargs,
                    )
                )
            )
            diagnostics.append(current_diagnostics)

        np.testing.assert_array_equal(outputs[0], outputs[1])
        relative_l2 = np.linalg.norm(
            outputs[0].astype(np.float64) - baseline.astype(np.float64)
        ) / max(np.linalg.norm(baseline.astype(np.float64)), 1e-30)
        self.assertLessEqual(relative_l2, 5e-6)
        self.assertEqual(diagnostics[0]["fft_backend"], "workspace_planned_buffers")
        self.assertFalse(diagnostics[0]["fft_plan_cache_hit"])
        self.assertTrue(diagnostics[1]["fft_plan_cache_hit"])
        self.assertEqual(workspace.fft_snapshot()["plan_entries"], 2)
        self.assertEqual(workspace.fft_snapshot()["plan_hits"], 2)
        self.assertEqual(workspace.fft_snapshot()["plan_misses"], 2)
        self.assertIn("ibfft.fft.frequency", workspace.buffers)
        self.assertIn("ibfft.fft.output", workspace.buffers)

    def test_fast_and_minimal_paths_keep_cupy_fft_wrapper(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = (
            np.random.default_rng(20260806)
            .uniform(-3.7, 8.9, size=(128, 2))
            .astype(np.float32)
        )
        for deterministic, policy in ((False, "performance"), (True, "minimal")):
            with self.subTest(deterministic=deterministic, policy=policy):
                workspace = GPUWorkspace()
                diagnostics = {}
                ibFFT_repulsive_sampling_GPU(
                    cp.asarray(embedding),
                    deterministic=deterministic,
                    p2m_mode="segmented",
                    workspace=workspace,
                    workspace_policy=policy,
                    p2m_diagnostics=diagnostics,
                    **_force_kwargs(1),
                )
                self.assertEqual(diagnostics["fft_backend"], "cupy_wrapper")
                self.assertEqual(workspace.fft_plans, {})
                self.assertNotIn("ibfft.fft.frequency", workspace.buffers)
                self.assertNotIn("ibfft.fft.output", workspace.buffers)

    def test_stable_segment_layout_matches_lexsort_reference(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import stable_segment_layout

        cases = (
            np.empty(0, dtype=np.int64),
            np.zeros(257, dtype=np.int64),
            np.tile(np.array([3, 1, 3, 2, 1], dtype=np.int64), 73),
            np.random.default_rng(20260805).integers(
                0, 19, size=2048, dtype=np.int64
            ),
            np.arange(1024, dtype=np.int64)[::-1],
        )
        for cell_ids in cases:
            with self.subTest(n_points=cell_ids.size):
                if cell_ids.size == 0:
                    expected = tuple(np.empty(0, dtype=np.int64) for _ in range(4))
                else:
                    point_ids = np.arange(cell_ids.size, dtype=np.int64)
                    order = np.lexsort((point_ids, cell_ids)).astype(
                        np.int64, copy=False
                    )
                    sorted_cells = cell_ids[order]
                    starts = np.flatnonzero(
                        np.concatenate(
                            (
                                np.ones(1, dtype=np.bool_),
                                sorted_cells[1:] != sorted_cells[:-1],
                            )
                        )
                    ).astype(np.int64, copy=False)
                    ends = np.concatenate(
                        (starts[1:], np.asarray([cell_ids.size], dtype=np.int64))
                    )
                    expected = (order, sorted_cells[starts], starts, ends)

                actual = tuple(
                    cp.asnumpy(values)
                    for values in stable_segment_layout(cp.asarray(cell_ids))
                )
                for actual_values, expected_values in zip(actual, expected):
                    np.testing.assert_array_equal(actual_values, expected_values)

    def test_fused_m2p_update_matches_cuda_fallback(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        base = np.random.default_rng(20260706).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        rng = np.random.default_rng(20260707)
        attr = rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
        neg_effects = rng.uniform(0.1, 1.5, size=base.shape[0]).astype(np.float32)
        alpha = np.float32(0.7)
        clip_norm = np.float32(0.45)

        for p in (1, 2, 3):
            with self.subTest(p=p):
                expected = cp.asarray(base)
                raw_force = ibFFT_repulsive_sampling_GPU(
                    expected,
                    workspace=GPUWorkspace(),
                    **_force_kwargs(p),
                )
                raw_force *= alpha
                raw_force *= cp.asarray(neg_effects).reshape((-1, 1))
                norms = cp.sqrt(cp.sum(raw_force * raw_force, axis=1))
                scales = cp.minimum(
                    cp.float32(1.0), clip_norm / (norms + cp.float32(1e-12))
                )
                raw_force *= scales.reshape((-1, 1))
                expected += cp.asarray(attr) + raw_force

                fused = cp.asarray(base)
                diagnostics = {}
                result = ibFFT_repulsive_sampling_GPU(
                    fused,
                    workspace=GPUWorkspace(),
                    p2m_diagnostics=diagnostics,
                    fused_update={
                        "embedding": fused,
                        "attr_force": cp.asarray(attr),
                        "alpha": alpha,
                        "neg_effects": cp.asarray(neg_effects),
                        "clip_norm": clip_norm,
                    },
                    **_force_kwargs(p),
                )

                self.assertIsNone(result)
                self.assertTrue(diagnostics["fused_m2p_update"])
                cp.testing.assert_allclose(fused, expected, rtol=2e-5, atol=2e-5)

    def test_nearby_spans_reuse_workspace_kernel_spectrum(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(20260706).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        workspace = GPUWorkspace()
        first_diagnostics = {}
        second_diagnostics = {}

        ibFFT_repulsive_sampling_GPU(
            cp.asarray(embedding),
            workspace=workspace,
            p2m_diagnostics=first_diagnostics,
            **_force_kwargs(1),
        )
        ibFFT_repulsive_sampling_GPU(
            cp.asarray(embedding * np.float32(0.999)),
            workspace=workspace,
            p2m_diagnostics=second_diagnostics,
            **_force_kwargs(1),
        )

        self.assertFalse(first_diagnostics["kernel_cache_hit"])
        self.assertTrue(second_diagnostics["kernel_cache_hit"])
        self.assertEqual(first_diagnostics["kernel_cache_policy"], "byte_lru")
        self.assertEqual(first_diagnostics["kernel_cache_scope"], "workspace")
        self.assertEqual(first_diagnostics["kernel_cache_miss_type"], "compulsory")
        self.assertIsNone(second_diagnostics["kernel_cache_miss_type"])
        self.assertGreater(first_diagnostics["kernel_cache_entry_bytes"], 0)
        self.assertGreaterEqual(second_diagnostics["kernel_cache_bytes"], 0)
        self.assertTrue(first_diagnostics["bounds_finite_fused_reduction"])
        self.assertTrue(second_diagnostics["bounds_finite_fused_reduction"])
        self.assertEqual(len(workspace.fft_kernel_cache), 1)

    def test_nonfinite_embedding_is_rejected_at_entry(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = cp.zeros((32, 2), dtype=cp.float32)
        embedding[3, 1] = cp.nan

        with self.assertRaisesRegex(FloatingPointError, "ibFFT input embedding"):
            ibFFT_repulsive_sampling_GPU(embedding, **_force_kwargs(1))

    def test_direct_force_accepts_python_scalars_and_matches_cpu(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = np.random.default_rng(20260622).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        for p in (1, 2, 3):
            with self.subTest(p=p):
                kwargs = _force_kwargs(p)
                ibfft._clear_fft_kernel_cache()
                cpu_force = ibfft.ibFFT_repulsive_sampling(embedding, **kwargs)
                gpu_force = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(cp.asarray(embedding), **kwargs)
                )
                np.testing.assert_allclose(
                    gpu_force,
                    cpu_force,
                    rtol=5e-2,
                    atol=5e-3,
                )

    def test_workspace_policies_preserve_direct_force_parity(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(20260623).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        kwargs = _force_kwargs(3)
        ibfft._clear_fft_kernel_cache()
        cpu_force = ibfft.ibFFT_repulsive_sampling(embedding, **kwargs)

        for policy in ("fast", "low_memory"):
            with self.subTest(policy=policy):
                workspace = GPUWorkspace()
                gpu_force = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        workspace=workspace,
                        workspace_policy=policy,
                        **kwargs,
                    )
                )
                np.testing.assert_allclose(
                    gpu_force,
                    cpu_force,
                    rtol=5e-2,
                    atol=5e-3,
                )
                self.assertIn("ibfft.ChargesQij", workspace.buffers)
                self.assertTrue(
                    any(
                        key.startswith("ibfft.mat_w.p3.")
                        for key in workspace.buffers
                    )
                )

    def test_fft_input_workspace_is_contiguous_term_first(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(31).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        workspace = GPUWorkspace()

        ibFFT_repulsive_sampling_GPU(
            cp.asarray(embedding),
            workspace=workspace,
            workspace_policy="fast",
            **_force_kwargs(2),
        )

        mat_w = next(
            value
            for key, value in workspace.buffers.items()
            if key.startswith("ibfft.mat_w.p2.")
        )
        self.assertEqual(mat_w.shape[0], 3)
        self.assertTrue(mat_w.flags.c_contiguous)

    def test_p1_fast_path_skips_interpolation_workspace_buffers(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(20260623).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)
        workspace = GPUWorkspace()

        ibFFT_repulsive_sampling_GPU(
            cp.asarray(embedding),
            workspace=workspace,
            workspace_policy="fast",
            **_force_kwargs(1),
        )

        self.assertNotIn("ibfft.y_in_box", workspace.buffers)
        self.assertNotIn("ibfft.interpolate_values.p1", workspace.buffers)
        self.assertNotIn("ibfft.denominator.p1", workspace.buffers)
        self.assertFalse(
            any(
                key.startswith("ibfft.w_coefficients.p1.")
                for key in workspace.buffers
            )
        )
        self.assertNotIn("ibfft.ChargesQij", workspace.buffers)
        self.assertNotIn("ibfft.potentialsQij", workspace.buffers)
        self.assertIn("ibfft.box_idx", workspace.buffers)
        self.assertTrue(
            any(key.startswith("ibfft.mat_w.p1.") for key in workspace.buffers)
        )

    def test_higher_order_paths_still_use_interpolation_workspace(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(20260624).uniform(
            -3.7, 8.9, size=(192, 2)
        ).astype(np.float32)

        for p in (2, 3):
            with self.subTest(p=p):
                workspace = GPUWorkspace()
                ibFFT_repulsive_sampling_GPU(
                    cp.asarray(embedding),
                    workspace=workspace,
                    workspace_policy="fast",
                    **_force_kwargs(p),
                )
                self.assertIn("ibfft.y_in_box", workspace.buffers)
                self.assertIn(f"ibfft.interpolate_values.p{p}", workspace.buffers)
                self.assertIn(f"ibfft.denominator.p{p}", workspace.buffers)
                self.assertIn("ibfft.ChargesQij", workspace.buffers)
                self.assertIn("ibfft.potentialsQij", workspace.buffers)
                self.assertTrue(
                    any(
                        key.startswith(f"ibfft.w_coefficients.p{p}.")
                        for key in workspace.buffers
                    )
                )

    def test_dynamic_launch_geometry_smoke_for_small_default_and_larger_n(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import (
            ibFFT_repulsive_sampling_GPU,
            point_launch,
        )

        self.assertEqual(point_launch(1)[0], (1,))
        self.assertEqual(point_launch(192)[0], (1,))
        self.assertGreater(point_launch(1024)[0][0], point_launch(192)[0][0])

        for n in (8, 192, 1024):
            with self.subTest(n=n):
                embedding = np.random.default_rng(1000 + n).uniform(
                    -3.7, 8.9, size=(n, 2)
                ).astype(np.float32)
                force = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        **_force_kwargs(1),
                    )
                )
                self.assertEqual(force.shape, embedding.shape)
                self.assertTrue(np.isfinite(force).all())

    def test_minimal_sampling_workspace_tracks_active_capacity(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(23).uniform(
            -2.5, 6.75, size=(192, 2)
        ).astype(np.float32)
        probabilities = cp.full(embedding.shape[0], 0.25, dtype=cp.float32)
        workspace = GPUWorkspace()
        active_counts = []
        kwargs = _force_kwargs(1)
        kwargs["probabilities"] = probabilities

        for seed in (101, 102):
            y_gpu = cp.asarray(embedding)
            y_before = cp.asnumpy(y_gpu)
            rng = cp.random.RandomState(seed)
            active_counts.append(
                int(cp.count_nonzero(rng.random_sample(embedding.shape[0]) <= probabilities).get())
            )
            force = ibFFT_repulsive_sampling_GPU(
                y_gpu,
                random_state=cp.random.RandomState(seed),
                workspace=workspace,
                workspace_policy="minimal",
                **kwargs,
            )
            self.assertEqual(force.shape, embedding.shape)
            np.testing.assert_array_equal(cp.asnumpy(y_gpu), y_before)
            self.assertEqual(
                workspace.buffers["ibfft.box_idx"].shape[0],
                active_counts[-1],
            )
            self.assertEqual(
                workspace.buffers["ibfft.neg_f"].shape[0],
                active_counts[-1],
            )

        self.assertLess(
            workspace.buffers["ibfft.box_idx"].shape[0],
            embedding.shape[0],
        )
        self.assertNotIn("ibfft.ChargesQij", workspace.buffers)
        self.assertNotIn("ibfft.potentialsQij", workspace.buffers)

    def test_segmented_matches_serial_and_records_workspace_for_all_orders(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(91).uniform(
            -2.5, 6.75, size=(512, 2)
        ).astype(np.float32)
        for p in (1, 2, 3):
            with self.subTest(p=p):
                serial = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        p2m_mode="serial",
                        **_force_kwargs(p),
                    )
                )
                diagnostics = {}
                workspace = GPUWorkspace()
                segmented = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        p2m_mode="segmented",
                        p2m_diagnostics=diagnostics,
                        workspace=workspace,
                        workspace_policy="performance",
                        **_force_kwargs(p),
                    )
                )

                np.testing.assert_array_equal(segmented, serial)
                self.assertEqual(diagnostics["resolved_mode"], "segmented")
                self.assertIn("ibfft.p2m.order", workspace.buffers)
                if p == 1:
                    self.assertIn("ibfft.p2m.segment_cells", workspace.buffers)
                else:
                    self.assertIn("ibfft.p2m.segment_boxes", workspace.buffers)
                    self.assertNotIn("ibfft.ChargesQij", workspace.buffers)
                    self.assertFalse(
                        any(
                            key.startswith("ibfft.w_coefficients.")
                            for key in workspace.buffers
                        )
                    )

    def test_segmented_reuses_retained_workspace_arrays_for_all_orders(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
        from ibumap.runtime import GPUWorkspace

        embedding = np.random.default_rng(20260722).uniform(
            -2.5, 6.75, size=(512, 2)
        ).astype(np.float32)
        for p in (1, 2, 3):
            with self.subTest(p=p):
                workspace = GPUWorkspace()
                first = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        p2m_mode="segmented",
                        workspace=workspace,
                        workspace_policy="performance",
                        **_force_kwargs(p),
                    )
                )
                retained = {
                    key: value
                    for key, value in workspace.buffers.items()
                    if key.startswith("ibfft.p2m.")
                }

                second = cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        p2m_mode="segmented",
                        workspace=workspace,
                        workspace_policy="performance",
                        **_force_kwargs(p),
                    )
                )

                np.testing.assert_array_equal(second, first)
                self.assertTrue(retained)
                for key, value in retained.items():
                    self.assertIs(workspace.buffers[key], value, key)

    def test_higher_order_segmented_is_exactly_repeatable(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = np.random.default_rng(911).uniform(
            -2.5, 6.75, size=(512, 2)
        ).astype(np.float32)
        for p in (2, 3):
            with self.subTest(p=p):
                outputs = [
                    cp.asnumpy(
                        ibFFT_repulsive_sampling_GPU(
                            cp.asarray(embedding),
                            p2m_mode="segmented",
                            deterministic=True,
                            **_force_kwargs(p),
                        )
                    )
                    for _ in range(2)
                ]
                np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_block_atomic_rejects_higher_order_interpolation(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = cp.zeros((32, 2), dtype=cp.float32)
        with self.assertRaisesRegex(NotImplementedError, "requires.*points=1"):
            ibFFT_repulsive_sampling_GPU(
                embedding,
                p2m_mode="block_atomic",
                **_force_kwargs(2),
            )

    def test_p1_block_atomic_is_finite(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = np.random.default_rng(92).uniform(
            -2.5, 6.75, size=(512, 2)
        ).astype(np.float32)
        diagnostics = {}
        force = cp.asnumpy(
            ibFFT_repulsive_sampling_GPU(
                cp.asarray(embedding),
                p2m_mode="block_atomic",
                p2m_diagnostics=diagnostics,
                **_force_kwargs(1),
            )
        )
        self.assertTrue(np.isfinite(force).all())
        self.assertEqual(diagnostics["resolved_mode"], "block_atomic")

    def test_strict_auto_selects_segmented_or_budgeted_serial(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = np.random.default_rng(93).uniform(
            -2.5, 6.75, size=(256, 2)
        ).astype(np.float32)
        for p in (1, 2, 3):
            for limit, expected in ((None, "segmented"), (1, "serial")):
                with self.subTest(p=p, workspace_limit_bytes=limit):
                    diagnostics = {}
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        deterministic=True,
                        workspace_limit_bytes=limit,
                        p2m_diagnostics=diagnostics,
                        **_force_kwargs(p),
                    )
                    self.assertEqual(diagnostics["resolved_mode"], expected)

    def test_deterministic_sampling_force_is_exactly_repeatable(self):
        import cupy as cp
        from ibumap.kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU

        embedding = np.random.default_rng(17).uniform(
            -2.5, 6.75, size=(192, 2)
        ).astype(np.float32)
        probabilities = cp.asarray(
            np.linspace(0.2, 0.9, embedding.shape[0]), dtype=cp.float32
        )
        kwargs = _force_kwargs(3)
        kwargs["probabilities"] = probabilities

        outputs = []
        for _ in range(2):
            outputs.append(
                cp.asnumpy(
                    ibFFT_repulsive_sampling_GPU(
                        cp.asarray(embedding),
                        random_state=cp.random.RandomState(314159),
                        deterministic=True,
                        **kwargs,
                    )
                )
            )

        np.testing.assert_array_equal(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
