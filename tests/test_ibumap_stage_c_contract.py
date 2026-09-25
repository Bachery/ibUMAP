import unittest

import numpy as np

from ibumap import FFTConfig, IBUMAP
from ibumap.kernels.cpu import ibfft
from ibumap.optimizers.ibumap_optimizer import (
    UMAP_ApplyForce,
    _clip_force_norm_inplace,
    _local_exact_repulsion_force,
    _local_exact_repulsion_force_profiled,
    tFDP_ReplForce_ibFFT_neg_effect,
)


def _embedding(n=96):
    rng = np.random.default_rng(20260623)
    # Keep the span deliberately non-unit so p=2/3 interpolation scale bugs are
    # visible in the same contract that protects the p=1 fast path.
    return rng.uniform(-2.5, 7.75, size=(n, 2)).astype(np.float32)


def _ibfft_kwargs(n_interpolation_points=1, probabilities=None):
    return {
        "n_interpolation_points": n_interpolation_points,
        "intervals_per_integer": 1.0,
        "min_num_intervals": 32,
        "gamma": 2.0,
        "paraFactor": 2.0,
        "probabilities": probabilities,
        "n_boxes_per_dim": 1.0,
        "kernel_method": "UMAP_kernel",
        "umap_a": 1.57,
        "umap_b": 0.895,
        "umap_gamma": 1.0,
        "umap_epsilon": 1e-3,
        "ibfft_kernel_clip": 4.0,
    }


class IBUMAPStageCContractTest(unittest.TestCase):
    def setUp(self):
        ibfft._clear_fft_kernel_cache()

    def tearDown(self):
        ibfft._clear_fft_kernel_cache()

    def test_force_path_is_finite_for_interpolation_orders_one_two_three(self):
        embedding = _embedding()

        for p in (1, 2, 3):
            with self.subTest(n_interpolation_points=p):
                force = ibfft.ibFFT_repulsive_sampling(
                    embedding,
                    **_ibfft_kwargs(n_interpolation_points=p),
                )

                self.assertEqual(force.shape, embedding.shape)
                self.assertEqual(force.dtype, np.float32)
                self.assertTrue(np.isfinite(force).all())

    def test_p1_workspace_reuse_preserves_force_values(self):
        embedding = _embedding()
        workspace = {}
        kwargs = _ibfft_kwargs(n_interpolation_points=1)

        first = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="fast",
            **kwargs,
        ).copy()
        buffers = dict(workspace)
        second = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="fast",
            **kwargs,
        ).copy()

        np.testing.assert_array_equal(second, first)
        for key, value in buffers.items():
            self.assertIs(workspace[key], value, key)

    def test_p1_reuses_explicit_fftw_plan_and_aligned_buffers(self):
        embedding = _embedding()
        workspace = {}
        first_diagnostics = {}
        second_diagnostics = {}
        kwargs = _ibfft_kwargs(n_interpolation_points=1)

        first = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="fast",
            p2m_diagnostics=first_diagnostics,
            **kwargs,
        ).copy()
        plan_keys = [
            key for key in workspace if key.startswith(ibfft._FFTW_PLAN_KEY_PREFIX)
        ]
        self.assertEqual(len(plan_keys), 1)
        plan_key = plan_keys[0]
        plan_bundle = workspace[plan_key]
        mat_w = workspace["mat_w"]

        self.assertIs(plan_bundle["input"], mat_w)
        self.assertEqual(mat_w.ctypes.data % ibfft._FFTW_ALIGNMENT, 0)
        self.assertEqual(
            plan_bundle["frequency"].ctypes.data % ibfft._FFTW_ALIGNMENT,
            0,
        )
        self.assertEqual(
            plan_bundle["output"].ctypes.data % ibfft._FFTW_ALIGNMENT,
            0,
        )
        self.assertFalse(first_diagnostics["fft_plan_cache_hit"])

        second = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="fast",
            p2m_diagnostics=second_diagnostics,
            **kwargs,
        ).copy()

        np.testing.assert_array_equal(second, first)
        self.assertIs(workspace[plan_key], plan_bundle)
        self.assertIs(workspace["mat_w"], mat_w)
        self.assertTrue(second_diagnostics["fft_plan_cache_hit"])

    def test_p1_fast_path_skips_interpolation_workspace_buffers(self):
        embedding = _embedding()
        workspace = {}

        ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="fast",
            **_ibfft_kwargs(n_interpolation_points=1),
        )

        self.assertNotIn("y_in_box", workspace)
        self.assertNotIn("interpolate_values", workspace)
        self.assertNotIn("ChargesQij", workspace)
        self.assertNotIn("w_coefficients", workspace)
        self.assertNotIn("potentialsQij", workspace)
        self.assertIn("box_idx", workspace)
        self.assertIn("mat_w", workspace)
        self.assertIn("neg_f", workspace)

    def test_fused_m2p_update_matches_force_postprocess_and_apply(self):
        base_embedding = _embedding(128)
        rng = np.random.default_rng(20260706)
        attr_force = rng.normal(0.0, 0.05, size=base_embedding.shape).astype(
            np.float32
        )
        neg_effects = rng.uniform(0.1, 1.5, size=base_embedding.shape[0]).astype(
            np.float32
        )
        alpha = 0.7
        clip_norm = 0.45

        for p in (1, 2, 3):
            with self.subTest(n_interpolation_points=p):
                ibfft._clear_fft_kernel_cache()
                expected_embedding = base_embedding.copy()
                raw_force = ibfft.ibFFT_repulsive_sampling(
                    expected_embedding,
                    **_ibfft_kwargs(n_interpolation_points=p),
                ).copy()
                repl_force = tFDP_ReplForce_ibFFT_neg_effect(
                    raw_force, alpha, neg_effects
                )
                _clip_force_norm_inplace(
                    repl_force, clip_norm, is_gpu=False, return_norms=False
                )
                UMAP_ApplyForce(
                    expected_embedding,
                    attr_force,
                    repl_force,
                    expected_embedding.shape[0],
                    2,
                )

                fused_embedding = base_embedding.copy()
                fused_workspace = {}
                diagnostics = {}
                result = ibfft.ibFFT_repulsive_sampling(
                    fused_embedding,
                    _workspace=fused_workspace,
                    p2m_diagnostics=diagnostics,
                    fused_update={
                        "embedding": fused_embedding,
                        "attr_force": attr_force,
                        "alpha": alpha,
                        "neg_effects": neg_effects,
                        "clip_norm": clip_norm,
                    },
                    **_ibfft_kwargs(n_interpolation_points=p),
                )

                self.assertIsNone(result)
                self.assertTrue(diagnostics["fused_m2p_update"])
                self.assertNotIn("neg_f", fused_workspace)
                self.assertNotIn("potentialsQij", fused_workspace)
                np.testing.assert_allclose(
                    fused_embedding,
                    expected_embedding,
                    rtol=2e-6,
                    atol=2e-6,
                )

    def test_p1_sampling_workspace_reuse_preserves_force_values(self):
        embedding = _embedding()
        probabilities = np.full(embedding.shape[0], 0.35, dtype=np.float64)
        workspace = {}
        kwargs = _ibfft_kwargs(n_interpolation_points=1, probabilities=probabilities)

        first = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="low_memory",
            random_state=np.random.RandomState(17),
            **kwargs,
        ).copy()
        second = ibfft.ibFFT_repulsive_sampling(
            embedding,
            _workspace=workspace,
            workspace_policy="low_memory",
            random_state=np.random.RandomState(17),
            **kwargs,
        ).copy()

        np.testing.assert_array_equal(second, first)
        self.assertEqual(first.shape, embedding.shape)
        self.assertGreater(np.count_nonzero(first), 0)

    def test_local_exact_force_preserves_embedding_dtype(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                embedding = np.array(
                    [[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]],
                    dtype=dtype,
                )
                args = (
                    embedding,
                    np.array([0, 0, 1], dtype=np.int64),
                    np.array([0, 0, 0], dtype=np.int64),
                    np.array([0, 1, 2], dtype=np.int64),
                    np.array([0, 2], dtype=np.int64),
                    np.array([2, 3], dtype=np.int64),
                    2,
                    1,
                    2,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.001,
                    0.5,
                    4.0,
                    True,
                    np.array([2, 1], dtype=np.int64),
                    np.array([0, 0, 1], dtype=np.int64),
                    False,
                    0.0,
                    False,
                )

                force = _local_exact_repulsion_force(*args)
                profiled_force, *_ = _local_exact_repulsion_force_profiled(*args, 4)

                self.assertEqual(force.dtype, np.dtype(dtype))
                self.assertEqual(profiled_force.dtype, np.dtype(dtype))
                self.assertTrue(np.isfinite(force).all())
                self.assertTrue(np.isfinite(profiled_force).all())

    def test_cpu_ibumap_p1_short_optimizer_repeats_with_same_seed(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(64, 6)).astype(np.float32)

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cpu",
                n_neighbors=8,
                n_epochs=3,
                random_state=99,
                deterministic=True,
                attraction_mode="sampling",
                repulsion_mode="sampling",
                fft=FFTConfig(n_interpolation_points=1),
            ).fit_transform(X)

        first = run()
        second = run()

        np.testing.assert_array_equal(second, first)


if __name__ == "__main__":
    unittest.main()
