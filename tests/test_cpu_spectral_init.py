import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy import sparse
import scipy.sparse.linalg

from ibumap.init import cpu_init
from ibumap.init.cpu_init import initialize_embedding_cpu
from ibumap.init._spectral_cpu_impl import spectral_layout_cpu
from ibumap.init._spectral_defaults import (
    auto_ncv,
    resolve_component_ncv,
    resolve_spectral_defaults,
)


class CPUSpectralInitTest(unittest.TestCase):
    @staticmethod
    def _disconnected_cycles(component_sizes):
        rows = []
        cols = []
        offset = 0
        for component_size in component_sizes:
            local = np.arange(component_size, dtype=np.int64) + offset
            nxt = np.roll(local, -1)
            rows.extend(np.concatenate([local, nxt]).tolist())
            cols.extend(np.concatenate([nxt, local]).tolist())
            offset += component_size
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(offset, offset),
        )

    def test_component_ncv_policy_and_safe_clipping(self):
        self.assertEqual(auto_ncv(15, 3), 7)
        self.assertEqual(auto_ncv(50_000, 3), 64)
        self.assertEqual(
            resolve_component_ncv(component_n=15, k=3, user_ncv=None),
            7,
        )
        self.assertEqual(
            resolve_component_ncv(component_n=15, k=3, user_ncv=64),
            14,
        )
        with self.assertRaisesRegex(ValueError, "greater than k \\+ 1"):
            resolve_component_ncv(component_n=4, k=3, user_ncv=None)

    def test_large_spectral_defaults_are_backend_specific(self):
        cpu = resolve_spectral_defaults(
            n_samples=2_000_000,
            n_components=2,
        )
        cuda = resolve_spectral_defaults(
            n_samples=2_000_000,
            n_components=2,
            backend="cuda",
        )

        self.assertEqual(cpu.method, "lobpcg")
        self.assertEqual(cpu.policy, "umap_learn_large_lobpcg")
        self.assertIsNone(cpu.ncv)
        self.assertEqual(cuda.method, "eigsh")
        self.assertEqual(cuda.policy, "ibumap_cuda_large_eigsh")
        self.assertEqual(cuda.tol, 1e-4)
        self.assertEqual(cuda.maxiter, 10_000_000)
        self.assertEqual(cuda.ncv, 64)

        with self.assertRaisesRegex(ValueError, "backend"):
            resolve_spectral_defaults(
                n_samples=2_000_000,
                n_components=2,
                backend="tpu",
            )

    def test_multicomponent_defaults_are_resolved_per_component(self):
        component_sizes = (5, 15)
        graph = self._disconnected_cycles(component_sizes)
        data = np.random.RandomState(7).normal(size=(graph.shape[0], 4))
        calls = []

        def fake_eigsh(matrix, k, **kwargs):
            calls.append(
                {
                    "component_n": matrix.shape[0],
                    "ncv": kwargs["ncv"],
                    "maxiter": kwargs["maxiter"],
                    "tol": kwargs["tol"],
                }
            )
            eigenvalues = np.arange(k, dtype=np.float64)
            eigenvectors = np.arange(
                1,
                matrix.shape[0] * k + 1,
                dtype=np.float64,
            ).reshape(matrix.shape[0], k)
            return eigenvalues, eigenvectors

        with patch("scipy.sparse.linalg.eigsh", side_effect=fake_eigsh):
            result = spectral_layout_cpu(
                data,
                graph,
                2,
                np.random.RandomState(11),
                auto_defaults=True,
            )

        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(
            calls,
            [
                {"component_n": 5, "ncv": 4, "maxiter": 25, "tol": 1e-4},
                {"component_n": 15, "ncv": 7, "maxiter": 75, "tol": 1e-4},
            ],
        )

    def test_explicit_ncv_is_clipped_per_component(self):
        component_sizes = (5, 15)
        graph = self._disconnected_cycles(component_sizes)
        data = np.random.RandomState(13).normal(size=(graph.shape[0], 4))
        calls = []

        def fake_eigsh(matrix, k, **kwargs):
            calls.append((matrix.shape[0], kwargs["ncv"]))
            return (
                np.arange(k, dtype=np.float64),
                np.arange(
                    1,
                    matrix.shape[0] * k + 1,
                    dtype=np.float64,
                ).reshape(matrix.shape[0], k),
            )

        with patch("scipy.sparse.linalg.eigsh", side_effect=fake_eigsh):
            result = spectral_layout_cpu(
                data,
                graph,
                2,
                np.random.RandomState(17),
                ncv=64,
            )

        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(calls, [(5, 4), (15, 14)])

    def test_nonfinite_eigensolver_output_falls_back(self):
        graph = self._disconnected_cycles((8,))
        data = np.random.RandomState(19).normal(size=(graph.shape[0], 4))
        diagnostics = {}

        def nonfinite_eigsh(matrix, k, **kwargs):
            return (
                np.full(k, np.nan, dtype=np.float64),
                np.full((matrix.shape[0], k), np.nan, dtype=np.float64),
            )

        with patch(
            "scipy.sparse.linalg.eigsh",
            side_effect=nonfinite_eigsh,
        ), self.assertWarns(UserWarning):
            result = spectral_layout_cpu(
                data,
                graph,
                2,
                np.random.RandomState(23),
                diagnostics=diagnostics,
            )

        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(diagnostics["spectral_fallback"])
        self.assertEqual(diagnostics["spectral_exception_type"], "FloatingPointError")

    def test_invalid_component_scaling_falls_back(self):
        graph = self._disconnected_cycles((6, 7))
        data = np.random.RandomState(29).normal(size=(graph.shape[0], 4))
        diagnostics = {}

        def zero_eigsh(matrix, k, **kwargs):
            return (
                np.arange(k, dtype=np.float64),
                np.zeros((matrix.shape[0], k), dtype=np.float64),
            )

        with patch(
            "scipy.sparse.linalg.eigsh",
            side_effect=zero_eigsh,
        ), self.assertWarns(UserWarning):
            result = spectral_layout_cpu(
                data,
                graph,
                2,
                np.random.RandomState(31),
                diagnostics=diagnostics,
            )

        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(diagnostics["spectral_fallback"])

    def test_global_spectral_scaling_rejects_nonfinite_layout(self):
        graph = self._disconnected_cycles((8,))
        data = np.random.RandomState(37).normal(size=(graph.shape[0], 4))

        with patch(
            "ibumap.init.cpu_init.spectral_layout_cpu",
            return_value=np.full((graph.shape[0], 2), np.nan, dtype=np.float32),
        ), self.assertRaisesRegex(
            FloatingPointError,
            "before global scaling",
        ):
            initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(41),
                "euclidean",
                {},
            )

    def test_spectral_jitter_is_added_after_final_normalization(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        spectral_layout = np.asarray(
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        seed = 43
        diagnostics = {}
        timing = {}

        with (
            patch(
                "ibumap.init.cpu_init.spectral_layout_cpu",
                return_value=spectral_layout,
            ),
            patch(
                "ibumap.init.cpu_init._duplicate_row_stats",
                side_effect=AssertionError("duplicate scan must be disabled"),
            ),
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(seed),
                "euclidean",
                {},
                init_timing=timing,
                init_diagnostics=diagnostics,
            )

        normalized = np.asarray(
            [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
            dtype=np.float32,
        )
        expected = normalized + np.random.RandomState(seed).normal(
            scale=1e-4,
            size=normalized.shape,
        ).astype(np.float32)
        np.testing.assert_array_equal(embedding, expected)
        self.assertEqual(np.unique(embedding, axis=0).shape[0], embedding.shape[0])
        self.assertEqual(diagnostics["init_final_jitter_scale"], 1e-4)
        self.assertNotIn("init_duplicate_ratio", diagnostics)
        self.assertIn("init_jitter_time", timing)

    def test_fft_compact_policy_preserves_small_spectral_range(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        spectral_layout = np.asarray(
            [[-0.02415, -0.01], [0.0, 0.005], [0.02415, 0.01]],
            dtype=np.float32,
        )
        diagnostics = {}

        with patch(
            "ibumap.init.cpu_init.spectral_layout_cpu",
            return_value=spectral_layout,
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(44),
                "euclidean",
                {},
                spectral_scale_policy="fft_compact",
                spectral_jitter_relative=0.0,
                init_diagnostics=diagnostics,
            )

        raw_span = float(spectral_layout.max() - spectral_layout.min())
        self.assertAlmostEqual(
            float(embedding.max() - embedding.min()),
            raw_span,
            places=7,
        )
        self.assertEqual(diagnostics["init_spectral_scale_policy"], "fft_compact")
        self.assertEqual(diagnostics["init_spectral_scale_factor"], 1.0)
        self.assertAlmostEqual(
            diagnostics["init_spectral_final_global_span"],
            raw_span,
            places=7,
        )
        self.assertEqual(diagnostics["init_final_jitter_scale"], 0.0)

    def test_fft_compact_policy_caps_large_spectral_range(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        spectral_layout = np.asarray(
            [[-10.0, -2.0], [0.0, 0.0], [10.0, 2.0]],
            dtype=np.float32,
        )
        diagnostics = {}

        with patch(
            "ibumap.init.cpu_init.spectral_layout_cpu",
            return_value=spectral_layout,
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(45),
                "euclidean",
                {},
                spectral_scale_policy="fft_compact",
                spectral_max_span=0.1,
                spectral_jitter_relative=0.0,
                init_diagnostics=diagnostics,
            )

        self.assertAlmostEqual(
            float(embedding.max() - embedding.min()),
            0.1,
            places=7,
        )
        self.assertEqual(diagnostics["init_spectral_scale_factor"], 0.005)
        self.assertAlmostEqual(
            diagnostics["init_spectral_pre_jitter_global_span"],
            0.1,
            places=7,
        )

    def test_raw_policy_reproduces_unprocessed_float32_layout(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        spectral_layout = np.asarray(
            [[-0.02, 0.01], [0.0, 0.01], [0.02, -0.01]],
            dtype=np.float64,
        )
        diagnostics = {}

        with patch(
            "ibumap.init.cpu_init.spectral_layout_cpu",
            return_value=spectral_layout,
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(46),
                "euclidean",
                {},
                spectral_scale_policy="raw",
                init_diagnostics=diagnostics,
            )

        np.testing.assert_array_equal(embedding, spectral_layout.astype(np.float32))
        self.assertEqual(diagnostics["init_spectral_scale_policy"], "raw")
        self.assertEqual(diagnostics["init_final_jitter_scale"], 0.0)

    def test_spectral_axis_orientation_canonicalizes_sign_flips(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        spectral_layout = np.asarray(
            [[0.0, 0.0], [0.1, 0.2], [10.0, 10.0]],
            dtype=np.float32,
        )
        embeddings = []
        diagnostics = []

        for layout in (spectral_layout, -spectral_layout):
            run_diagnostics = {}
            with patch(
                "ibumap.init.cpu_init.spectral_layout_cpu",
                return_value=layout,
            ):
                embeddings.append(
                    initialize_embedding_cpu(
                        data,
                        graph,
                        2,
                        "spectral",
                        np.random.RandomState(45),
                        "euclidean",
                        {},
                        init_diagnostics=run_diagnostics,
                    )
                )
            diagnostics.append(run_diagnostics)

        np.testing.assert_allclose(
            embeddings[0],
            embeddings[1],
            rtol=0.0,
            atol=4 * np.spacing(np.float32(10.0)),
        )
        self.assertEqual(
            diagnostics[0]["init_axis_orientation_reflected"],
            [False, False],
        )
        self.assertEqual(
            diagnostics[1]["init_axis_orientation_reflected"],
            [True, True],
        )
        for run_diagnostics in diagnostics:
            means = run_diagnostics["init_axis_orientation_means_after"]
            midpoints = run_diagnostics["init_axis_orientation_midpoints"]
            self.assertTrue(
                all(mean <= midpoint for mean, midpoint in zip(means, midpoints))
            )

    def test_duplicate_rows_are_reported_without_being_required_to_be_unique(self):
        graph = self._disconnected_cycles((3,))
        data = np.zeros((3, 2), dtype=np.float32)
        diagnostics = {}
        timing = {}

        # Keep the deliberately collapsed spectral layout collapsed to exercise
        # the final duplicate-rate warning policy.
        with (
            self.assertWarnsRegex(RuntimeWarning, "66.67%"),
            patch(
                "ibumap.init.cpu_init.spectral_layout_cpu",
                return_value=np.ones((3, 2), dtype=np.float32),
            ),
            patch.object(cpu_init, "_FINAL_JITTER_SCALE", 0.0),
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(47),
                "euclidean",
                {},
                report_duplicate_ratio=True,
                init_timing=timing,
                init_diagnostics=diagnostics,
            )

        self.assertEqual(embedding.shape, (3, 2))
        self.assertEqual(np.unique(embedding, axis=0).shape[0], 1)
        self.assertEqual(diagnostics["init_post_jitter_duplicate_rows"], 2)
        self.assertEqual(diagnostics["init_unique_rows"], 1)
        self.assertEqual(diagnostics["init_duplicate_rows"], 2)
        self.assertEqual(diagnostics["init_duplicate_ratio"], 2 / 3)
        self.assertEqual(diagnostics["init_max_duplicate_group"], 3)
        self.assertIn("init_duplicate_ratio_time", timing)

    def test_low_duplicate_ratio_is_recorded_without_warning(self):
        for ratio in (0.005, 0.01):
            with self.subTest(ratio=ratio):
                diagnostics = {}
                stats = {
                    "unique_rows": 199,
                    "duplicate_rows": 1,
                    "duplicate_ratio": ratio,
                    "max_duplicate_group": 2,
                }
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cpu_init._record_duplicate_diagnostics(diagnostics, stats)

                self.assertEqual(caught, [])
                self.assertEqual(diagnostics["init_duplicate_rows"], 1)
                self.assertEqual(diagnostics["init_max_duplicate_group"], 2)

    def test_high_duplicate_ratio_warns_without_raising(self):
        diagnostics = {}
        stats = {
            "unique_rows": 98,
            "duplicate_rows": 2,
            "duplicate_ratio": 0.02,
            "max_duplicate_group": 3,
        }

        with self.assertWarnsRegex(RuntimeWarning, "2.00%"):
            cpu_init._record_duplicate_diagnostics(diagnostics, stats)

        self.assertEqual(diagnostics["init_duplicate_rows"], 2)
        self.assertEqual(diagnostics["init_duplicate_ratio"], 0.02)

    def test_cpu_spectral_init_uses_local_impl_and_records_parameters(self):
        rng = np.random.RandomState(123)
        n_samples = 32
        rows = np.arange(n_samples, dtype=np.int64)
        cols = (rows + 1) % n_samples
        graph = sparse.csr_matrix(
            (
                np.ones(n_samples * 2, dtype=np.float32),
                (
                    np.concatenate([rows, cols]),
                    np.concatenate([cols, rows]),
                ),
            ),
            shape=(n_samples, n_samples),
        )
        data = rng.normal(size=(n_samples, 5)).astype(np.float32)
        timing = {}
        diagnostics = {}

        with patch(
            "umap.spectral.spectral_layout",
            side_effect=AssertionError("umap-learn spectral_layout should not run"),
        ):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                rng,
                "euclidean",
                {},
                spectral_method="eigsh",
                spectral_tol=1e-5,
                spectral_maxiter=1234,
                spectral_ncv=17,
                init_timing=timing,
                init_diagnostics=diagnostics,
            )

        self.assertEqual(embedding.shape, (n_samples, 2))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())
        self.assertEqual(diagnostics["init_method"], "spectral")
        self.assertEqual(diagnostics["spectral_result_device"], "cpu")
        self.assertEqual(diagnostics["spectral_method"], "eigsh")
        self.assertEqual(diagnostics["spectral_tol"], 1e-5)
        self.assertEqual(diagnostics["spectral_maxiter"], 1234)
        self.assertEqual(diagnostics["spectral_ncv"], 17)
        self.assertFalse(diagnostics["spectral_fallback"])
        for key in (
            "init_connected_components_time",
            "init_laplacian_time",
            "init_solver_setup_time",
            "init_eigensolver_time",
            "init_postprocess_time",
            "init_jitter_time",
            "init_normalize_time",
        ):
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)

    def test_cpu_spectral_multicomponent_fallback_is_not_reset(self):
        rng = np.random.RandomState(123)
        component_size = 8
        rows = []
        cols = []
        for offset in (0, component_size):
            local = np.arange(component_size, dtype=np.int64) + offset
            nxt = np.roll(local, -1)
            rows.extend(np.concatenate([local, nxt]).tolist())
            cols.extend(np.concatenate([nxt, local]).tolist())
        n_samples = component_size * 2
        graph = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_samples, n_samples),
        )
        data = rng.normal(size=(n_samples, 4)).astype(np.float32)
        timing = {}
        diagnostics = {}
        original_eigsh = scipy.sparse.linalg.eigsh
        calls = {"count": 0}

        def flaky_eigsh(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("forced component failure")
            return original_eigsh(*args, **kwargs)

        with patch("scipy.sparse.linalg.eigsh", side_effect=flaky_eigsh), self.assertWarns(UserWarning):
            embedding = initialize_embedding_cpu(
                data,
                graph,
                2,
                "spectral",
                rng,
                "euclidean",
                {},
                spectral_method="eigsh",
                spectral_tol=1e-4,
                spectral_maxiter=1000,
                spectral_ncv=6,
                init_timing=timing,
                init_diagnostics=diagnostics,
            )

        self.assertEqual(embedding.shape, (n_samples, 2))
        self.assertTrue(np.isfinite(embedding).all())
        self.assertGreaterEqual(calls["count"], 2)
        self.assertTrue(diagnostics["spectral_fallback"])
        self.assertEqual(diagnostics["spectral_exception_type"], "RuntimeError")
        self.assertIn("forced component failure", diagnostics["spectral_exception_message"])
        # The forced failure is recorded first; some SciPy/ARPACK versions also fail
        # on the second 8-vertex component, which is recorded after it.
        self.assertEqual(diagnostics["spectral_exceptions"][0]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
