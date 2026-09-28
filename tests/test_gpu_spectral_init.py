import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import UMAPConfig, IBUMAP
from ibumap.graph import build_cpu_graph_from_knn
from ibumap.init._custom_init import custom_init_jitter_scale


N_SAMPLES = 128
N_FEATURES = 10
N_NEIGHBORS = 12
N_EPOCHS = 5


def _gpu_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _fixed_inputs():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="euclidean")
    nn.fit(X)
    knn_distances, knn_indices = nn.kneighbors(X)
    graph = build_cpu_graph_from_knn(
        X,
        n_neighbors=N_NEIGHBORS,
        metric="euclidean",
        metric_kwds={},
        random_state=123,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        angular_rp_forest=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        densmap_or_output_dens=False,
        verbose=False,
    )
    return X, knn_indices.astype(np.int64), knn_distances.astype(np.float32), graph


def _model(init):
    return IBUMAP(
        algorithm="ibumap",
        device="cuda",
        n_neighbors=N_NEIGHBORS,
        random_state=42,
        deterministic=True,
        umap=UMAPConfig(n_epochs=N_EPOCHS, init=init),
    )


class GPUSpectralInitTest(unittest.TestCase):
    def assert_embedding_ok(self, embedding):
        self.assertEqual(embedding.shape, (N_SAMPLES, 2))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())

    def test_custom_init_jitter_scale_matches_umap_learn_rule(self):
        from sklearn.neighbors import KDTree

        unique_init = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        self.assertIsNone(custom_init_jitter_scale(np, unique_init))

        duplicate_init = np.array(
            [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]],
            dtype=np.float32,
        )
        distances, _ = KDTree(duplicate_init).query(duplicate_init, k=2)
        expected = 0.001 * float(np.mean(distances[:, 1]))
        actual = custom_init_jitter_scale(np, duplicate_init)
        self.assertAlmostEqual(float(actual), expected, places=9)

    def test_deterministic_optimizer_degree_uses_fixed_row_sum(self):
        from types import SimpleNamespace

        from ibumap.pipeline.gpu_pipeline import GPUPipeline

        pipeline = GPUPipeline.__new__(GPUPipeline)
        pipeline.model = SimpleNamespace(
            runtime=SimpleNamespace(algorithm="ibumap", deterministic=True),
            _time_costs={},
        )
        graph = object()
        expected = object()

        with patch(
            "ibumap.pipeline.gpu_pipeline.deterministic_csr_row_sum_cupy",
            return_value=expected,
        ) as row_sum:
            actual = pipeline._optimizer_degrees(graph)

        self.assertIs(actual, expected)
        row_sum.assert_called_once_with(graph)
        self.assertEqual(
            pipeline.model._time_costs["optimizer_degree_deterministic"],
            1,
        )

    def test_fast_optimizer_degree_keeps_native_cupy_sum(self):
        from types import SimpleNamespace

        from ibumap.pipeline.gpu_pipeline import GPUPipeline

        class FakeGraph:
            def __init__(self):
                self.sum_axes = []

            def sum(self, *, axis):
                self.sum_axes.append(axis)
                return np.asarray([[1.25], [2.5]], dtype=np.float64)

        pipeline = GPUPipeline.__new__(GPUPipeline)
        pipeline.model = SimpleNamespace(
            runtime=SimpleNamespace(algorithm="ibumap", deterministic=False),
            _time_costs={},
        )
        graph = FakeGraph()
        fake_cupy = SimpleNamespace(float32=np.float32)

        with (
            patch.dict(sys.modules, {"cupy": fake_cupy}),
            patch(
                "ibumap.pipeline.gpu_pipeline.deterministic_csr_row_sum_cupy"
            ) as row_sum,
        ):
            actual = pipeline._optimizer_degrees(graph)

        np.testing.assert_array_equal(
            actual,
            np.asarray([1.25, 2.5], dtype=np.float32),
        )
        self.assertEqual(graph.sum_axes, [1])
        row_sum.assert_not_called()
        self.assertEqual(
            pipeline.model._time_costs["optimizer_degree_deterministic"],
            0,
        )

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_shared_deterministic_row_sum_is_bitwise_repeatable(self):
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse

        from ibumap.kernels.gpu.cupy_sparse import (
            deterministic_csr_row_sum_cupy,
        )

        graph_cpu = sparse.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.1, 0.2, 0.3],
                    [0.4, 0.0, 0.5, 0.6],
                    [0.7, 0.8, 0.0, 0.9],
                    [1.0, 1.1, 1.2, 0.0],
                ],
                dtype=np.float32,
            )
        )
        graph = cupyx_sparse.csr_matrix(graph_cpu)
        results = [
            cp.asnumpy(deterministic_csr_row_sum_cupy(graph))
            for _ in range(3)
        ]

        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])
        np.testing.assert_allclose(
            results[0],
            np.asarray(graph_cpu.sum(axis=1)).reshape(-1),
            rtol=1e-6,
            atol=0.0,
        )

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_component_labels_are_canonical_across_native_permutations(self):
        import cupy as cp

        from ibumap.kernels.gpu.cupy_sparse import (
            canonicalize_component_labels_cupy,
        )

        first_native = cp.asarray([2, 2, 0, 0, 1, 1], dtype=cp.int32)
        second_native = cp.asarray([1, 1, 2, 2, 0, 0], dtype=cp.int32)
        expected = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)

        first = cp.asnumpy(canonicalize_component_labels_cupy(first_native, 3))
        second = cp.asnumpy(
            canonicalize_component_labels_cupy(second_native, 3)
        )

        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_fit_transform_from_knn_custom_duplicate_init_is_seeded_and_jittered(self):
        X, knn_indices, knn_distances, _ = _fixed_inputs()
        init = np.random.RandomState(17).normal(size=(N_SAMPLES, 2)).astype(
            np.float32
        )
        init[1] = init[0]

        first = _model(init.copy())
        second = _model(init.copy())
        first_embedding = first.fit_transform_from_knn(
            X, knn_indices, knn_distances
        )
        second_embedding = second.fit_transform_from_knn(
            X, knn_indices, knn_distances
        )

        self.assert_embedding_ok(first_embedding)
        self.assert_embedding_ok(second_embedding)
        np.testing.assert_array_equal(
            first.get_init_embedding(),
            second.get_init_embedding(),
        )
        self.assertEqual(
            np.unique(first.get_init_embedding(), axis=0).shape[0],
            N_SAMPLES,
        )
        costs = first.get_time_costs()
        self.assertIn("init_custom_prepare_time", costs)
        self.assertIn("init_custom_to_gpu_time", costs)
        self.assertIn("init_custom_jitter_time", costs)
        self.assertIn("init_normalize_time", costs)
        diagnostics = first.get_init_diagnostics()
        self.assertEqual(diagnostics["init_method"], "custom")
        self.assertTrue(diagnostics["init_custom_duplicates_detected"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_spectral_layout_cupy_returns_cupy_array(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        X, _, _, graph = _fixed_inputs()
        diagnostics = {}
        timing = {}

        embedding = spectral_layout_cupy(
            X,
            graph,
            2,
            random_state=42,
            method="eigsh",
            tol=1e-3,
            maxiter=5000,
            ncv=32,
            timing=timing,
            diagnostics=diagnostics,
        )

        self.assertIsInstance(embedding, cp.ndarray)
        self.assertEqual(embedding.shape, (N_SAMPLES, 2))
        self.assertTrue(bool(cp.isfinite(embedding).all().item()))
        self.assertEqual(diagnostics["spectral_ncv"], 32)
        self.assertEqual(diagnostics["spectral_tol"], 1e-3)
        self.assertEqual(diagnostics["spectral_maxiter"], 5000)
        self.assertEqual(diagnostics["spectral_initial_guess"], "none")
        self.assertFalse(diagnostics["spectral_initial_guess_generated"])
        self.assertEqual(
            diagnostics["spectral_degree_reduction"],
            "cupy_csr_sum_axis0",
        )
        self.assertEqual(
            diagnostics["spectral_laplacian_construction"],
            "cupy_sparse_diagonal_product",
        )
        self.assertIn("init_eigensolver_time", timing)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_deterministic_laplacian_builder_is_bitwise_repeatable(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import (
            _build_normalized_laplacian_cupy,
        )

        graph = sparse.diags(
            [np.ones(127, dtype=np.float32), np.ones(127, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(128, 128),
            format="csr",
        )
        references = None
        diagnostics = {}
        for _ in range(3):
            laplacian, sqrt_degree = _build_normalized_laplacian_cupy(
                graph,
                deterministic=True,
                diagnostics=diagnostics,
            )
            current = (
                cp.asnumpy(laplacian.indptr),
                cp.asnumpy(laplacian.indices),
                cp.asnumpy(laplacian.data),
                cp.asnumpy(sqrt_degree),
            )
            if references is None:
                references = current
            else:
                for actual, expected in zip(current, references):
                    np.testing.assert_array_equal(actual, expected)

        expected_degree = np.asarray(graph.sum(axis=0)).reshape(-1)
        expected_sqrt_degree = np.sqrt(expected_degree).astype(np.float32)
        np.testing.assert_allclose(
            references[3],
            expected_sqrt_degree,
            rtol=1e-6,
            atol=0.0,
        )
        expected_scale = sparse.diags(
            1.0 / expected_sqrt_degree,
            format="csr",
        )
        expected_laplacian = (
            sparse.eye(128, dtype=np.float32, format="csr")
            - expected_scale @ graph @ expected_scale
        )
        actual_laplacian = sparse.csr_matrix(
            (references[2], references[1], references[0]),
            shape=graph.shape,
        )
        np.testing.assert_allclose(
            actual_laplacian.toarray(),
            expected_laplacian.toarray(),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(
            diagnostics["spectral_degree_reduction"],
            "deterministic_csr_row_sum",
        )
        self.assertEqual(
            diagnostics["spectral_laplacian_construction"],
            "deterministic_csr_direct_scaling",
        )
        self.assertTrue(diagnostics["spectral_symmetric_graph_assumed"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_eigsh_does_not_consume_rng_for_unused_initial_guess(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        graph = sparse.diags(
            [np.ones(127, dtype=np.float32), np.ones(127, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(128, 128),
            format="csr",
        )
        random_state = cp.random.RandomState(42)
        diagnostics = {}

        spectral_layout_cupy(
            None,
            graph,
            2,
            random_state=random_state,
            method="eigsh",
            tol=1e-3,
            maxiter=5000,
            ncv=32,
            deterministic=True,
            diagnostics=diagnostics,
        )
        actual_next_draw = random_state.normal(size=32)
        expected_next_draw = cp.random.RandomState(42).normal(size=32)

        cp.testing.assert_array_equal(actual_next_draw, expected_next_draw)
        self.assertEqual(diagnostics["spectral_initial_guess"], "none")
        self.assertFalse(diagnostics["spectral_initial_guess_generated"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_eigsh_dispatch_selects_alg2_or_alg1_and_reuses_canonical_csr(self):
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse

        from ibumap.init import _spectral_cupy_impl as spectral_module

        matrix = cupyx_sparse.eye(8, dtype=cp.float32, format="csr")
        expected = (
            cp.arange(3, dtype=cp.float32),
            cp.eye(8, 3, dtype=cp.float32),
        )

        deterministic_diagnostics = {}
        with (
            patch.object(
                spectral_module,
                "deterministic_csr_spmv_algorithm",
                return_value=(3, "CUSPARSE_CSRMV_ALG2"),
            ),
            patch.object(
                spectral_module,
                "eigsh_csr_alg2",
                return_value=expected,
            ) as alg2,
        ):
            actual = spectral_module._solve_eigsh_cupy(
                matrix,
                k=3,
                ncv=7,
                tol=1e-4,
                maxiter=40,
                deterministic=True,
                matrix_is_canonical=True,
                diagnostics=deterministic_diagnostics,
            )

        self.assertIs(actual, expected)
        self.assertIs(alg2.call_args.args[0], matrix)
        self.assertEqual(
            deterministic_diagnostics["spectral_csr_preparation"],
            "upstream_canonical_reused",
        )
        self.assertFalse(deterministic_diagnostics["spectral_csr_solver_copy"])
        self.assertEqual(
            deterministic_diagnostics["spectral_spmv_algorithm"],
            "CUSPARSE_CSRMV_ALG2",
        )
        self.assertTrue(
            deterministic_diagnostics["spectral_spmv_bitwise_deterministic"]
        )

        fast_diagnostics = {}
        with (
            patch.object(
                spectral_module,
                "fast_csr_spmv_algorithm",
                return_value=(2, "CUSPARSE_CSRMV_ALG1"),
            ),
            patch.object(
                spectral_module,
                "eigsh_csr_alg1",
                return_value=expected,
            ) as alg1,
        ):
            actual = spectral_module._solve_eigsh_cupy(
                matrix,
                k=3,
                ncv=7,
                tol=1e-4,
                maxiter=40,
                deterministic=False,
                diagnostics=fast_diagnostics,
            )

        self.assertIs(actual, expected)
        self.assertIs(alg1.call_args.args[0], matrix)
        self.assertEqual(
            fast_diagnostics["spectral_spmv_algorithm"],
            "CUSPARSE_CSRMV_ALG1",
        )
        self.assertFalse(fast_diagnostics["spectral_spmv_bitwise_deterministic"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_lobpcg_generates_seeded_initial_guess_in_its_branch(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        graph = sparse.diags(
            [np.ones(127, dtype=np.float32), np.ones(127, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(128, 128),
            format="csr",
        )
        captured_guesses = []

        def fake_lobpcg(matrix, initial_guess, **kwargs):
            captured_guesses.append(cp.asnumpy(initial_guess))
            k = initial_guess.shape[1]
            eigenvalues = cp.arange(k, dtype=cp.float32)
            eigenvectors = cp.zeros((matrix.shape[0], k), dtype=cp.float32)
            eigenvectors[:k] = cp.eye(k, dtype=cp.float32)
            return eigenvalues, eigenvectors

        diagnostics = {}
        random_state = cp.random.RandomState(42)
        with patch(
            "cupyx.scipy.sparse.linalg.lobpcg",
            side_effect=fake_lobpcg,
        ):
            spectral_layout_cupy(
                None,
                graph,
                2,
                random_state=random_state,
                method="lobpcg",
                tol=1e-3,
                maxiter=5000,
                ncv=32,
                diagnostics=diagnostics,
            )

        actual_next_draw = random_state.normal(size=32)
        fresh_next_draw = cp.random.RandomState(42).normal(size=32)
        self.assertFalse(bool(cp.array_equal(actual_next_draw, fresh_next_draw)))
        self.assertEqual(len(captured_guesses), 1)
        self.assertEqual(captured_guesses[0].shape, (128, 3))
        self.assertEqual(diagnostics["spectral_initial_guess"], "random")
        self.assertTrue(diagnostics["spectral_initial_guess_generated"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_connected_spectral_layout_does_not_materialize_features(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        graph = sparse.diags(
            [np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(8, 8),
            format="csr",
        )
        diagnostics = {}

        embedding = spectral_layout_cupy(
            object(),
            graph,
            2,
            random_state=42,
            method="eigsh",
            tol=1e-3,
            maxiter=5000,
            ncv=7,
            diagnostics=diagnostics,
        )

        self.assertIsInstance(embedding, cp.ndarray)
        self.assertEqual(embedding.shape, (8, 2))
        self.assertEqual(diagnostics["spectral_connected_components"], 1)
        self.assertFalse(diagnostics["spectral_feature_data_required"])
        self.assertFalse(diagnostics["spectral_feature_data_materialized"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_small_component_count_does_not_materialize_features(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        component = sparse.diags(
            [np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(3, 3),
            format="csr",
        )
        graph = sparse.block_diag([component, component], format="csr")
        diagnostics = {}

        embedding = spectral_layout_cupy(
            object(),
            graph,
            2,
            random_state=42,
            diagnostics=diagnostics,
            deterministic=True,
        )

        self.assertIsInstance(embedding, cp.ndarray)
        self.assertEqual(embedding.shape, (6, 2))
        self.assertEqual(diagnostics["spectral_connected_components"], 2)
        self.assertTrue(
            diagnostics["spectral_component_labels_canonicalized"]
        )
        self.assertEqual(
            diagnostics["spectral_component_label_order"],
            "minimum_vertex_index",
        )
        self.assertFalse(diagnostics["spectral_feature_data_required"])
        self.assertFalse(diagnostics["spectral_feature_data_materialized"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_many_components_materialize_features_lazily(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        component = sparse.diags(
            [np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(3, 3),
            format="csr",
        )
        graph = sparse.block_diag([component] * 5, format="csr")
        X = np.arange(15 * 4, dtype=np.float32).reshape(15, 4)
        diagnostics = {}
        timing = {}

        embedding = spectral_layout_cupy(
            X,
            graph,
            2,
            random_state=42,
            timing=timing,
            diagnostics=diagnostics,
        )

        self.assertIsInstance(embedding, cp.ndarray)
        self.assertEqual(embedding.shape, (15, 2))
        self.assertEqual(diagnostics["spectral_connected_components"], 5)
        self.assertTrue(diagnostics["spectral_feature_data_required"])
        self.assertTrue(diagnostics["spectral_feature_data_materialized"])
        self.assertIn("init_feature_transfer_time", timing)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_large_components_preserve_top_level_feature_diagnostics(self):
        import cupy as cp

        from ibumap.init._spectral_cupy_impl import spectral_layout_cupy

        component = sparse.diags(
            [np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32)],
            offsets=[-1, 1],
            shape=(6, 6),
            format="csr",
        )
        graph = sparse.block_diag([component] * 5, format="csr")
        X = np.arange(30 * 4, dtype=np.float32).reshape(30, 4)
        diagnostics = {}

        embedding = spectral_layout_cupy(
            X,
            graph,
            2,
            random_state=42,
            method="eigsh",
            tol=1e-3,
            maxiter=5000,
            ncv=5,
            diagnostics=diagnostics,
            deterministic=True,
        )

        self.assertIsInstance(embedding, cp.ndarray)
        self.assertEqual(embedding.shape, (30, 2))
        self.assertEqual(diagnostics["spectral_connected_components"], 5)
        self.assertTrue(diagnostics["spectral_feature_data_required"])
        self.assertTrue(diagnostics["spectral_feature_data_materialized"])
        self.assertFalse(diagnostics["spectral_fallback"])

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_spectral_final_jitter_makes_coordinates_unique(self):
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse

        from ibumap.init import gpu_init

        data = cp.zeros((3, 2), dtype=cp.float32)
        graph = cupyx_sparse.csr_matrix(cp.eye(3, dtype=cp.float32))
        spectral_layout = cp.asarray(
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
            dtype=cp.float32,
        )
        diagnostics = {}
        with (
            patch(
                "ibumap.init.gpu_init.spectral_layout_cupy",
                return_value=spectral_layout,
            ),
            patch.object(
                gpu_init,
                "_duplicate_row_stats",
                side_effect=AssertionError("duplicate scan must be disabled"),
            ),
        ):
            embedding = gpu_init.initialize_embedding_gpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(43),
                "euclidean",
                {},
                init_diagnostics=diagnostics,
            )

        self.assertEqual(int(cp.unique(embedding, axis=0).shape[0]), 3)
        self.assertEqual(diagnostics["init_final_jitter_scale"], 1e-4)
        self.assertNotIn("init_duplicate_ratio", diagnostics)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_spectral_axis_orientation_reports_remaining_duplicates(self):
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse

        from ibumap.init import gpu_init

        data = cp.zeros((4, 2), dtype=cp.float32)
        graph = cupyx_sparse.csr_matrix(cp.eye(4, dtype=cp.float32))
        diagnostics = {}
        with (
            self.assertWarnsRegex(RuntimeWarning, "25.00%"),
            patch(
                "ibumap.init.gpu_init.spectral_layout_cupy",
                return_value=cp.asarray(
                    [[10.0, 10.0], [10.0, 10.0], [9.9, 9.8], [0.0, 0.0]],
                    dtype=cp.float32,
                ),
            ),
            patch.object(gpu_init, "_FINAL_JITTER_SCALE", 0.0),
        ):
            embedding = gpu_init.initialize_embedding_gpu(
                data,
                graph,
                2,
                "spectral",
                np.random.RandomState(45),
                "euclidean",
                {},
                report_duplicate_ratio=True,
                init_diagnostics=diagnostics,
            )

        self.assertEqual(int(cp.unique(embedding, axis=0).shape[0]), 3)
        self.assertEqual(
            diagnostics["init_axis_orientation_reflected"],
            [True, True],
        )
        self.assertEqual(diagnostics["init_post_jitter_duplicate_rows"], 1)
        self.assertEqual(diagnostics["init_duplicate_rows"], 1)
        self.assertEqual(diagnostics["init_duplicate_ratio"], 0.25)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_fit_transform_from_knn_spectral_records_cuda_diagnostics(self):
        X, knn_indices, knn_distances, _ = _fixed_inputs()
        model = _model("spectral")

        embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

        self.assert_embedding_ok(embedding)
        costs = model.get_time_costs()
        for key in (
            "init_transfer_time",
            "init_connected_components_time",
            "init_laplacian_time",
            "init_solver_setup_time",
            "init_eigensolver_time",
            "init_postprocess_time",
            # algorithm="ibumap" resolves spectral_scale_policy="auto" to fft_compact
            "init_spectral_cast_scale_time",
            "init_spectral_compact_time",
        ):
            self.assertIn(key, costs)
            self.assertGreaterEqual(costs[key], 0.0)

        diagnostics = model.get_init_diagnostics()
        self.assertEqual(diagnostics["init_method"], "spectral")
        self.assertEqual(diagnostics["spectral_result_device"], "gpu")
        self.assertFalse(diagnostics["spectral_fallback"])
        self.assertEqual(diagnostics["spectral_connected_components"], 1)
        self.assertFalse(
            diagnostics["spectral_component_labels_canonicalized"]
        )
        self.assertEqual(
            diagnostics["spectral_component_label_order"],
            "trivial_single_component",
        )
        self.assertEqual(diagnostics["spectral_method"], "eigsh")
        self.assertTrue(diagnostics["spectral_deterministic_requested"])
        self.assertIn(
            diagnostics["spectral_spmv_algorithm"],
            {"CUSPARSE_SPMV_CSR_ALG2", "CUSPARSE_CSRMV_ALG2"},
        )
        self.assertTrue(diagnostics["spectral_spmv_bitwise_deterministic"])
        self.assertEqual(
            diagnostics["spectral_csr_preparation"],
            "upstream_canonical_reused",
        )
        self.assertFalse(diagnostics["spectral_csr_solver_copy"])
        self.assertEqual(diagnostics["spectral_initial_guess"], "none")
        self.assertFalse(diagnostics["spectral_initial_guess_generated"])
        self.assertEqual(
            diagnostics["spectral_degree_reduction"],
            "deterministic_csr_row_sum",
        )
        self.assertEqual(
            diagnostics["spectral_laplacian_construction"],
            "deterministic_csr_direct_scaling",
        )
        self.assertIn("spectral_tol", diagnostics)
        self.assertIn("spectral_maxiter", diagnostics)
        self.assertIn("spectral_ncv", diagnostics)
        self.assertEqual(costs["optimizer_degree_deterministic"], 1)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_fit_transform_from_knn_random_skips_eigensolver_diagnostics(self):
        X, knn_indices, knn_distances, _ = _fixed_inputs()
        model = _model("random")

        embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

        self.assert_embedding_ok(embedding)
        costs = model.get_time_costs()
        self.assertIn("init_random_time", costs)
        self.assertIn("init_normalize_time", costs)
        self.assertNotIn("init_eigensolver_time", costs)
        self.assertNotIn("init_laplacian_time", costs)
        self.assertEqual(model.get_init_diagnostics()["init_method"], "random")

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_spectral_eigensolver_failure_records_fallback_diagnostics(self):
        X, knn_indices, knn_distances, _ = _fixed_inputs()
        model = _model("spectral")

        with patch(
            "ibumap.init._spectral_cupy_impl._solve_eigsh_cupy",
            side_effect=RuntimeError("forced eigsh failure"),
        ), self.assertWarns(UserWarning):
            embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

        self.assert_embedding_ok(embedding)
        costs = model.get_time_costs()
        self.assertIn("init_fallback_time", costs)
        self.assertIn("init_spectral_compact_time", costs)
        diagnostics = model.get_init_diagnostics()
        self.assertTrue(diagnostics["spectral_fallback"])
        self.assertEqual(diagnostics["spectral_exception_type"], "RuntimeError")
        self.assertIn("forced eigsh failure", diagnostics["spectral_exception_message"])


if __name__ == "__main__":
    unittest.main()
