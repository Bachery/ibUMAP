import unittest
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import FFTConfig, PreparedInputs, IBUMAP
from ibumap.graph import CPUGraphThreadPolicy, build_cpu_graph_from_knn
from ibumap.utils import cal_degrees, preprocess_graph, preprocess_graph_csr
import pytest

# These tests keep the historical optimize_from_graph(X, graph, init) call, which
# CUDA still requires and CPU still accepts (with a FutureWarning).
# The combine_stages test checks the deprecated option itself.
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:optimize_from_graph\(X, fuzzy_graph, init_embedding\) is deprecated:FutureWarning"
    ),
    pytest.mark.filterwarnings(r"ignore:FFTConfig.combine_stages=True is deprecated:FutureWarning"),
]



N_SAMPLES = 128
N_FEATURES = 10
N_NEIGHBORS = 10
N_EPOCHS = 5


def _fixed_inputs():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="euclidean")
    nn.fit(X)
    knn_distances, knn_indices = nn.kneighbors(X)
    graph = build_cpu_graph_from_knn(
        X,
        n_neighbors=N_NEIGHBORS,
        metric="euclidean",
        metric_kwds={},
        random_state=42,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        angular_rp_forest=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        densmap_or_output_dens=False,
        verbose=False,
    )
    init_embedding = rng.uniform(0.0, 10.0, size=(N_SAMPLES, 2)).astype(np.float32)
    return X, knn_indices.astype(np.int64), knn_distances.astype(np.float32), graph, init_embedding


def _model(algorithm, device="cpu"):
    return IBUMAP(
        algorithm=algorithm,
        device=device,
        n_neighbors=N_NEIGHBORS,
        n_epochs=N_EPOCHS,
        random_state=42,
        deterministic=True,
    )


def _gpu_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class FixedInputEntrypointsTest(unittest.TestCase):
    def assert_embedding_ok(self, embedding):
        self.assertEqual(embedding.shape, (N_SAMPLES, 2))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())

    def test_csr_graph_preprocess_matches_legacy_coo_preprocess(self):
        rows = np.asarray([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int32)
        cols = np.asarray([1, 1, 2, 0, 2, 3, 0, 2], dtype=np.int32)
        data = np.asarray([0.2, 0.3, 0.001, 0.4, 0.0, 0.8, 0.01, 0.7])
        graph = sparse.coo_matrix((data, (rows, cols)), shape=(4, 4)).tocsr()
        graph_before = graph.copy()

        legacy_coo, legacy_epochs = preprocess_graph(graph, n_epochs=10)
        fast_csr, fast_epochs, degrees, stats = preprocess_graph_csr(
            graph,
            n_epochs=10,
            dtype=np.float64,
            return_degrees=True,
        )

        self.assertEqual(fast_epochs, legacy_epochs)
        diff = (fast_csr != legacy_coo.tocsr()).nnz
        self.assertEqual(diff, 0)
        np.testing.assert_allclose(
            degrees,
            cal_degrees(fast_csr.shape[0], fast_csr.indptr, fast_csr.data),
        )
        self.assertEqual(stats["input_nnz"], graph_before.nnz)
        self.assertEqual(stats["nnz"], fast_csr.nnz)
        self.assertGreater(stats["removed_edges"], 0)
        np.testing.assert_array_equal(graph.indptr, graph_before.indptr)
        np.testing.assert_array_equal(graph.indices, graph_before.indices)
        np.testing.assert_array_equal(graph.data, graph_before.data)

    def test_cpu_umap_preserves_umap_learn_coo_edge_weight_order(self):
        rows = np.asarray([2, 0, 1, 0, 2, 1, 0], dtype=np.int32)
        cols = np.asarray([0, 2, 0, 2, 1, 2, 1], dtype=np.int32)
        weights = np.asarray(
            [0.8, 0.2, 0.4, 0.3, 0.01, 0.7, 0.001],
            dtype=np.float32,
        )
        graph = sparse.coo_matrix((weights, (rows, cols)), shape=(3, 3))
        graph_before = graph.copy()

        expected = graph.tocoo(copy=True)
        expected.sum_duplicates()
        expected.data[expected.data < expected.data.max() / 20.0] = 0.0
        expected.eliminate_zeros()

        def capture_optimizer(*args, **kwargs):
            return args[0], {}

        model = IBUMAP(
            algorithm="umap",
            n_neighbors=2,
            n_epochs=20,
            random_state=42,
            deterministic=True,
        )
        with patch(
            "ibumap.pipeline.cpu_pipeline.umap_optimize_layout_euclidean",
            side_effect=capture_optimizer,
        ):
            model.optimize_from_graph(
                graph,
                np.zeros((3, 2), dtype=np.float32),
            )

        state = model._pipeline.state
        self.assertTrue(sparse.isspmatrix_coo(state.graph))
        np.testing.assert_array_equal(state.head, expected.row)
        np.testing.assert_array_equal(state.tail, expected.col)
        np.testing.assert_array_equal(state.weights, expected.data)
        np.testing.assert_array_equal(graph.row, graph_before.row)
        np.testing.assert_array_equal(graph.col, graph_before.col)
        np.testing.assert_array_equal(graph.data, graph_before.data)

    def test_prepared_cpu_umap_reconstructs_the_same_ordered_edge_stream(self):
        rows = np.asarray([2, 0, 1, 0, 2, 1, 0], dtype=np.int32)
        cols = np.asarray([0, 2, 0, 2, 1, 2, 1], dtype=np.int32)
        weights = np.asarray(
            [0.8, 0.2, 0.4, 0.3, 0.01, 0.7, 0.001],
            dtype=np.float32,
        )
        graph = sparse.coo_matrix((weights, (rows, cols)), shape=(3, 3))
        X = np.zeros((3, 2), dtype=np.float32)

        expected = graph.tocoo(copy=True)
        expected.sum_duplicates()
        expected.data[expected.data < expected.data.max() / 20.0] = 0.0
        expected.eliminate_zeros()

        prepare_model = IBUMAP(
            algorithm="umap",
            n_neighbors=2,
            n_epochs=20,
            random_state=42,
            deterministic=True,
        )
        prepared = prepare_model.prepare_fixed_inputs_from_graph(
            X,
            graph,
            return_init=False,
        )

        def capture_optimizer(*args, **kwargs):
            return args[0], {}

        model = IBUMAP(
            algorithm="umap",
            n_neighbors=2,
            n_epochs=20,
            random_state=42,
            deterministic=True,
        )
        with patch(
            "ibumap.pipeline.cpu_pipeline.umap_optimize_layout_euclidean",
            side_effect=capture_optimizer,
        ):
            model.optimize_from_prepared_graph(
                X,
                prepared.optimizer_graph,
                np.zeros((3, 2), dtype=np.float32),
            )

        state = model._pipeline.state
        np.testing.assert_array_equal(state.head, expected.row)
        np.testing.assert_array_equal(state.tail, expected.col)
        np.testing.assert_array_equal(state.weights, expected.data)

    def test_ibumap_graph_state_stays_on_csr_preprocessing_path(self):
        X, _, _, graph, init_embedding = _fixed_inputs()

        def capture_optimizer(*args, **kwargs):
            return args[0], {}

        model = _model("ibumap")
        with patch(
            "ibumap.pipeline.cpu_pipeline.preprocess_graph",
            side_effect=AssertionError("ibUMAP must not enter ordered COO preprocessing"),
        ), patch(
            "ibumap.pipeline.cpu_pipeline.umap_true_loss_optimization",
            side_effect=capture_optimizer,
        ):
            model.optimize_from_graph(X, graph, init_embedding)

        self.assertTrue(sparse.isspmatrix_csr(model._pipeline.state.graph))
        self.assertIsNone(model._pipeline.state.head)
        self.assertIsNone(model._pipeline.state.tail)
        self.assertIsNotNone(model._pipeline.state.degrees)

    def test_prepare_fixed_inputs_cpu_returns_raw_and_optimizer_graphs(self):
        X, _, _, _, _ = _fixed_inputs()
        model = _model("umap")

        bundle = model.prepare_fixed_inputs(X, device="cpu")

        self.assertIsInstance(bundle, PreparedInputs)
        self.assertEqual(bundle.device, "cpu")
        self.assertTrue(bundle.host_output)
        self.assertEqual(bundle.knn_indices.shape, (N_SAMPLES, N_NEIGHBORS))
        self.assertEqual(bundle.knn_indices.dtype, np.int64)
        self.assertEqual(bundle.knn_distances.shape, (N_SAMPLES, N_NEIGHBORS))
        self.assertEqual(bundle.knn_distances.dtype, np.float32)
        self.assertEqual(bundle.fuzzy_graph.shape, (N_SAMPLES, N_SAMPLES))
        self.assertEqual(bundle.optimizer_graph.shape, (N_SAMPLES, N_SAMPLES))
        self.assertLessEqual(bundle.optimizer_graph.nnz, bundle.fuzzy_graph.nnz)
        self.assertTrue(bundle.optimizer_graph.has_canonical_format)
        self.assertTrue(bundle.optimizer_graph.has_sorted_indices)
        self.assertEqual(bundle.init_embedding.shape, (N_SAMPLES, 2))

    def test_prepare_fixed_inputs_shared_rng_reuses_graph_state_for_init(self):
        X, knn_indices, knn_distances, graph, init_embedding = _fixed_inputs()
        random_state_ids = []
        policy = CPUGraphThreadPolicy(
            requested_n_jobs=-1,
            effective_n_jobs=8,
            deterministic=True,
            policy="auto_fixed_parallel",
            runtime_verified=True,
            thread_capacity=16,
            threading_layer="tbb",
        )

        def fake_graph(*args, **kwargs):
            self.assertTrue(kwargs["preserve_edge_order"])
            self.assertEqual(kwargs["n_jobs"], 8)
            random_state_ids.append(id(kwargs["random_state"]))
            kwargs["random_state"].randint(0, 2**31 - 1)
            return graph, knn_indices, knn_distances, None

        def fake_init(X, graph, n_components, init, random_state, *args, **kwargs):
            self.assertTrue(sparse.isspmatrix_coo(graph))
            random_state_ids.append(id(random_state))
            random_state.randint(0, 2**31 - 1)
            return init_embedding.copy()

        model = IBUMAP(
            algorithm="umap",
            n_neighbors=N_NEIGHBORS,
            n_epochs=N_EPOCHS,
            random_state=42,
            rng_lifecycle="shared_rng",
        )
        with patch(
            "ibumap.prepared.build_cpu_graph",
            side_effect=fake_graph,
        ), patch(
            "ibumap.prepared.resolve_cpu_graph_thread_policy",
            return_value=policy,
        ), patch(
            "ibumap.prepared.initialize_embedding_cpu",
            side_effect=fake_init,
        ):
            bundle = model.prepare_fixed_inputs(X)

        self.assertEqual(random_state_ids[0], random_state_ids[1])
        np.testing.assert_array_equal(bundle.init_embedding, init_embedding)
        self.assertTrue(np.isfinite(bundle.init_embedding).all())
        self.assertIn("knn_and_fuzzy_graph_time", bundle.timings)
        self.assertIn("graph_preprocess_time", bundle.timings)
        self.assertIn("initialization_time", bundle.timings)
        self.assertIn("optimizer_graph", bundle.diagnostics)
        self.assertEqual(
            bundle.diagnostics["cpu_graph_thread_policy"][
                "graph_n_jobs_effective"
            ],
            8,
        )
        self.assertEqual(bundle.effective_config["preparation"]["device"], "cpu")
        self.assertFalse(model._fitted)
        self.assertIsNone(model.graph_)

    def test_prepare_fixed_inputs_from_knn_skips_search_and_honors_return_flags(self):
        X, knn_indices, knn_distances, _, _ = _fixed_inputs()
        model = _model("umap")

        with patch(
            "ibumap.graph.cpu_graph.nearest_neighbors",
            side_effect=AssertionError("nearest-neighbor search should be skipped"),
        ):
            bundle = model.prepare_fixed_inputs_from_knn(
                X,
                knn_indices,
                knn_distances,
                return_knn=True,
                return_raw_graph=False,
                return_optimizer_graph=True,
                return_init=False,
            )

        np.testing.assert_array_equal(bundle.knn_indices, knn_indices)
        np.testing.assert_array_equal(bundle.knn_distances, knn_distances)
        self.assertIsNone(bundle.fuzzy_graph)
        self.assertIsNotNone(bundle.optimizer_graph)
        self.assertIsNone(bundle.init_embedding)
        self.assertEqual(bundle.diagnostics["knn_backend"], "caller_provided")

    def test_prepare_fixed_inputs_from_graph_reuses_raw_graph(self):
        X, _, _, graph, _ = _fixed_inputs()
        model = _model("umap")

        with patch(
            "ibumap.graph.cpu_graph.nearest_neighbors",
            side_effect=AssertionError("nearest-neighbor search should be skipped"),
        ):
            bundle = model.prepare_fixed_inputs_from_graph(X, graph)

        self.assertIsNone(bundle.knn_indices)
        self.assertIsNone(bundle.knn_distances)
        self.assertEqual(bundle.diagnostics["source"], "raw_graph")
        self.assertEqual(bundle.fuzzy_graph.shape, graph.shape)
        self.assertTrue(bundle.optimizer_graph.has_canonical_format)
        self.assertEqual(bundle.init_embedding.shape, (N_SAMPLES, 2))

    def test_optimize_from_prepared_graph_skips_graph_preprocessing(self):
        X, _, _, _, _ = _fixed_inputs()
        prepared = _model("umap").prepare_fixed_inputs(X)
        model = _model("umap")

        with patch(
            "ibumap.pipeline.cpu_pipeline.preprocess_graph_csr",
            side_effect=AssertionError("prepared graph must not be preprocessed again"),
        ):
            embedding = model.optimize_from_prepared_graph(
                prepared.optimizer_graph,
                prepared.init_embedding,
            )

        self.assert_embedding_ok(embedding)
        self.assertIsNone(model._raw_data)
        costs = model.get_time_costs()
        self.assertEqual(costs["graph_preprocess_time"], 0.0)
        self.assertEqual(costs["graph_preprocessing_bypassed"], 1)
        self.assertIn("optimization_from_prepared_graph_time", costs)
        self.assertEqual(
            model.explain_effective_config()["entrypoint_bypasses"],
            [
                "nearest_neighbor_search",
                "graph_construction",
                "graph_preprocessing",
                "initialization",
            ],
        )
        with self.assertRaisesRegex(NotImplementedError, "no raw X"):
            model.update_embedding()

    def test_prepared_graph_form_with_X_retains_update_data(self):
        X, _, _, _, _ = _fixed_inputs()
        prepared = _model("umap").prepare_fixed_inputs(X)
        model = _model("umap")

        embedding = model.optimize_from_prepared_graph(
            X,
            prepared.optimizer_graph,
            prepared.init_embedding,
        )

        self.assert_embedding_ok(embedding)
        np.testing.assert_array_equal(model._raw_data, X)

    def test_prepared_graph_form_without_X_validates_init_shape(self):
        X, _, _, _, _ = _fixed_inputs()
        prepared = _model("umap").prepare_fixed_inputs(X)
        model = _model("umap")

        with self.assertRaisesRegex(ValueError, "init_embedding shape"):
            model.optimize_from_prepared_graph(
                prepared.optimizer_graph,
                np.zeros((N_SAMPLES - 1, 2), dtype=np.float32),
            )

    def test_cpu_ibumap_prepared_graph_form_without_X(self):
        _, _, _, graph, init_embedding = _fixed_inputs()
        optimizer_graph, _, _, _ = preprocess_graph_csr(
            graph,
            N_EPOCHS,
            dtype=np.float32,
            return_degrees=False,
        )
        model = _model("ibumap", device="cpu")

        embedding = model.optimize_from_prepared_graph(
            optimizer_graph,
            init_embedding,
        )

        self.assert_embedding_ok(embedding)
        self.assertIsNone(model._raw_data)
        self.assertTrue(model.get_init_diagnostics()["graph_preprocessing_bypassed"])

    def test_cuda_prepared_graph_form_without_X_dispatches_without_raw_data(self):
        optimizer_graph = sparse.eye(N_SAMPLES, dtype=np.float32, format="csr")
        init_embedding = np.zeros((N_SAMPLES, 2), dtype=np.float32)
        calls = []

        class FakeCudaPipeline:
            def __init__(self):
                self.state = type("State", (), {"graph": optimizer_graph})()

            def optimize_from_prepared_graph(
                self,
                X=None,
                optimizer_graph=None,
                init_embedding=None,
                rng_state=None,
            ):
                calls.append(
                    {
                        "X": X,
                        "optimizer_graph": optimizer_graph,
                        "init_embedding": init_embedding,
                        "rng_state": rng_state,
                    }
                )
                return np.asarray(init_embedding, dtype=np.float32).copy()

        model = _model("ibumap", device="cuda")
        pipeline = FakeCudaPipeline()

        def install_pipeline(*, require_graph_backend=True):
            self.assertFalse(require_graph_backend)
            model._pipeline = pipeline
            model._pipeline_device = "cuda"

        with patch.object(model, "_ensure_pipeline", side_effect=install_pipeline):
            embedding = model.optimize_from_prepared_graph(
                optimizer_graph,
                init_embedding,
            )

        self.assert_embedding_ok(embedding)
        self.assertIsNone(model._raw_data)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["X"])
        self.assertIs(calls[0]["optimizer_graph"], optimizer_graph)
        self.assertIs(calls[0]["init_embedding"], init_embedding)
        self.assertIsNone(calls[0]["rng_state"])

    def test_cpu_ibumap_reports_repulsion_substage_times(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = IBUMAP(
            algorithm="ibumap",
            device="cpu",
            attraction_mode="sampling",
            repulsion_mode="true_loss",
            n_neighbors=N_NEIGHBORS,
            n_epochs=2,
            random_state=42,
            deterministic=True,
        )

        embedding = model.optimize_from_graph(X, graph, init_embedding)
        costs = model.get_time_costs()
        exclusive_keys = (
            "repl_setup_time",
            "repl_kernel_build_time",
            "repl_kernel_fft_time",
            "repl_point_setup_time",
            "repl_mesh_clear_time",
            "p2m_time",
            "repl_fft_plan_build_time",
            "repl_fft_forward_time",
            "repl_fft_multiply_time",
            "repl_fft_inverse_time",
            "repl_m2p_time",
            "repl_sampling_restore_time",
            "repl_ibfft_other_time",
            "repl_postprocess_time",
        )

        self.assert_embedding_ok(embedding)
        for key in exclusive_keys:
            self.assertIn(key, costs)
            self.assertGreaterEqual(costs[key], 0.0)
        self.assertAlmostEqual(
            sum(costs[key] for key in exclusive_keys),
            costs["repl_time"],
            places=9,
        )
        self.assertLessEqual(costs["repl_ibfft_total_time"], costs["repl_time"])
        self.assertEqual(
            costs["repl_kernel_cache_hits"] + costs["repl_kernel_cache_misses"],
            2,
        )
        self.assertGreaterEqual(costs["repl_kernel_cache_hit_rate"], 0.0)
        self.assertLessEqual(costs["repl_kernel_cache_hit_rate"], 1.0)
        self.assertEqual(
            costs["repl_fft_plan_cache_hits"] + costs["repl_fft_plan_cache_misses"],
            2,
        )
        self.assertGreaterEqual(costs["repl_fft_plan_cache_hit_rate"], 0.0)
        self.assertLessEqual(costs["repl_fft_plan_cache_hit_rate"], 1.0)
        self.assertGreaterEqual(costs["repl_grid_quantization_level_count"], 1)
        self.assertEqual(costs["fused_m2p_update_epochs"], 2)
        self.assertEqual(costs["fused_m2p_update_rate"], 1.0)
        self.assertEqual(costs["attraction_kernel_mode"], "thread_per_row")
        self.assertEqual(costs["attraction_schedule_mode"], "row_scan")
        self.assertEqual(costs["attraction_thread_per_row_epochs"], 2)
        self.assertEqual(costs["attraction_warp_per_row_epochs"], 0)
        self.assertEqual(costs["attraction_calendar_build_time"], 0.0)
        self.assertEqual(costs["attraction_calendar_execute_time"], 0.0)
        self.assertEqual(costs["attraction_active_event_ratio"], 0.0)
        self.assertEqual(costs["attraction_active_row_ratio"], 0.0)
        self.assertEqual(costs["appl_time"], 0.0)
        self.assertIn("graph_preprocess_time", costs)
        self.assertEqual(costs["graph_preprocess_input_nnz"], graph.nnz)
        self.assertEqual(costs["graph_preprocess_nnz"], model.graph_.nnz)
        self.assertGreaterEqual(costs["graph_preprocess_removed_edges"], 0)

    def test_cpu_ibumap_combine_stages_persists_each_interpolation_order(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        observed_orders = []

        def fake_ibfft(
            embedding,
            n_interpolation_points,
            *args,
            p2m_diagnostics=None,
            timing_diagnostics=None,
            return_grid_context=False,
            fused_update=None,
            **kwargs,
        ):
            observed_orders.append(int(n_interpolation_points))
            if p2m_diagnostics is not None:
                p2m_diagnostics.update(
                    {
                        "time_s": 0.0,
                        "resolved_mode": "serial",
                        "fft_plan_cache_hit": len(observed_orders) > 1,
                    }
                )
            if timing_diagnostics is not None:
                timing_diagnostics.update(
                    {
                        "setup_time_s": 0.0,
                        "kernel_build_time_s": 0.0,
                        "kernel_fft_time_s": 0.0,
                        "point_setup_time_s": 0.0,
                        "mesh_clear_time_s": 0.0,
                        "p2m_time_s": 0.0,
                        "fft_plan_build_time_s": 0.0,
                        "fft_forward_time_s": 0.0,
                        "fft_multiply_time_s": 0.0,
                        "fft_inverse_time_s": 0.0,
                        "m2p_time_s": 0.0,
                        "sampling_restore_time_s": 0.0,
                        "ibfft_other_time_s": 0.0,
                    }
                )
            if fused_update is not None:
                return None
            force = np.zeros_like(embedding)
            return (force, None) if return_grid_context else force

        with self.assertWarnsRegex(FutureWarning, "interpolation_schedule"):
            fft = FFTConfig(combine_stages=True)
        model = IBUMAP(
            algorithm="ibumap",
            device="cpu",
            attraction_mode="sampling",
            repulsion_mode="true_loss",
            fft=fft,
            n_neighbors=N_NEIGHBORS,
            n_epochs=20,
            random_state=42,
            deterministic=True,
        )

        with patch(
            "ibumap.optimizers.ibumap_optimizer.ibFFT_repulsive_sampling",
            side_effect=fake_ibfft,
        ):
            embedding = model.optimize_from_graph(X, graph, init_embedding)

        self.assert_embedding_ok(embedding)
        self.assertEqual(observed_orders, [1] * 18 + [2, 3])
        costs = model.get_time_costs()
        self.assertEqual(costs["fft_stage_schedule"], "0:18:p1,18:19:p2,19:20:p3")
        self.assertEqual(costs["fft_stage_transition_epochs"], "18,19")
        self.assertEqual(costs["fft_stage_epochs_p1"], 18)
        self.assertEqual(costs["fft_stage_epochs_p2"], 1)
        self.assertEqual(costs["fft_stage_epochs_p3"], 1)

    def test_ibumap_random_state_repeats_graph_init_and_sampling(self):
        X, _, _, _, _ = _fixed_inputs()

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cpu",
                attraction_mode="sampling",
                repulsion_mode="sampling",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=77,
                deterministic=False,
            ).fit_transform(X)

        first = run()
        second = run()

        np.testing.assert_array_equal(first, second)

    def test_ibumap_deterministic_repeats_without_explicit_seed(self):
        X, _, _, _, _ = _fixed_inputs()

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cpu",
                attraction_mode="sampling",
                repulsion_mode="sampling",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=None,
                deterministic=True,
            ).fit_transform(X)

        first = run()
        second = run()

        np.testing.assert_array_equal(first, second)

    def test_fit_transform_from_knn_cpu_skips_nearest_neighbor_search(self):
        X, knn_indices, knn_distances, _, _ = _fixed_inputs()

        for algorithm in ("umap", "ibumap"):
            with self.subTest(algorithm=algorithm):
                model = _model(algorithm)
                with patch(
                    "ibumap.graph.cpu_graph.nearest_neighbors",
                    side_effect=AssertionError("nearest-neighbor search should be skipped"),
                ):
                    embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

                self.assert_embedding_ok(embedding)
                costs = model.get_time_costs()
                self.assertIn("fixed_knn_total_time", costs)
                self.assertGreater(costs["fuzzy_graph_time"], 0.0)
                self.assertGreater(costs["build_graph_time"], 0.0)
                self.assertGreaterEqual(costs["embd_init_time"], 0.0)
                self.assertGreater(costs["embd_opt_time"], 0.0)
                self.assertIn("opt_prep_time", costs)
                report = model.explain_effective_config()
                self.assertEqual(report["parameter_usage_path"], "fixed_knn")
                self.assertEqual(
                    report["entrypoint_bypasses"], ["nearest_neighbor_search"]
                )

    def test_cpu_memory_diagnostics_write_stage_jsonl(self):
        X, knn_indices, knn_distances, _, _ = _fixed_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.jsonl"
            model = IBUMAP(
                algorithm="ibumap",
                device="cpu",
                attraction_mode="sampling",
                repulsion_mode="sampling",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=42,
                deterministic=True,
                diagnostics_memory_path=str(memory_path),
                diagnostics_memory_epoch_stride=2,
            )
            embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

            self.assert_embedding_ok(embedding)
            self.assertTrue(memory_path.exists())
            rows = [
                json.loads(line)
                for line in memory_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stages = {(row["stage"], row["event"]) for row in rows}
            self.assertIn(("graph_from_knn", "begin"), stages)
            self.assertIn(("initialization", "begin"), stages)
            self.assertIn(("optimization", "begin"), stages)
            self.assertIn(("optimizer_attraction", "begin"), stages)
            self.assertIn(("optimizer_repulsion", "end"), stages)
            self.assertIn(("optimizer_apply", "end"), stages)

            attraction_epochs = sorted(
                {
                    row["epoch"]
                    for row in rows
                    if row["stage"] == "optimizer_attraction"
                    and row["event"] == "begin"
                }
            )
            self.assertEqual(attraction_epochs, [0, 2, 4])
            self.assertTrue(
                all("cpu" in row and "ru_maxrss_bytes" in row["cpu"] for row in rows)
            )
            self.assertIsNone(model._memory_recorder)

    def test_optimize_from_graph_cpu_skips_graph_construction_and_initialization(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        graph_before = graph.copy()

        for algorithm in ("umap", "ibumap"):
            with self.subTest(algorithm=algorithm):
                init_for_run = init_embedding.copy()
                init_before = init_for_run.copy()
                model = _model(algorithm)

                with patch(
                    "ibumap.graph.cpu_graph.nearest_neighbors",
                    side_effect=AssertionError("nearest-neighbor search should be skipped"),
                ), patch(
                    "ibumap.graph.cpu_graph.fuzzy_simplicial_set",
                    side_effect=AssertionError("fuzzy graph construction should be skipped"),
                ), patch(
                    "ibumap.pipeline.cpu_pipeline.initialize_embedding_cpu",
                    side_effect=AssertionError("initialization should be skipped"),
                ):
                    embedding = model.optimize_from_graph(X, graph, init_for_run)

                self.assert_embedding_ok(embedding)
                np.testing.assert_array_equal(init_for_run, init_before)
                np.testing.assert_array_equal(graph.indptr, graph_before.indptr)
                np.testing.assert_array_equal(graph.indices, graph_before.indices)
                np.testing.assert_array_equal(graph.data, graph_before.data)
                costs = model.get_time_costs()
                self.assertIn("optimization_only_time", costs)
                self.assertEqual(costs["build_graph_time"], 0.0)
                self.assertEqual(costs["fuzzy_graph_time"], 0.0)
                self.assertEqual(costs["embd_init_time"], 0.0)
                self.assertIn("graph_preprocess_time", costs)
                self.assertEqual(costs["graph_preprocess_input_nnz"], graph.nnz)
                self.assertEqual(costs["graph_preprocess_nnz"], model.graph_.nnz)
                self.assertGreater(costs["embd_opt_time"], 0.0)
                self.assertIn("opt_prep_time", costs)
                init_diagnostics = model.get_init_diagnostics()
                self.assertEqual(init_diagnostics["init_method"], "external")
                self.assertTrue(init_diagnostics["initialization_bypassed"])
                report = model.explain_effective_config()
                self.assertEqual(
                    report["parameter_usage_path"], "optimization_only"
                )
                self.assertIn("initialization", report["entrypoint_bypasses"])

    def test_optimization_only_explicit_rng_state_is_forwarded_as_private_int64_copy(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        explicit_state = np.asarray([11, -22, 33], dtype=np.int32)
        state_before = explicit_state.copy()
        captured = []

        def capture_optimizer(*args, **kwargs):
            captured.append(np.array(args[6], copy=True))
            return args[0], {}

        with patch(
            "ibumap.pipeline.cpu_pipeline.umap_optimize_layout_euclidean",
            side_effect=capture_optimizer,
        ):
            _model("umap").optimize_from_graph(
                graph,
                init_embedding,
                rng_state=explicit_state,
            )

            optimizer_graph, _, _, _ = preprocess_graph_csr(
                graph,
                N_EPOCHS,
                dtype=np.float32,
                return_degrees=False,
            )
            _model("umap").optimize_from_prepared_graph(
                X,
                optimizer_graph,
                init_embedding,
                rng_state=explicit_state,
            )

        self.assertEqual(len(captured), 2)
        for actual in captured:
            self.assertEqual(actual.dtype, np.int64)
            np.testing.assert_array_equal(actual, state_before)
        np.testing.assert_array_equal(explicit_state, state_before)

    def test_optimization_only_rng_state_validation_and_scope(self):
        _, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("umap")

        with self.assertRaisesRegex(ValueError, r"shape \(3,\)"):
            model.optimize_from_graph(
                graph,
                init_embedding,
                rng_state=np.asarray([1, 2], dtype=np.int64),
            )
        with self.assertRaisesRegex(TypeError, "integers"):
            model.optimize_from_graph(
                graph,
                init_embedding,
                rng_state=np.asarray([1.0, 2.0, 3.0]),
            )
        with self.assertRaisesRegex(ValueError, "CPU algorithm='umap'"):
            _model("ibumap").optimize_from_graph(
                graph,
                init_embedding,
                rng_state=np.asarray([1, 2, 3], dtype=np.int64),
            )

    def test_rng_lifecycle_does_not_control_cpu_umap_parallel_switch(self):
        _, _, _, graph, init_embedding = _fixed_inputs()

        for deterministic, expected_parallel in ((False, True), (True, False)):
            with self.subTest(deterministic=deterministic):
                captured = {}

                def capture_optimizer(*args, **kwargs):
                    captured["parallel"] = kwargs["parallel"]
                    return args[0], {}

                model = IBUMAP(
                    algorithm="umap",
                    device="cpu",
                    n_neighbors=N_NEIGHBORS,
                    n_epochs=N_EPOCHS,
                    random_state=42,
                    deterministic=deterministic,
                    rng_lifecycle="shared_rng",
                )
                with patch(
                    "ibumap.pipeline.cpu_pipeline.umap_optimize_layout_euclidean",
                    side_effect=capture_optimizer,
                ):
                    model.optimize_from_graph(graph, init_embedding)

                self.assertEqual(captured["parallel"], expected_parallel)

    def test_ibumap_state_does_not_retain_coo_indices(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap")

        model.optimize_from_graph(X, graph, init_embedding)

        self.assertIsNone(model._pipeline.state.head)
        self.assertIsNone(model._pipeline.state.tail)
        self.assertIsNotNone(model._pipeline.state.degrees)

    def test_init_diagnostics_are_returned_as_copy(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap")

        model.optimize_from_graph(X, graph, init_embedding)
        diagnostics = model.get_init_diagnostics()
        diagnostics["init_method"] = "mutated"

        self.assertEqual(model.get_init_diagnostics()["init_method"], "external")

    def test_ibumap_update_reuses_cached_degrees(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap")
        model.optimize_from_graph(X, graph, init_embedding)
        cached = model._pipeline.state.degrees

        with patch(
            "ibumap.pipeline.cpu_pipeline.cal_degrees",
            side_effect=AssertionError("degrees should be cached in pipeline state"),
        ):
            model.update_embedding(n_epochs=2)

        self.assertIs(model._pipeline.state.degrees, cached)

    def test_ibumap_update_reuses_fast_cpu_workspace(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap")
        model.optimize_from_graph(X, graph, init_embedding)
        attr_force = model._pipeline.workspace.buffers["optimizer.attr_force"]
        box_idx = model._pipeline.workspace.buffers["box_idx"]
        self.assertNotIn("neg_f", model._pipeline.workspace.buffers)
        self.assertEqual(model.get_time_costs()["fused_m2p_update_epochs"], N_EPOCHS)

        model.update_embedding(n_epochs=2)

        self.assertIs(
            model._pipeline.workspace.buffers["optimizer.attr_force"], attr_force
        )
        self.assertIs(model._pipeline.workspace.buffers["box_idx"], box_idx)
        self.assertNotIn("neg_f", model._pipeline.workspace.buffers)
        self.assertEqual(model.get_time_costs()["fused_m2p_update_epochs"], 2)

    def test_total_update_clip_uses_unfused_force_fallback(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        fused_model = IBUMAP(
            algorithm="ibumap",
            device="cpu",
            n_neighbors=N_NEIGHBORS,
            n_epochs=2,
            random_state=42,
            deterministic=True,
        )
        model = IBUMAP(
            algorithm="ibumap",
            device="cpu",
            n_neighbors=N_NEIGHBORS,
            n_epochs=2,
            random_state=42,
            deterministic=True,
            total_update_clip_norm=1e6,
        )

        fused_embedding = fused_model.optimize_from_graph(X, graph, init_embedding)
        embedding = model.optimize_from_graph(X, graph, init_embedding)

        self.assert_embedding_ok(embedding)
        np.testing.assert_allclose(embedding, fused_embedding, rtol=2e-6, atol=2e-6)
        self.assertEqual(model.get_time_costs()["fused_m2p_update_epochs"], 0)
        self.assertIn("neg_f", model._pipeline.workspace.buffers)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_ibumap_update_reuses_fast_gpu_workspace(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap", device="cuda")
        model.optimize_from_graph(X, graph, init_embedding)
        initial_costs = model.get_time_costs()
        self.assertEqual(
            initial_costs["repl_fft_plan_cache_hits"]
            + initial_costs["repl_fft_plan_cache_misses"],
            N_EPOCHS,
        )
        self.assertEqual(model._pipeline.workspace.fft_snapshot()["plan_entries"], 2)
        self.assertIn("ibfft.fft.frequency", model._pipeline.workspace.buffers)
        self.assertIn("ibfft.fft.output", model._pipeline.workspace.buffers)
        buffers_before = {
            key: value
            for key, value in model._pipeline.workspace.buffers.items()
            if key.startswith("optimizer.") or key.startswith("ibfft.")
        }

        model.update_embedding(n_epochs=N_EPOCHS)

        buffers_after = model._pipeline.workspace.buffers
        self.assertEqual(set(buffers_after), set(buffers_before))
        for key, value in buffers_before.items():
            self.assertIs(buffers_after[key], value, key)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_clear_workspace_drops_gpu_buffers(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap", device="cuda")
        model.optimize_from_graph(X, graph, init_embedding)
        self.assertTrue(model._pipeline.workspace.buffers)

        model.clear_workspace()

        self.assertEqual(model._pipeline.workspace.buffers, {})
        self.assertEqual(
            model._pipeline.workspace.fft_snapshot(),
            {"plan_entries": 0, "plan_hits": 0, "plan_misses": 0},
        )

    def test_default_ibumap_skips_local_diagnostic_objects(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("ibumap")

        with patch(
            "ibumap.optimizers.ibumap_optimizer._local_force_diagnostics",
            side_effect=AssertionError("disabled diagnostics should take the fast path"),
        ):
            model.optimize_from_graph(X, graph, init_embedding)

    def test_cpu_umap_state_retains_required_coo_indices(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = _model("umap")

        model.optimize_from_graph(X, graph, init_embedding)

        self.assertIsNotNone(model._pipeline.state.head)
        self.assertIsNotNone(model._pipeline.state.tail)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_fit_transform_from_knn_gpu_ibumap(self):
        X, knn_indices, knn_distances, _, _ = _fixed_inputs()
        model = _model("ibumap", device="cuda")

        embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)

        self.assert_embedding_ok(embedding)
        costs = model.get_time_costs()
        self.assertIn("fixed_knn_total_time", costs)
        self.assertGreater(costs["fuzzy_graph_time"], 0.0)
        self.assertGreaterEqual(costs["graph_to_gpu_time"], 0.0)
        self.assertGreater(costs["embd_opt_time"], 0.0)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_optimize_from_graph_gpu_ibumap_does_not_mutate_init(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        init_before = init_embedding.copy()
        model = _model("ibumap", device="cuda")

        embedding = model.optimize_from_graph(X, graph, init_embedding)

        self.assert_embedding_ok(embedding)
        np.testing.assert_array_equal(init_embedding, init_before)
        costs = model.get_time_costs()
        self.assertIn("optimization_only_time", costs)
        self.assertEqual(costs["build_graph_time"], 0.0)
        self.assertEqual(costs["embd_init_time"], 0.0)
        self.assertIn("graph_preprocess_time", costs)
        self.assertEqual(costs["graph_preprocess_input_nnz"], graph.nnz)
        self.assertEqual(costs["graph_preprocess_nnz"], model.graph_.nnz)
        self.assertGreaterEqual(costs["graph_to_gpu_time"], 0.0)
        self.assertGreater(costs["embd_opt_time"], 0.0)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_prepare_fixed_inputs_cuda_returns_host_artifacts(self):
        X, _, _, _, _ = _fixed_inputs()
        model = _model("ibumap", device="cuda")

        bundle = model.prepare_fixed_inputs(X, device="cuda", host_output=True)

        self.assertEqual(bundle.device, "cuda")
        self.assertTrue(bundle.host_output)
        self.assertIsInstance(bundle.knn_indices, np.ndarray)
        self.assertIsInstance(bundle.knn_distances, np.ndarray)
        self.assertIsInstance(bundle.init_embedding, np.ndarray)
        self.assertEqual(bundle.fuzzy_graph.shape, (N_SAMPLES, N_SAMPLES))
        self.assertEqual(bundle.optimizer_graph.shape, (N_SAMPLES, N_SAMPLES))
        self.assert_embedding_ok(bundle.init_embedding)
        self.assertEqual(bundle.diagnostics["knn_backend"], "cuml.neighbors.NearestNeighbors")
        self.assertIn("initialization", bundle.diagnostics)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_optimize_from_graph_gpu_deterministic_sampling_repeats(self):
        X, _, _, graph, init_embedding = _fixed_inputs()

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cuda",
                attraction_mode="true_loss",
                repulsion_mode="sampling",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=42,
                deterministic=True,
            ).optimize_from_graph(X, graph, init_embedding)

        np.testing.assert_array_equal(run(), run())

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_optimize_from_graph_gpu_ibumap_does_not_mutate_init_and_repeats(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        init_before = init_embedding.copy()

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cuda",
                attraction_mode="true_loss",
                repulsion_mode="true_loss",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=42,
                deterministic=True,
            ).optimize_from_graph(X, graph, init_embedding)

        first = run()
        second = run()

        np.testing.assert_array_equal(init_embedding, init_before)
        np.testing.assert_array_equal(first, second)
        self.assert_embedding_ok(first)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_optimize_from_graph_gpu_ibumap_fixed_epoch_counts(self):
        X, _, _, graph, init_embedding = _fixed_inputs()

        for n_epochs in (1, N_EPOCHS):
            with self.subTest(n_epochs=n_epochs):
                embedding = IBUMAP(
                    algorithm="ibumap",
                    device="cuda",
                    attraction_mode="true_loss",
                    repulsion_mode="true_loss",
                    n_neighbors=N_NEIGHBORS,
                    n_epochs=n_epochs,
                    random_state=42,
                    deterministic=True,
                ).optimize_from_graph(X, graph, init_embedding)
                self.assert_embedding_ok(embedding)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_gpu_ibumap_stage_times_are_cuda_event_totals(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        model = IBUMAP(
            algorithm="ibumap",
            device="cuda",
            attraction_mode="true_loss",
            repulsion_mode="true_loss",
            n_neighbors=N_NEIGHBORS,
            n_epochs=N_EPOCHS,
            random_state=42,
            deterministic=False,
        )

        embedding = model.optimize_from_graph(X, graph, init_embedding)
        costs = model.get_time_costs()
        stage_total = costs["attr_time"] + costs["repl_time"] + costs["appl_time"]

        self.assert_embedding_ok(embedding)
        self.assertGreaterEqual(costs["attr_time"], 0.0)
        self.assertGreaterEqual(costs["repl_time"], 0.0)
        self.assertGreaterEqual(costs["appl_time"], 0.0)
        self.assertGreater(stage_total, 0.0)
        self.assertLessEqual(stage_total, costs["embd_opt_time"] * 1.5 + 0.05)
        self.assertEqual(costs["fused_m2p_update_epochs"], N_EPOCHS)
        self.assertEqual(costs["fused_m2p_update_rate"], 1.0)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_gpu_ibumap_fused_repulsion_clip_repeats(self):
        X, _, _, graph, init_embedding = _fixed_inputs()

        def run():
            return IBUMAP(
                algorithm="ibumap",
                device="cuda",
                attraction_mode="true_loss",
                repulsion_mode="true_loss",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=42,
                deterministic=True,
                repulsion_clip_norm=0.25,
            ).optimize_from_graph(X, graph, init_embedding)

        first = run()
        second = run()

        self.assert_embedding_ok(first)
        np.testing.assert_array_equal(first, second)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_gpu_ibumap_optimize_from_prepared_graph_accepts_default_rng_state(self):
        X, _, _, graph, init_embedding = _fixed_inputs()
        optimizer_graph, _, _, _ = preprocess_graph_csr(
            graph,
            N_EPOCHS,
            dtype=np.float32,
            return_degrees=False,
        )
        model = IBUMAP(
            algorithm="ibumap",
            device="cuda",
            n_neighbors=N_NEIGHBORS,
            n_epochs=N_EPOCHS,
            random_state=42,
            deterministic=True,
        )

        embedding = model.optimize_from_prepared_graph(
            optimizer_graph,
            init_embedding,
        )

        self.assert_embedding_ok(embedding)
        diagnostics = model.get_init_diagnostics()
        self.assertTrue(diagnostics["initialization_bypassed"])
        self.assertTrue(diagnostics["graph_preprocessing_bypassed"])
        self.assertEqual(model.get_time_costs()["p2m_resolved_mode"], "segmented")

    def test_gpu_umap_fixed_input_entrypoints_raise_instead_of_full_fallback(self):
        X, knn_indices, knn_distances, graph, init_embedding = _fixed_inputs()
        model = _model("umap", device="cuda")

        with self.assertRaises(NotImplementedError):
            model.fit_transform_from_knn(X, knn_indices, knn_distances)
        with self.assertRaises(NotImplementedError):
            model.optimize_from_graph(X, graph, init_embedding)


if __name__ == "__main__":
    unittest.main()
