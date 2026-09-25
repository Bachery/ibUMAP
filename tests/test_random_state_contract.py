import sys
import unittest
from numbers import Integral
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse

from ibumap import IBUMAP
from ibumap.graph.cpu_thread_policy import CPUGraphThreadPolicy
from ibumap.graph.gpu_graph_cuml import build_gpu_graph_cuml
from ibumap.kernels.cpu import ibfft
from ibumap.optimizers.ibumap_optimizer import _resolve_noise_seed
from ibumap.pipeline.cpu_pipeline import CPUPipeline
from ibumap.pipeline.gpu_pipeline import GPUPipeline
from ibumap.utils import derive_random_seed, get_rng_state


class RandomStateContractTest(unittest.TestCase):
    def test_cpu_pipeline_passes_resolved_deterministic_graph_threads(self):
        X = np.zeros((6, 4), dtype=np.float32)
        graph = sparse.csr_matrix((6, 6), dtype=np.float32)
        expected_state = object()
        policy = CPUGraphThreadPolicy(
            requested_n_jobs=-1,
            effective_n_jobs=8,
            deterministic=True,
            policy="auto_fixed_parallel",
            runtime_verified=True,
            thread_capacity=16,
            threading_layer="tbb",
        )
        model = SimpleNamespace(
            random_state=42,
            n_jobs=-1,
            n_neighbors=2,
            metric="euclidean",
            metric_kwds={},
            angular_rp_forest=False,
            low_memory=True,
            verbose=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            input_distance_func=None,
            runtime=SimpleNamespace(
                deterministic=True,
                algorithm="ibumap",
                device="cpu",
                rng_lifecycle="independent",
            ),
            _time_costs={},
            _memory_recorder=None,
        )
        pipeline = CPUPipeline(model)
        build_graph = Mock(return_value=(graph, None, None, None))
        resolve_policy = Mock(return_value=policy)

        with patch.dict(
            pipeline._prepare_state.__func__.__globals__,
            {
                "build_cpu_graph": build_graph,
                "resolve_cpu_graph_thread_policy": resolve_policy,
            },
        ), patch.object(
            pipeline,
            "_state_from_graph",
            return_value=expected_state,
        ):
            state = pipeline._prepare_state(X)

        self.assertIs(state, expected_state)
        self.assertEqual(build_graph.call_args.kwargs["n_jobs"], 8)
        self.assertEqual(model._time_costs["graph_n_jobs_effective"], 8)
        self.assertEqual(
            model._time_costs["graph_thread_policy"], "auto_fixed_parallel"
        )

    def test_none_preserves_unseeded_fast_path(self):
        for stream in ("graph", "init", "sampling", "noise"):
            self.assertIsNone(derive_random_seed(None, stream))

    def test_deterministic_pipeline_supplies_a_fixed_root_seed(self):
        deterministic = CPUPipeline(
            SimpleNamespace(
                random_state=None,
                runtime=SimpleNamespace(deterministic=True),
            )
        )
        default = CPUPipeline(
            SimpleNamespace(
                random_state=None,
                runtime=SimpleNamespace(deterministic=False),
            )
        )

        self.assertIsInstance(deterministic._random_seed("graph"), int)
        self.assertEqual(
            deterministic._random_seed("graph"),
            deterministic._random_seed("graph"),
        )
        self.assertIsNone(default._random_seed("graph"))

    def test_shared_rng_reuses_one_call_scoped_state_in_stage_order(self):
        pipeline = CPUPipeline(
            SimpleNamespace(
                random_state=42,
                runtime=SimpleNamespace(
                    deterministic=False,
                    rng_lifecycle="shared_rng",
                ),
            )
        )
        expected = np.random.RandomState(42)

        with pipeline._rng_run():
            graph_rng = pipeline._stage_random_source("graph")
            init_rng = pipeline._stage_random_state("init")
            self.assertIs(graph_rng, init_rng)
            self.assertEqual(
                graph_rng.randint(0, 2**31 - 1),
                expected.randint(0, 2**31 - 1),
            )
            np.testing.assert_array_equal(
                pipeline._sampling_rng_state(),
                get_rng_state(expected),
            )

        self.assertIsNone(pipeline._shared_random_state)

    def test_shared_rng_integer_seed_restarts_for_each_pipeline_call(self):
        pipeline = CPUPipeline(
            SimpleNamespace(
                random_state=19,
                runtime=SimpleNamespace(
                    deterministic=False,
                    rng_lifecycle="shared_rng",
                ),
            )
        )

        draws = []
        for _ in range(2):
            with pipeline._rng_run():
                draws.append(
                    pipeline._stage_random_state("init").randint(0, 2**31 - 1)
                )
        self.assertEqual(draws[0], draws[1])

    def test_cpu_umap_fit_consumes_shared_rng_across_graph_init_and_sampling(self):
        X = np.arange(24, dtype=np.float32).reshape(6, 4)
        graph = sparse.csr_matrix(
            np.asarray(
                [
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        random_state_ids = []
        captured_rng_state = []

        def fake_graph(*args, **kwargs):
            self.assertTrue(kwargs["preserve_edge_order"])
            random_state = kwargs["random_state"]
            random_state_ids.append(id(random_state))
            random_state.randint(0, 2**31 - 1)
            return graph, np.zeros((6, 2), dtype=np.int64), np.zeros((6, 2)), None

        def fake_init(X, graph, n_components, init, random_state, *args, **kwargs):
            random_state_ids.append(id(random_state))
            random_state.randint(0, 2**31 - 1)
            return np.zeros((X.shape[0], n_components), dtype=np.float32)

        def fake_optimize(*args, **kwargs):
            captured_rng_state.append(np.array(args[6], copy=True))
            return args[0], {}

        model = IBUMAP(
            algorithm="umap",
            n_neighbors=2,
            n_epochs=2,
            init="random",
            random_state=42,
            rng_lifecycle="shared_rng",
        )
        with patch(
            "ibumap.pipeline.cpu_pipeline.build_cpu_graph",
            side_effect=fake_graph,
        ), patch(
            "ibumap.pipeline.cpu_pipeline.initialize_embedding_cpu",
            side_effect=fake_init,
        ), patch(
            "ibumap.pipeline.cpu_pipeline.umap_optimize_layout_euclidean",
            side_effect=fake_optimize,
        ):
            model.fit_transform(X)

        expected = np.random.RandomState(42)
        expected.randint(0, 2**31 - 1)
        expected.randint(0, 2**31 - 1)
        self.assertEqual(random_state_ids[0], random_state_ids[1])
        np.testing.assert_array_equal(captured_rng_state[0], get_rng_state(expected))

    def test_gpu_pipeline_preserves_unseeded_default_seed_contract(self):
        deterministic = GPUPipeline(
            SimpleNamespace(
                numeric_dtype=np.dtype(np.float32),
                random_state=None,
                runtime=SimpleNamespace(deterministic=True),
            )
        )
        default = GPUPipeline(
            SimpleNamespace(
                numeric_dtype=np.dtype(np.float32),
                random_state=None,
                runtime=SimpleNamespace(deterministic=False),
            )
        )

        self.assertIsInstance(deterministic._random_seed("graph"), int)
        self.assertEqual(
            deterministic._random_seed("graph"),
            deterministic._random_seed("graph"),
        )
        self.assertIsNone(default._random_seed("graph"))

    def test_gpu_ibumap_fit_keeps_features_on_host_for_graph_and_init(self):
        X = np.arange(24, dtype=np.float32).reshape(6, 4)
        state = SimpleNamespace(
            graph=sparse.csr_matrix((6, 6), dtype=np.float32),
            edgesrc=np.zeros(7, dtype=np.int32),
            edgetgt=np.empty(0, dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            degrees=np.zeros(6, dtype=np.float32),
            n_vertices=6,
        )
        model = SimpleNamespace(
            numeric_dtype=np.dtype(np.float32),
            random_state=None,
            runtime=SimpleNamespace(
                algorithm="ibumap",
                graph_backend="cuml",
                deterministic=False,
                attraction_mode="sampling",
                repulsion_mode="true_loss",
            ),
            config=SimpleNamespace(
                runtime=SimpleNamespace(workspace_policy="performance"),
            ),
            n_components=2,
            negative_sample_rate=5,
            _time_costs={},
            _memory_recorder=None,
        )
        pipeline = GPUPipeline(model)
        cupy = ModuleType("cupy")
        cupy.float32 = np.float32
        cupy.asarray = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full feature input must not be materialized eagerly")
        )
        cupy.asnumpy = np.asarray

        def fake_prepare(received):
            self.assertIs(received, X)
            model._resolved_n_epochs = 2
            return state

        def fake_initialize(received, init):
            self.assertIs(received, X)
            self.assertEqual(init, "spectral")
            return np.zeros((6, 2), dtype=np.float32)

        fit_globals = pipeline.fit.__func__.__globals__
        with (
            patch.dict(sys.modules, {"cupy": cupy}),
            patch.dict(
                fit_globals,
                {
                    "ensure_gpu_runtime": Mock(),
                    "initialize_embedding_gpu": object(),
                    "make_epochs_per_sample_gpu": Mock(
                        return_value=np.empty(0, dtype=np.float32)
                    ),
                    "umap_true_loss_optimization": Mock(
                        side_effect=lambda embedding, *args, **kwargs: (
                            embedding,
                            {},
                        )
                    ),
                },
            ),
            patch.object(pipeline, "_prepare_state", side_effect=fake_prepare),
            patch.object(
                pipeline,
                "_initialize_embedding",
                side_effect=fake_initialize,
            ),
            patch.object(pipeline, "_legacy_params", return_value={}),
        ):
            embedding = pipeline.fit(X, init="spectral")

        self.assertEqual(embedding.shape, (6, 2))
        self.assertIsInstance(model.init_embedding_, np.ndarray)

    def test_cuml_graph_adapter_omits_random_state_for_none(self):
        calls = []

        def fake_fuzzy_simplicial_set(X, **kwargs):
            calls.append((X, kwargs))
            return "graph"

        with self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set):
            result = build_gpu_graph_cuml(
                np.zeros((3, 2), dtype=np.float32),
                n_neighbors=2,
                metric="euclidean",
                random_state=None,
                verbose=False,
            )

        self.assertEqual(result, "graph")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("random_state", calls[0][1])
        self.assertEqual(calls[0][1]["n_neighbors"], 2)
        self.assertEqual(calls[0][1]["metric"], "euclidean")
        self.assertFalse(calls[0][1]["verbose"])

    def test_cuml_graph_adapter_moves_only_public_fallback_input_to_gpu(self):
        calls = []
        X = np.zeros((3, 2), dtype=np.float32)
        X_device = object()

        def fake_fuzzy_simplicial_set(received, **kwargs):
            calls.append((received, kwargs))
            return "graph"

        to_device = Mock(return_value=X_device)
        with self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set), patch.dict(
            build_gpu_graph_cuml.__globals__,
            {"_as_cupy_float32": to_device},
        ):
            result = build_gpu_graph_cuml(
                X,
                n_neighbors=2,
                metric="euclidean",
                random_state=None,
                verbose=False,
            )

        self.assertEqual(result, "graph")
        to_device.assert_called_once_with(X)
        self.assertIs(calls[0][0], X_device)

    def test_cuml_graph_adapter_forwards_fuzzy_graph_options(self):
        calls = []

        def fake_fuzzy_simplicial_set(X, **kwargs):
            calls.append((X, kwargs))
            return "graph"

        with self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set):
            build_gpu_graph_cuml(
                np.zeros((3, 2), dtype=np.float32),
                n_neighbors=2,
                metric="minkowski",
                metric_kwds={"p": 1.5},
                random_state=None,
                verbose=True,
                set_op_mix_ratio=0.75,
                local_connectivity=2.0,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["metric_kwds"], {"p": 1.5})
        self.assertEqual(calls[0][1]["set_op_mix_ratio"], 0.75)
        self.assertEqual(calls[0][1]["local_connectivity"], 2.0)

    def test_cuml_graph_adapter_passes_integer_seed_for_seeded_input(self):
        calls = []

        def fake_fuzzy_simplicial_set(X, **kwargs):
            calls.append((X, kwargs))
            return "graph"

        with self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set):
            build_gpu_graph_cuml(
                np.zeros((3, 2), dtype=np.float32),
                n_neighbors=2,
                metric="euclidean",
                random_state=42,
                verbose=False,
            )

        self.assertEqual(len(calls), 1)
        self.assertIn("random_state", calls[0][1])
        self.assertIsInstance(calls[0][1]["random_state"], Integral)

    def test_cuml_graph_adapter_uses_optional_nn_descent_ext_for_large_unseeded(self):
        calls = []

        def fake_fuzzy_simplicial_set(X, **kwargs):
            raise AssertionError("public cuML fuzzy helper should not be called")

        graph_ext = ModuleType("ibumap.graph._cuml_graph_ext")

        def fake_nn_descent_graph(X, **kwargs):
            calls.append((X, kwargs))
            return "nn_descent_graph"

        graph_ext.fuzzy_simplicial_set_nn_descent = fake_nn_descent_graph

        with (
            self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set),
            patch.dict(
                sys.modules,
                {"ibumap.graph._cuml_graph_ext": graph_ext},
            ),
            patch(
                "ibumap.graph.gpu_graph_cuml._as_cupy_float32",
                side_effect=AssertionError(
                    "NN-Descent input must remain on the host"
                ),
            ),
        ):
            X = np.zeros((50001, 2), dtype=np.float32)
            result = build_gpu_graph_cuml(
                X,
                n_neighbors=15,
                metric="euclidean",
                metric_kwds={},
                random_state=None,
                verbose=False,
                set_op_mix_ratio=1.0,
                local_connectivity=1.0,
            )

        self.assertEqual(result, "nn_descent_graph")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], X)
        self.assertIn("random_state", calls[0][1])
        self.assertIsNone(calls[0][1]["random_state"])
        self.assertEqual(calls[0][1]["n_neighbors"], 15)
        self.assertEqual(calls[0][1]["metric"], "euclidean")

    def test_cuml_graph_adapter_retries_unseeded_cuml_overflow_without_seed(self):
        calls = []

        def fake_fuzzy_simplicial_set(X, **kwargs):
            calls.append((X, kwargs))
            if len(calls) == 1:
                raise OverflowError("can't convert negative value to uint64_t")
            return "graph"

        with self._fake_cuml_fuzzy_graph(fake_fuzzy_simplicial_set):
            result = build_gpu_graph_cuml(
                np.zeros((3, 2), dtype=np.float32),
                n_neighbors=2,
                metric="euclidean",
                random_state=None,
                verbose=False,
            )

        self.assertEqual(result, "graph")
        self.assertEqual(len(calls), 2)
        for _, kwargs in calls:
            self.assertNotIn("random_state", kwargs)

    def test_root_seed_derives_stable_independent_streams(self):
        streams = ("graph", "init", "sampling", "noise")
        first = [derive_random_seed(42, stream) for stream in streams]
        second = [derive_random_seed(42, stream) for stream in streams]

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(streams))

    def test_derivation_does_not_mutate_random_state_object(self):
        random_state = np.random.RandomState(19)
        expected = random_state.get_state()

        derive_random_seed(random_state, "sampling")

        actual = random_state.get_state()
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2:], actual[2:])

    def test_explicit_noise_seed_overrides_root_noise_stream(self):
        self.assertEqual(
            _resolve_noise_seed({"noise_seed": 7, "noise_random_state": 11}),
            7,
        )
        self.assertEqual(
            _resolve_noise_seed({"noise_seed": None, "noise_random_state": 11}),
            11,
        )

    def test_cpu_ibfft_sampling_accepts_a_local_rng(self):
        ibfft._clear_fft_kernel_cache()
        embedding = np.random.default_rng(5).uniform(
            0.0, 10.0, size=(96, 2)
        ).astype(np.float32)
        probabilities = np.linspace(0.15, 0.85, embedding.shape[0])
        kwargs = {
            "n_interpolation_points": 1,
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
            "umap_epsilon": 0.001,
            "ibfft_kernel_clip": 4.0,
        }

        first = ibfft.ibFFT_repulsive_sampling(
            embedding, random_state=np.random.RandomState(123), **kwargs
        )
        second = ibfft.ibFFT_repulsive_sampling(
            embedding, random_state=np.random.RandomState(123), **kwargs
        )
        different = ibfft.ibFFT_repulsive_sampling(
            embedding, random_state=np.random.RandomState(124), **kwargs
        )

        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.max(np.abs(first - different))), 0.0)

    def _fake_cuml_fuzzy_graph(self, fake_fuzzy_simplicial_set):
        cupy = ModuleType("cupy")
        cupy.float32 = np.float32
        cupy.asarray = lambda value, dtype=None: np.asarray(value, dtype=dtype)
        cuml = ModuleType("cuml")
        manifold = ModuleType("cuml.manifold")
        umap = ModuleType("cuml.manifold.umap")
        umap.fuzzy_simplicial_set = fake_fuzzy_simplicial_set
        cuml.manifold = manifold
        manifold.umap = umap
        return patch.dict(
            sys.modules,
            {
                "cupy": cupy,
                "cuml": cuml,
                "cuml.manifold": manifold,
                "cuml.manifold.umap": umap,
            },
        )


if __name__ == "__main__":
    unittest.main()
