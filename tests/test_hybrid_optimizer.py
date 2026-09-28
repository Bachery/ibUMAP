import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from ibumap import (
    ConstraintConfig,
    HybridOptimizerConfig,
    IBUMAPConfig,
    IBUMAP,
)


def _ring_inputs(n_samples=16):
    rows = np.arange(n_samples, dtype=np.int32)
    cols = (rows + 1) % n_samples
    graph = sparse.csr_matrix(
        (
            np.ones(2 * n_samples, dtype=np.float32),
            (
                np.concatenate([rows, cols]),
                np.concatenate([cols, rows]),
            ),
        ),
        shape=(n_samples, n_samples),
    )
    init = np.random.RandomState(7).normal(size=(n_samples, 2)).astype(np.float32)
    X = np.random.RandomState(8).normal(size=(n_samples, 4)).astype(np.float32)
    return X, graph, init


class HybridConfigTest(unittest.TestCase):
    def test_defaults_and_epoch_split(self):
        config = HybridOptimizerConfig()

        self.assertEqual(config.refinement_fraction, 0.10)
        self.assertEqual(config.refinement_learning_rate, 1.0)
        self.assertEqual(config.refinement_negative_sample_rate, 5.0)
        self.assertEqual(config.split_epochs(200), (180, 20))
        self.assertEqual(config.split_epochs(500), (450, 50))
        self.assertEqual(config.split_epochs(5), (4, 1))

    def test_flat_and_grouped_configuration(self):
        flat = IBUMAP(
            algorithm="hybrid",
            n_epochs=20,
            refinement_fraction=0.25,
            refinement_learning_rate=0.4,
            refinement_negative_sample_rate=3.0,
        )
        self.assertEqual(flat.config.hybrid.refinement_fraction, 0.25)
        self.assertEqual(flat.refinement_learning_rate, 0.4)
        self.assertEqual(flat.refinement_negative_sample_rate, 3.0)

        grouped = IBUMAP(
            algorithm="hybrid",
            n_epochs=20,
            hybrid=HybridOptimizerConfig(refinement_fraction=0.2),
        )
        self.assertEqual(grouped.config.hybrid.refinement_fraction, 0.2)

        with self.assertRaisesRegex(ValueError, "Conflicting"):
            IBUMAP(
                algorithm="hybrid",
                hybrid=HybridOptimizerConfig(refinement_fraction=0.2),
                refinement_fraction=0.3,
            )

    def test_validation(self):
        invalid_configs = (
            {"refinement_fraction": 0.0},
            {"refinement_fraction": 1.0},
            {"refinement_learning_rate": 0.0},
            {"refinement_negative_sample_rate": -1.0},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    HybridOptimizerConfig(**kwargs)

        with self.assertRaisesRegex(ValueError, "n_epochs >= 2"):
            IBUMAP(algorithm="hybrid", n_epochs=1)
        with self.assertRaisesRegex(NotImplementedError, "device='cpu' only"):
            IBUMAP(algorithm="hybrid", device="cuda")
        with self.assertRaisesRegex(ValueError, "n_components=2"):
            IBUMAP(algorithm="hybrid", n_components=3)

    def test_non_default_mechanisms_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "experimental mechanisms"):
            IBUMAP(algorithm="hybrid", local_exact_repulsion=True)
        with self.assertRaisesRegex(ValueError, "experimental mechanisms"):
            IBUMAP(algorithm="hybrid", attraction_schedule_mode="auto")
        with self.assertRaisesRegex(ValueError, "constraints"):
            IBUMAP(
                algorithm="hybrid",
                constraints=ConstraintConfig(
                    known_points_indices=[0],
                    known_points_positions=np.zeros((1, 2), dtype=np.float32),
                ),
            )
        with self.assertRaisesRegex(ValueError, "attraction_mode='sampling'"):
            IBUMAP(
                algorithm="hybrid",
                ibumap=IBUMAPConfig(attraction_mode="true_loss"),
            )


class HybridOptimizerTest(unittest.TestCase):
    def _mocked_run(self, *, legacy=False):
        X, graph, init = _ring_inputs()
        schedule_epochs = []
        schedule_dtypes = []

        from ibumap.pipeline import cpu_pipeline

        real_make_epochs_per_sample = cpu_pipeline.make_epochs_per_sample

        def record_schedule(weights, n_epochs, dtype=None):
            schedule_epochs.append(int(n_epochs))
            schedule = real_make_epochs_per_sample(weights, n_epochs, dtype=dtype)
            schedule_dtypes.append(schedule.dtype)
            return schedule

        def fake_ibumap(embedding, *args):
            self.assertEqual(args[4], 18)
            self.assertEqual(args[9:11], ("sampling", "true_loss"))
            return embedding + np.float32(1.0), {
                "opt_prep_time": 0.1,
                "attr_time": 0.2,
                "repl_time": 0.3,
                "appl_time": 0.4,
            }

        def fake_refinement(embedding, *args, **kwargs):
            self.assertEqual(args[2], 2)
            np.testing.assert_allclose(embedding, init + np.float32(1.0))
            self.assertEqual(kwargs["fft_params"]["umap_initial_alpha"], 0.25)
            negative_schedule = args[7]
            positive_schedule = args[6]
            np.testing.assert_allclose(negative_schedule, positive_schedule / 3.0)
            return embedding + np.float32(2.0), {
                "opt_prep_time": 0.5,
                "optimization_time": 0.6,
                "appl_time": 0.0,
            }

        model = IBUMAP(
            algorithm="hybrid",
            n_epochs=20,
            refinement_fraction=0.1,
            refinement_learning_rate=0.25,
            refinement_negative_sample_rate=3.0,
            random_state=42,
            deterministic=True,
        )
        with (
            patch.object(
                cpu_pipeline,
                "make_epochs_per_sample",
                side_effect=record_schedule,
            ),
            patch.object(
                cpu_pipeline,
                "umap_true_loss_optimization",
                side_effect=fake_ibumap,
            ),
            patch.object(
                cpu_pipeline,
                "umap_optimize_layout_euclidean",
                side_effect=fake_refinement,
            ),
        ):
            if legacy:
                with self.assertWarnsRegex(FutureWarning, "deprecated"):
                    result = model.optimize_from_graph(X, graph, init)
            else:
                result = model.optimize_from_graph(graph, init)

        np.testing.assert_allclose(result, init + np.float32(3.0))
        np.testing.assert_array_equal(init, _ring_inputs()[2])
        self.assertEqual(schedule_epochs, [18, 2])
        self.assertEqual(
            schedule_dtypes,
            [np.dtype(np.float64), np.dtype(np.float64)],
        )
        return model, result

    def test_stage_handoff_and_independent_schedules(self):
        model, _ = self._mocked_run()
        costs = model.get_time_costs()

        self.assertEqual(costs["hybrid_total_epochs"], 20)
        self.assertEqual(costs["hybrid_ibumap_epochs"], 18)
        self.assertEqual(costs["hybrid_refinement_epochs"], 2)
        self.assertEqual(costs["hybrid_ibumap_attr_time"], 0.2)
        self.assertEqual(costs["hybrid_refinement_optimization_time"], 0.6)
        self.assertAlmostEqual(costs["opt_prep_time"], 0.6)
        self.assertAlmostEqual(costs["appl_time"], 0.4)

    def test_legacy_three_argument_entrypoint(self):
        self._mocked_run(legacy=True)

    def test_graph_only_run_cannot_update(self):
        model, _ = self._mocked_run()
        with self.assertRaisesRegex(NotImplementedError, "only optimize_from_graph"):
            model.update_embedding()

    def test_other_entrypoints_are_rejected(self):
        X, graph, init = _ring_inputs()
        model = IBUMAP(algorithm="hybrid", n_epochs=2)
        with self.assertRaisesRegex(NotImplementedError, "only optimize_from_graph"):
            model.fit(X)
        with self.assertRaisesRegex(NotImplementedError, "only optimize_from_graph"):
            model.fit_transform_from_knn(
                X,
                np.zeros((X.shape[0], 2), dtype=np.int64),
                np.zeros((X.shape[0], 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(NotImplementedError, "only optimize_from_graph"):
            model.optimize_from_prepared_graph(X, graph, init)

    def test_numeric_smoke_and_repeatability(self):
        _, graph, init = _ring_inputs(n_samples=12)

        def run():
            return IBUMAP(
                algorithm="hybrid",
                n_epochs=2,
                random_state=42,
                deterministic=True,
            ).optimize_from_graph(graph, init)

        first = run()
        second = run()
        self.assertEqual(first.shape, init.shape)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(init, _ring_inputs(n_samples=12)[2])

    def test_effective_config_reports_hybrid_path(self):
        model, _ = self._mocked_run()
        report = model.explain_effective_config()
        self.assertEqual(report["canonical_algorithm"], "hybrid")
        self.assertEqual(report["execution_path"], "hybrid_cpu")
        self.assertEqual(report["config"]["hybrid"]["refinement_fraction"], 0.1)


if __name__ == "__main__":
    unittest.main()
