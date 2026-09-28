import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
from ibumap.graph import build_cpu_graph_from_knn


N_SAMPLES = 64
N_FEATURES = 6
N_NEIGHBORS = 8
N_EPOCHS = 5


def _fixed_inputs():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="euclidean")
    neighbors.fit(X)
    knn_distances, knn_indices = neighbors.kneighbors(X)
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
    init_embedding = rng.uniform(-5.0, 5.0, size=(N_SAMPLES, 2)).astype(np.float32)
    return X, graph, init_embedding


def _model(mode="asynchronous", **kwargs):
    return IBUMAP(
        algorithm="umap",
        device="cpu",
        umap_sgd_update_mode=mode,
        n_neighbors=N_NEIGHBORS,
        n_epochs=N_EPOCHS,
        random_state=42,
        deterministic=True,
        **kwargs,
    )


class SynchronousUMAPTest(unittest.TestCase):
    def assert_embedding_ok(self, embedding):
        self.assertEqual(embedding.shape, (N_SAMPLES, 2))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())
        self.assertTrue(np.all(np.var(embedding, axis=0) > 1e-6))

    def test_default_mode_routes_to_asynchronous_optimizer(self):
        X, graph, init_embedding = _fixed_inputs()
        runtime_cpu_pipeline = importlib.import_module(
            "ibumap.pipeline.cpu_pipeline"
        )
        model = IBUMAP(
            algorithm="umap",
            device="cpu",
            n_neighbors=N_NEIGHBORS,
            n_epochs=N_EPOCHS,
            random_state=42,
            deterministic=True,
        )

        with patch.object(
            runtime_cpu_pipeline,
            "umap_optimize_layout_euclidean",
            wraps=runtime_cpu_pipeline.umap_optimize_layout_euclidean,
        ) as asynchronous_optimizer, patch.object(
            runtime_cpu_pipeline,
            "umap_optimize_layout_euclidean_synchronous",
            side_effect=AssertionError("synchronous optimizer should not be used"),
        ):
            embedding = model.optimize_from_graph(graph, init_embedding)

        asynchronous_optimizer.assert_called_once()
        optimizer_args = asynchronous_optimizer.call_args.args
        self.assertEqual(optimizer_args[7].dtype, np.float64)
        self.assertEqual(optimizer_args[8].dtype, np.float64)
        self.assert_embedding_ok(embedding)

    def test_asynchronous_and_synchronous_modes_produce_finite_embeddings(self):
        X, graph, init_embedding = _fixed_inputs()

        asynchronous = _model("asynchronous").optimize_from_graph(graph, init_embedding)
        synchronous = _model("synchronous").optimize_from_graph(graph, init_embedding)

        self.assert_embedding_ok(asynchronous)
        self.assert_embedding_ok(synchronous)

    def test_synchronous_repulsion_clip_uses_epoch_total_norm(self):
        X, graph, init_embedding = _fixed_inputs()
        runtime_optimizer = importlib.import_module(
            "ibumap.optimizers.umap_optimizer"
        )
        original_clip = runtime_optimizer._clip_force_norm_inplace
        observations = []

        def record_clip(force, clip_norm, is_gpu):
            pre_clip_norms = np.linalg.norm(force, axis=1)
            result = original_clip(force, clip_norm, is_gpu)
            observations.append(
                (
                    float(clip_norm),
                    pre_clip_norms,
                    np.linalg.norm(force, axis=1),
                )
            )
            return result

        model = _model(
            "synchronous",
            repulsion_clip_norm=0.25,
            repulsion_clip_with_alpha=True,
            repulsion_clip_epoch_range=(1, 4),
        )
        with patch.object(
            runtime_optimizer,
            "_clip_force_norm_inplace",
            side_effect=record_clip,
        ):
            embedding = model.optimize_from_graph(graph, init_embedding)

        self.assert_embedding_ok(embedding)
        self.assertEqual(len(observations), 3)
        np.testing.assert_allclose(
            [observation[0] for observation in observations],
            [0.25, 0.20, 0.15],
        )
        self.assertTrue(
            any(
                np.any(pre_clip_norms > clip_norm)
                for clip_norm, pre_clip_norms, _ in observations
            )
        )
        for clip_norm, _, post_clip_norms in observations:
            self.assertLessEqual(float(post_clip_norms.max()), clip_norm + 1e-6)


if __name__ == "__main__":
    unittest.main()
