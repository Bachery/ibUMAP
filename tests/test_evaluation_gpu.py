from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.evaluation import (  # noqa: E402
    compute_geodesic_source_state,
    continuity,
    compute_persistence_diagram,
    evaluate_embedding_from_states,
    geodesic_distance_correlation,
    neighborhood_preservation,
    prepare_embedding_state,
    prepare_source_state,
    persistent_homology_distance,
    pseudotime_correlation,
    trustworthiness,
)
from ibumap.evaluation._device import gpu_available  # noqa: E402


def _gpu_available() -> bool:
    available, _ = gpu_available()
    return available


@unittest.skipUnless(_gpu_available(), "CuPy/CUDA unavailable")
class EvaluationGpuTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.high_dimensional = rng.normal(size=(40, 6)).astype(np.float32)
        self.embedding = (
            self.high_dimensional[:, :2] + 0.05 * rng.normal(size=(40, 2))
        ).astype(np.float32)

    def test_rank_metrics_match_cpu(self):
        cpu_trust = trustworthiness(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            low_memory=True,
            low_memory_batch_size=9,
        )
        gpu_trust = trustworthiness(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            low_memory=True,
            low_memory_batch_size=9,
            dtype=np.float64,
            device="gpu",
            gpu_batch_size=9,
        )
        cpu_cont = continuity(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            low_memory=True,
            low_memory_batch_size=9,
        )
        gpu_cont = continuity(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            low_memory=True,
            low_memory_batch_size=9,
            dtype=np.float64,
            device="gpu",
            gpu_batch_size=9,
        )

        self.assertAlmostEqual(gpu_trust, cpu_trust, places=6)
        self.assertAlmostEqual(gpu_cont, cpu_cont, places=6)

    def test_neighborhood_preservation_matches_cpu(self):
        cpu = neighborhood_preservation(
            self.high_dimensional,
            self.embedding,
            n_neighbors=6,
        )
        gpu = neighborhood_preservation(
            self.high_dimensional,
            self.embedding,
            n_neighbors=6,
            device="gpu",
            gpu_batch_size=10,
        )

        self.assertAlmostEqual(gpu.score, cpu.score, places=6)
        self.assertEqual(gpu.local_scores.shape, cpu.local_scores.shape)
        np.testing.assert_allclose(gpu.local_scores, cpu.local_scores, rtol=1e-5, atol=1e-6)

    def test_geodesic_landmark_matches_cpu(self):
        cpu = geodesic_distance_correlation(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            mode="landmark",
            n_landmarks=8,
            random_state=7,
        )
        gpu = geodesic_distance_correlation(
            self.high_dimensional,
            self.embedding,
            n_neighbors=5,
            mode="landmark",
            n_landmarks=8,
            random_state=7,
            device="gpu",
            gpu_batch_size=10,
        )

        self.assertAlmostEqual(gpu.spearman_correlation, cpu.spearman_correlation, places=6)
        self.assertEqual(gpu.n_pairs, cpu.n_pairs)
        self.assertEqual(gpu.mode, "landmark")
        self.assertEqual(gpu.n_landmarks, 8)

    def test_geodesic_gpu_rejects_unsupported_modes(self):
        for mode in ("full", "exact_low_memory", "approx_knn"):
            with self.subTest(mode=mode):
                with self.assertRaises(NotImplementedError):
                    geodesic_distance_correlation(
                        self.high_dimensional,
                        self.embedding,
                        n_neighbors=5,
                        mode=mode,
                        n_landmarks=8,
                        random_state=7,
                        device="gpu",
                    )

    def test_geodesic_source_state_gpu_rejects_unsupported_modes(self):
        for mode in ("full", "exact_low_memory", "approx_knn"):
            with self.subTest(mode=mode):
                with self.assertRaises(NotImplementedError):
                    compute_geodesic_source_state(
                        self.high_dimensional,
                        n_neighbors=5,
                        mode=mode,
                        n_landmarks=8,
                        random_state=7,
                        device="gpu",
                    )

    def test_precomputed_state_api_uses_gpu_paths(self):
        cpu_source = prepare_source_state(
            self.high_dimensional,
            n_neighbors=5,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            geodesic_n_landmarks=8,
            geodesic_random_state=7,
            compute_persistence_state=True,
            persistence_max_points=12,
            dtype=np.float64,
            device="cpu",
        )
        cpu_embedding = prepare_embedding_state(
            self.embedding,
            cpu_source,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            compute_persistence_state=True,
            persistence_max_points=12,
            dtype=np.float64,
            device="cpu",
        )
        gpu_source = prepare_source_state(
            self.high_dimensional,
            n_neighbors=5,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            geodesic_n_landmarks=8,
            geodesic_random_state=7,
            compute_persistence_state=True,
            persistence_max_points=12,
            dtype=np.float64,
            device="gpu",
            gpu_batch_size=10,
        )
        gpu_embedding = prepare_embedding_state(
            self.embedding,
            gpu_source,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            compute_persistence_state=True,
            persistence_max_points=12,
            dtype=np.float64,
            device="gpu",
            gpu_batch_size=10,
        )

        cpu_scores = evaluate_embedding_from_states(
            cpu_source,
            cpu_embedding,
            metrics=("trustworthiness", "continuity", "neighborhood", "geodesic", "persistent_homology"),
            geodesic_random_state=7,
            dtype=np.float64,
            device="cpu",
        )
        gpu_scores = evaluate_embedding_from_states(
            gpu_source,
            gpu_embedding,
            metrics=("trustworthiness", "continuity", "neighborhood", "geodesic", "persistent_homology"),
            geodesic_random_state=7,
            dtype=np.float64,
            device="gpu",
            gpu_batch_size=10,
        )

        self.assertAlmostEqual(gpu_scores["trustworthiness"], cpu_scores["trustworthiness"], places=6)
        self.assertAlmostEqual(gpu_scores["continuity"], cpu_scores["continuity"], places=6)
        self.assertAlmostEqual(
            gpu_scores["neighborhood_preservation"].score,
            cpu_scores["neighborhood_preservation"].score,
            places=6,
        )
        self.assertAlmostEqual(
            gpu_scores["geodesic"].spearman_correlation,
            cpu_scores["geodesic"].spearman_correlation,
            places=6,
        )
        self.assertAlmostEqual(
            gpu_scores["persistent_homology"].wasserstein_total,
            cpu_scores["persistent_homology"].wasserstein_total,
            places=6,
        )

    def test_pseudotime_matches_cpu(self):
        pseudotime = np.linspace(0.0, 1.0, self.embedding.shape[0], dtype=np.float32)
        cpu = pseudotime_correlation(self.embedding, pseudotime)
        gpu = pseudotime_correlation(self.embedding, pseudotime, device="gpu")

        self.assertAlmostEqual(gpu.correlation, cpu.correlation, places=6)
        np.testing.assert_allclose(gpu.arc_length, cpu.arc_length, rtol=1e-5, atol=1e-6)

    def test_persistence_homology_hybrid_gpu_path(self):
        diagram = compute_persistence_diagram(
            self.embedding,
            max_points=12,
            dtype=np.float64,
            device="gpu",
        )
        comparison = persistent_homology_distance(
            self.high_dimensional,
            self.embedding,
            max_points=12,
            dtype=np.float64,
            device="gpu",
        )

        self.assertEqual(diagram.h0.dtype, np.dtype(np.float64))
        self.assertEqual(diagram.sampled_indices.shape, (12,))
        self.assertEqual(comparison.source_diagram.sampled_indices.shape, (12,))
        self.assertTrue(np.isfinite(comparison.wasserstein_total))


if __name__ == "__main__":
    unittest.main()
