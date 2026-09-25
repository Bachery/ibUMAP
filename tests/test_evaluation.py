from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.evaluation import (
    betti_numbers_at_scale,
    compute_geodesic_embedding_state,
    compute_geodesic_source_state,
    compute_distance_spearman_source_state,
    compute_neighbors,
    compute_random_triplet_source_state,
    continuity,
    continuity_from_source_neighbors,
    distance_spearman_correlation,
    distance_spearman_correlation_from_state,
    evaluate_embedding_from_states,
    compute_persistence_diagram,
    geodesic_distance_correlation,
    geodesic_distance_correlation_from_state,
    neighborhood_preservation,
    neighborhood_preservation_from_neighbors,
    prepare_embedding_state,
    prepare_source_state,
    persistent_homology_distance,
    persistent_homology_distance_from_diagrams,
    pseudotime_correlation,
    random_triplet_accuracy,
    random_triplet_accuracy_from_state,
    trustworthiness,
    trustworthiness_from_embedding_neighbors,
)


def _pynndescent_available() -> bool:
    try:
        from pynndescent import NNDescent  # noqa: F401
    except Exception:
        return False
    return True


class EvaluationMetricsTest(unittest.TestCase):
    def test_global_structure_metrics_prefer_distance_preserving_embedding(self):
        rng = np.random.default_rng(5)
        source = rng.normal(size=(48, 3))
        good_embedding = source.copy()
        bad_embedding = source[rng.permutation(source.shape[0])]

        rta_good = random_triplet_accuracy(source, good_embedding, n_triplets=500, random_state=17)
        rta_bad = random_triplet_accuracy(source, bad_embedding, n_triplets=500, random_state=17)
        spearman_good = distance_spearman_correlation(source, good_embedding, n_pairs=500, random_state=19)
        spearman_bad = distance_spearman_correlation(source, bad_embedding, n_pairs=500, random_state=19)

        self.assertAlmostEqual(rta_good.score, 1.0)
        self.assertGreater(rta_good.score, rta_bad.score)
        self.assertAlmostEqual(spearman_good.spearman_correlation, 1.0)
        self.assertGreater(spearman_good.spearman_correlation, spearman_bad.spearman_correlation)

    def test_global_structure_source_states_are_reproducible_and_reusable(self):
        rng = np.random.default_rng(11)
        source = rng.normal(size=(36, 4))
        embedding = source[:, :2] + 0.03 * rng.normal(size=(36, 2))

        rta_state = compute_random_triplet_source_state(source, n_triplets=300, random_state=23)
        rta_state_again = compute_random_triplet_source_state(source, n_triplets=300, random_state=23)
        np.testing.assert_array_equal(rta_state.anchor_indices, rta_state_again.anchor_indices)
        np.testing.assert_array_equal(rta_state.first_indices, rta_state_again.first_indices)
        np.testing.assert_array_equal(rta_state.second_indices, rta_state_again.second_indices)
        self.assertAlmostEqual(
            random_triplet_accuracy_from_state(rta_state, embedding).score,
            random_triplet_accuracy(source, embedding, n_triplets=300, random_state=23).score,
        )

        spearman_state = compute_distance_spearman_source_state(source, n_pairs=300, random_state=29)
        spearman_state_again = compute_distance_spearman_source_state(source, n_pairs=300, random_state=29)
        np.testing.assert_array_equal(spearman_state.first_indices, spearman_state_again.first_indices)
        np.testing.assert_array_equal(spearman_state.second_indices, spearman_state_again.second_indices)
        self.assertAlmostEqual(
            distance_spearman_correlation_from_state(spearman_state, embedding).spearman_correlation,
            distance_spearman_correlation(source, embedding, n_pairs=300, random_state=29).spearman_correlation,
        )

    def test_neighborhood_preservation_matches_dense_reference(self):
        rng = np.random.default_rng(7)
        high_dimensional = rng.normal(size=(36, 5))
        embedding = high_dimensional[:, :2] + 0.05 * rng.normal(size=(36, 2))
        n_neighbors = 6

        source_distances = pairwise_distances(high_dimensional)
        embedded_distances = pairwise_distances(embedding)
        np.fill_diagonal(source_distances, np.inf)
        np.fill_diagonal(embedded_distances, np.inf)
        source_neighbors = np.argsort(source_distances, axis=1)[:, :n_neighbors]
        embedded_neighbors = np.argsort(embedded_distances, axis=1)[:, :n_neighbors]

        expected_local_scores = np.asarray(
            [
                len(set(source_row).intersection(embedded_row)) / n_neighbors
                for source_row, embedded_row in zip(source_neighbors, embedded_neighbors)
            ],
            dtype=np.float64,
        )

        actual = neighborhood_preservation(
            high_dimensional,
            embedding,
            n_neighbors=n_neighbors,
        )

        np.testing.assert_allclose(actual.local_scores, expected_local_scores)
        self.assertAlmostEqual(actual.score, float(np.mean(expected_local_scores)))

    def test_precomputed_neighbors_match_neighbor_metrics(self):
        rng = np.random.default_rng(17)
        high_dimensional = rng.normal(size=(44, 6))
        embedding = high_dimensional[:, :2] + 0.07 * rng.normal(size=(44, 2))
        n_neighbors = 5

        source_neighbors = compute_neighbors(
            high_dimensional,
            n_neighbors=n_neighbors,
        )
        embedding_neighbors = compute_neighbors(
            embedding,
            n_neighbors=n_neighbors,
        )

        direct_trust = trustworthiness(high_dimensional, embedding, n_neighbors=n_neighbors)
        cached_trust = trustworthiness_from_embedding_neighbors(
            high_dimensional,
            embedding_neighbors.indices,
            n_neighbors=n_neighbors,
        )
        direct_continuity = continuity(high_dimensional, embedding, n_neighbors=n_neighbors)
        cached_continuity = continuity_from_source_neighbors(
            embedding,
            source_neighbors.indices,
            n_neighbors=n_neighbors,
        )
        direct_overlap = neighborhood_preservation(
            high_dimensional,
            embedding,
            n_neighbors=n_neighbors,
        )
        cached_overlap = neighborhood_preservation_from_neighbors(
            source_neighbors.indices,
            embedding_neighbors.indices,
        )

        self.assertAlmostEqual(cached_trust, direct_trust)
        self.assertAlmostEqual(cached_continuity, direct_continuity)
        self.assertAlmostEqual(cached_overlap.score, direct_overlap.score)
        np.testing.assert_allclose(cached_overlap.local_scores, direct_overlap.local_scores)

    def test_geodesic_distance_correlation_prefers_structure_preserving_embedding(self):
        parameter = np.linspace(-1.0, 1.0, 32)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        good_embedding = np.column_stack((parameter, np.zeros_like(parameter)))
        bad_embedding = np.column_stack((np.abs(parameter), np.zeros_like(parameter)))

        good = geodesic_distance_correlation(high_dimensional, good_embedding, n_neighbors=4)
        bad = geodesic_distance_correlation(high_dimensional, bad_embedding, n_neighbors=4)

        self.assertGreater(good.spearman_correlation, 0.95)
        self.assertLess(bad.spearman_correlation, 0.6)
        self.assertLess(bad.spearman_correlation, good.spearman_correlation)
        self.assertEqual(good.mode, "landmark")
        self.assertEqual(good.n_landmarks, high_dimensional.shape[0])

    def test_geodesic_exact_low_memory_matches_full_mode(self):
        parameter = np.linspace(-1.0, 1.0, 40)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        embedding = np.column_stack((parameter, np.zeros_like(parameter)))

        full = geodesic_distance_correlation(
            high_dimensional,
            embedding,
            n_neighbors=4,
            mode="full",
        )
        low_memory = geodesic_distance_correlation(
            high_dimensional,
            embedding,
            n_neighbors=4,
            mode="exact_low_memory",
            shortest_path_batch_size=7,
        )

        self.assertAlmostEqual(low_memory.spearman_correlation, full.spearman_correlation)
        self.assertEqual(low_memory.n_pairs, full.n_pairs)
        self.assertEqual(low_memory.mode, "exact_low_memory")

    def test_geodesic_landmark_mode_prefers_structure_preserving_embedding(self):
        parameter = np.linspace(-1.0, 1.0, 40)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        good_embedding = np.column_stack((parameter, np.zeros_like(parameter)))
        bad_embedding = np.column_stack((np.abs(parameter), np.zeros_like(parameter)))

        good = geodesic_distance_correlation(
            high_dimensional,
            good_embedding,
            n_neighbors=4,
            mode="landmark",
            n_landmarks=8,
            random_state=0,
        )
        bad = geodesic_distance_correlation(
            high_dimensional,
            bad_embedding,
            n_neighbors=4,
            mode="landmark",
            n_landmarks=8,
            random_state=0,
        )

        self.assertGreater(good.spearman_correlation, 0.9)
        self.assertLess(bad.spearman_correlation, good.spearman_correlation)
        self.assertEqual(good.n_landmarks, 8)

    def test_geodesic_precomputed_state_matches_direct_landmark_mode(self):
        parameter = np.linspace(-1.0, 1.0, 40)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        embedding = np.column_stack((parameter, np.zeros_like(parameter)))

        source_state = compute_geodesic_source_state(
            high_dimensional,
            n_neighbors=4,
            mode="landmark",
            n_landmarks=8,
            random_state=0,
        )
        embedding_state = compute_geodesic_embedding_state(embedding, source_state)
        cached = geodesic_distance_correlation_from_state(
            source_state,
            embedding,
            embedding_state=embedding_state,
        )
        direct = geodesic_distance_correlation(
            high_dimensional,
            embedding,
            n_neighbors=4,
            mode="landmark",
            n_landmarks=8,
            random_state=0,
        )

        self.assertAlmostEqual(cached.spearman_correlation, direct.spearman_correlation)
        self.assertEqual(cached.n_pairs, direct.n_pairs)
        self.assertEqual(cached.n_landmarks, direct.n_landmarks)

    @unittest.skipUnless(_pynndescent_available(), "pynndescent is unavailable")
    def test_geodesic_approx_knn_mode_runs(self):
        parameter = np.linspace(-1.0, 1.0, 32)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        embedding = np.column_stack((parameter, np.zeros_like(parameter)))

        result = geodesic_distance_correlation(
            high_dimensional,
            embedding,
            n_neighbors=4,
            mode="approx_knn",
            n_landmarks=6,
            random_state=0,
            approx_knn_n_jobs=1,
        )

        self.assertGreater(result.spearman_correlation, 0.8)
        self.assertEqual(result.mode, "approx_knn")
        self.assertEqual(result.n_landmarks, 6)

    def test_trustworthiness_matches_sklearn_reference(self):
        parameter = np.linspace(-1.0, 1.0, 24)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        embedding = np.column_stack((parameter, np.zeros_like(parameter)))

        expected = sklearn_trustworthiness(high_dimensional, embedding, n_neighbors=4)
        actual = trustworthiness(high_dimensional, embedding, n_neighbors=4)

        self.assertAlmostEqual(actual, expected, places=12)

    def test_persistent_homology_precomputed_diagrams_match_direct_metric(self):
        rng = np.random.default_rng(23)
        high_dimensional = rng.normal(size=(30, 4))
        embedding = high_dimensional[:, :2] + 0.05 * rng.normal(size=(30, 2))

        source_diagram = compute_persistence_diagram(
            high_dimensional,
            max_points=12,
            normalize=True,
        )
        embedding_diagram = compute_persistence_diagram(
            embedding,
            max_points=12,
            sampled_indices=source_diagram.sampled_indices,
            normalize=True,
        )
        cached = persistent_homology_distance_from_diagrams(source_diagram, embedding_diagram)
        direct = persistent_homology_distance(
            high_dimensional,
            embedding,
            max_points=12,
            normalize=True,
        )

        self.assertAlmostEqual(cached.wasserstein_h0, direct.wasserstein_h0)
        self.assertAlmostEqual(cached.wasserstein_h1, direct.wasserstein_h1)
        self.assertAlmostEqual(cached.wasserstein_total, direct.wasserstein_total)

    def test_precomputed_evaluation_state_matches_direct_metrics(self):
        rng = np.random.default_rng(29)
        high_dimensional = rng.normal(size=(38, 5))
        embedding = high_dimensional[:, :2] + 0.05 * rng.normal(size=(38, 2))
        n_neighbors = 5

        source_state = prepare_source_state(
            high_dimensional,
            n_neighbors=n_neighbors,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            geodesic_n_landmarks=7,
            geodesic_random_state=0,
            compute_persistence_state=True,
            persistence_max_points=12,
            compute_rta_state=True,
            rta_n_triplets=300,
            rta_random_state=3,
            compute_distance_spearman_state=True,
            distance_spearman_n_pairs=300,
            distance_spearman_random_state=5,
        )
        embedding_state = prepare_embedding_state(
            embedding,
            source_state,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            compute_persistence_state=True,
            persistence_max_points=12,
        )

        scores = evaluate_embedding_from_states(
            source_state,
            embedding_state,
            metrics=(
                "trustworthiness",
                "continuity",
                "neighborhood",
                "geodesic",
                "persistent_homology",
                "rta",
                "distance_spearman",
            ),
        )

        self.assertAlmostEqual(
            scores["trustworthiness"],
            trustworthiness(high_dimensional, embedding, n_neighbors=n_neighbors),
        )
        self.assertAlmostEqual(
            scores["continuity"],
            continuity(high_dimensional, embedding, n_neighbors=n_neighbors),
        )
        self.assertAlmostEqual(
            scores["neighborhood_preservation"].score,
            neighborhood_preservation(high_dimensional, embedding, n_neighbors=n_neighbors).score,
        )
        self.assertAlmostEqual(
            scores["geodesic"].spearman_correlation,
            geodesic_distance_correlation(
                high_dimensional,
                embedding,
                n_neighbors=n_neighbors,
                mode="landmark",
                n_landmarks=7,
                random_state=0,
            ).spearman_correlation,
        )
        direct_persistence = persistent_homology_distance(
            high_dimensional,
            embedding,
            max_points=12,
            normalize=True,
        )
        self.assertAlmostEqual(
            scores["persistent_homology"].wasserstein_total,
            direct_persistence.wasserstein_total,
        )
        self.assertAlmostEqual(
            scores["rta"].score,
            random_triplet_accuracy(
                high_dimensional,
                embedding,
                n_triplets=300,
                random_state=3,
            ).score,
        )
        self.assertAlmostEqual(
            scores["distance_spearman"].spearman_correlation,
            distance_spearman_correlation(
                high_dimensional,
                embedding,
                n_pairs=300,
                random_state=5,
            ).spearman_correlation,
        )

    def test_precomputed_evaluation_records_optional_component_timings(self):
        rng = np.random.default_rng(31)
        high_dimensional = rng.normal(size=(34, 5))
        embedding = high_dimensional[:, :2] + 0.04 * rng.normal(size=(34, 2))
        source_timings = {}
        embedding_timings = {}
        score_timings = {}

        source_state = prepare_source_state(
            high_dimensional,
            n_neighbors=5,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            geodesic_n_landmarks=7,
            geodesic_random_state=0,
            compute_persistence_state=True,
            persistence_max_points=12,
            timings=source_timings,
        )
        embedding_state = prepare_embedding_state(
            embedding,
            source_state,
            compute_neighbor_state=True,
            compute_geodesic_state=True,
            compute_persistence_state=True,
            persistence_max_points=12,
            timings=embedding_timings,
        )
        scores = evaluate_embedding_from_states(
            source_state,
            embedding_state,
            metrics=("trustworthiness", "continuity", "neighborhood", "geodesic", "persistent_homology"),
            timings=score_timings,
        )

        self.assertEqual(
            set(source_timings),
            {"neighbors_seconds", "geodesic_seconds", "persistence_seconds"},
        )
        self.assertEqual(
            set(embedding_timings),
            {"neighbors_seconds", "geodesic_seconds", "persistence_seconds"},
        )
        self.assertEqual(
            set(score_timings),
            {
                "trustworthiness_seconds",
                "continuity_seconds",
                "neighborhood_preservation_seconds",
                "geodesic_seconds",
                "persistent_homology_seconds",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in source_timings.values()))
        self.assertTrue(all(value >= 0.0 for value in embedding_timings.values()))
        self.assertTrue(all(value >= 0.0 for value in score_timings.values()))
        self.assertEqual(
            set(scores),
            {"trustworthiness", "continuity", "neighborhood_preservation", "geodesic", "persistent_homology"},
        )

    def test_precomputed_timing_collector_only_contains_requested_components(self):
        rng = np.random.default_rng(37)
        high_dimensional = rng.normal(size=(24, 4))
        embedding = high_dimensional[:, :2]
        source_timings = {}
        embedding_timings = {}
        score_timings = {}

        source_state = prepare_source_state(
            high_dimensional,
            n_neighbors=4,
            compute_neighbor_state=False,
            compute_geodesic_state=False,
            compute_persistence_state=False,
            timings=source_timings,
        )
        embedding_state = prepare_embedding_state(
            embedding,
            source_state,
            compute_neighbor_state=True,
            compute_geodesic_state=False,
            compute_persistence_state=False,
            timings=embedding_timings,
        )
        scores = evaluate_embedding_from_states(
            source_state,
            embedding_state,
            metrics=("trustworthiness",),
            timings=score_timings,
        )

        self.assertEqual(source_timings, {})
        self.assertEqual(set(embedding_timings), {"neighbors_seconds"})
        self.assertEqual(set(score_timings), {"trustworthiness_seconds"})
        self.assertEqual(set(scores), {"trustworthiness"})

    def test_rank_metrics_low_memory_match_dense_mode(self):
        rng = np.random.default_rng(11)
        high_dimensional = rng.normal(size=(42, 7))
        embedding = high_dimensional[:, :2] + 0.1 * rng.normal(size=(42, 2))

        dense_trust = trustworthiness(
            high_dimensional,
            embedding,
            n_neighbors=6,
            low_memory=False,
        )
        low_memory_trust = trustworthiness(
            high_dimensional,
            embedding,
            n_neighbors=6,
            low_memory=True,
            low_memory_batch_size=9,
        )
        dense_continuity = continuity(
            high_dimensional,
            embedding,
            n_neighbors=6,
            low_memory=False,
        )
        low_memory_continuity = continuity(
            high_dimensional,
            embedding,
            n_neighbors=6,
            low_memory=True,
            low_memory_batch_size=9,
        )

        self.assertAlmostEqual(low_memory_trust, dense_trust, places=12)
        self.assertAlmostEqual(low_memory_continuity, dense_continuity, places=12)

    def test_rank_metrics_reject_invalid_low_memory_batch_size(self):
        rng = np.random.default_rng(13)
        high_dimensional = rng.normal(size=(12, 4))
        embedding = rng.normal(size=(12, 2))

        with self.assertRaises(ValueError):
            trustworthiness(
                high_dimensional,
                embedding,
                n_neighbors=3,
                low_memory=True,
                low_memory_batch_size=0,
            )

    def test_evaluation_dtype_defaults_to_float32_and_accepts_float64(self):
        parameter = np.linspace(-1.0, 1.0, 24)
        high_dimensional = np.column_stack((parameter, parameter ** 2, parameter ** 3))
        embedding = np.column_stack((parameter, np.zeros_like(parameter)))

        trustworthiness(high_dimensional, embedding, n_neighbors=4)
        continuity(high_dimensional, embedding, n_neighbors=4, dtype=np.float64)
        neighborhood_preservation(high_dimensional, embedding, n_neighbors=4, dtype=np.float64)
        geodesic_distance_correlation(
            high_dimensional,
            embedding,
            n_neighbors=4,
            n_landmarks=6,
            dtype=np.float64,
        )

        diagram32 = compute_persistence_diagram(embedding, max_points=None)
        diagram64 = compute_persistence_diagram(embedding, max_points=None, dtype=np.float64)
        self.assertEqual(diagram32.h0.dtype, np.dtype(np.float32))
        self.assertEqual(diagram32.h1.dtype, np.dtype(np.float32))
        self.assertEqual(diagram64.h0.dtype, np.dtype(np.float64))
        self.assertEqual(diagram64.h1.dtype, np.dtype(np.float64))

        homology64 = persistent_homology_distance(
            high_dimensional,
            embedding,
            max_points=None,
            dtype=np.float64,
        )
        self.assertEqual(homology64.source_diagram.h0.dtype, np.dtype(np.float64))

        pseudo32 = pseudotime_correlation(embedding, parameter)
        pseudo64 = pseudotime_correlation(embedding, parameter, dtype=np.float64)
        self.assertEqual(pseudo32.arc_length.dtype, np.dtype(np.float32))
        self.assertEqual(pseudo64.arc_length.dtype, np.dtype(np.float64))

        with self.assertRaises(ValueError):
            trustworthiness(high_dimensional, embedding, n_neighbors=4, dtype=np.float16)

    def test_classic_neighbor_metrics_prefer_structure_preserving_embedding(self):
        parameter = np.linspace(-1.0, 1.0, 48)
        high_dimensional = np.column_stack((parameter, 0.3 * parameter, -0.7 * parameter))
        good_embedding = np.column_stack((parameter, np.zeros_like(parameter)))
        bad_embedding = np.column_stack((np.abs(parameter), np.sin(3.0 * np.pi * parameter)))

        trust_good = trustworthiness(high_dimensional, good_embedding, n_neighbors=5)
        trust_bad = trustworthiness(high_dimensional, bad_embedding, n_neighbors=5)
        continuity_good = continuity(high_dimensional, good_embedding, n_neighbors=5)
        continuity_bad = continuity(high_dimensional, bad_embedding, n_neighbors=5)
        overlap_good = neighborhood_preservation(high_dimensional, good_embedding, n_neighbors=5)
        overlap_bad = neighborhood_preservation(high_dimensional, bad_embedding, n_neighbors=5)

        self.assertGreater(trust_good, trust_bad)
        self.assertGreater(continuity_good, continuity_bad)
        self.assertGreater(overlap_good.score, overlap_bad.score)
        self.assertGreater(trust_good, 0.95)
        self.assertGreater(continuity_good, 0.95)
        self.assertGreater(overlap_good.score, 0.9)
        self.assertEqual(overlap_good.local_scores.shape[0], parameter.shape[0])

    def test_persistence_diagram_recovers_circle_betti_number(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        circle = np.column_stack((np.cos(theta), np.sin(theta)))
        chord_1 = 2.0 * np.sin(np.pi / 12.0)
        chord_2 = 2.0 * np.sin(2.0 * np.pi / 12.0)
        scale = 0.5 * (chord_1 + chord_2)

        diagram = compute_persistence_diagram(circle, max_points=None, max_edge_length=scale)
        betti = betti_numbers_at_scale(diagram, scale=scale)

        self.assertEqual(betti[0], 1)
        self.assertEqual(betti[1], 1)

    def test_persistent_homology_distance_prefers_topology_preserving_embedding(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        source = np.column_stack((np.cos(theta), np.sin(theta), 0.1 * np.sin(2.0 * theta)))
        good_embedding = np.column_stack((np.cos(theta), np.sin(theta)))
        bad_embedding = np.column_stack((np.linspace(-1.0, 1.0, theta.shape[0]), np.zeros_like(theta)))

        good = persistent_homology_distance(source, good_embedding, max_points=None)
        bad = persistent_homology_distance(source, bad_embedding, max_points=None)

        self.assertLess(good.wasserstein_total, bad.wasserstein_total)
        self.assertLess(good.wasserstein_h1, bad.wasserstein_h1)

    def test_pseudotime_correlation_tracks_embedding_trajectory(self):
        pseudotime = np.linspace(0.0, 1.0, 20)
        embedding = np.column_stack((pseudotime, pseudotime ** 2))
        shuffled = np.random.default_rng(0).permutation(pseudotime)

        forward = pseudotime_correlation(embedding, pseudotime)
        shuffled_result = pseudotime_correlation(embedding, shuffled)

        self.assertGreater(forward.correlation, 0.99)
        self.assertLess(shuffled_result.correlation, 0.7)


if __name__ == "__main__":
    unittest.main()
