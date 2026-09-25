from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import pairwise_distances

from ibumap.evaluation import (
    SCDEEDResult,
    permute_features_across_samples,
    scdeed,
    scdeed_from_embeddings,
)


def _dense_similarity_reference(
    source: np.ndarray,
    embedding: np.ndarray,
    *,
    similarity_percent: float,
) -> np.ndarray:
    n_samples = source.shape[0]
    n_selected = int(np.floor(n_samples * similarity_percent))
    source_distances = pairwise_distances(source)
    embedding_distances = pairwise_distances(embedding)
    np.fill_diagonal(source_distances, np.inf)
    np.fill_diagonal(embedding_distances, np.inf)

    scores = np.empty(n_samples, dtype=np.float64)
    for index in range(n_samples):
        source_neighbors = np.argsort(
            source_distances[index],
            kind="stable",
        )[:n_selected]
        source_neighbor_distances_after_embedding = embedding_distances[
            index,
            source_neighbors,
        ]
        nearest_embedding_distances = np.sort(
            embedding_distances[index],
            kind="stable",
        )[:n_selected]
        scores[index] = np.corrcoef(
            source_neighbor_distances_after_embedding,
            nearest_embedding_distances,
        )[0, 1]
    return scores


class SCDEEDTest(unittest.TestCase):
    def test_feature_permutation_preserves_column_marginals(self):
        X = np.arange(48, dtype=np.float64).reshape(12, 4)
        original = X.copy()

        first = permute_features_across_samples(
            X,
            random_state=17,
            dtype=np.float64,
        )
        second = permute_features_across_samples(
            X,
            random_state=17,
            dtype=np.float64,
        )

        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(first, second)
        for feature_index in range(X.shape[1]):
            np.testing.assert_array_equal(
                np.sort(first[:, feature_index]),
                np.sort(X[:, feature_index]),
            )
        self.assertFalse(np.array_equal(first, X))

    def test_scores_and_classification_match_dense_reference(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(28, 6))
        projection = rng.normal(size=(6, 2))
        embedding = X @ projection + 0.03 * rng.normal(size=(28, 2))
        X_permuted = permute_features_across_samples(
            X,
            random_state=101,
            dtype=np.float64,
        )
        embedding_permuted = (
            X_permuted @ projection + 0.03 * rng.normal(size=(28, 2))
        )
        similarity_percent = 0.5
        dubious_cutoff = 0.2
        trustworthy_cutoff = 0.8

        result = scdeed_from_embeddings(
            X,
            embedding,
            X_permuted,
            embedding_permuted,
            similarity_percent=similarity_percent,
            dubious_cutoff=dubious_cutoff,
            trustworthy_cutoff=trustworthy_cutoff,
            low_memory=False,
            dtype=np.float64,
        )

        expected_original = _dense_similarity_reference(
            X,
            embedding,
            similarity_percent=similarity_percent,
        )
        expected_permuted = _dense_similarity_reference(
            X_permuted,
            embedding_permuted,
            similarity_percent=similarity_percent,
        )
        expected_dubious_threshold = float(
            np.quantile(expected_permuted, dubious_cutoff)
        )
        expected_trustworthy_threshold = float(
            np.quantile(expected_permuted, trustworthy_cutoff)
        )
        expected_dubious = np.flatnonzero(
            expected_original < expected_dubious_threshold
        )
        expected_trustworthy = np.flatnonzero(
            expected_original > expected_trustworthy_threshold
        )
        expected_intermediate = np.setdiff1d(
            np.arange(X.shape[0]),
            np.concatenate((expected_dubious, expected_trustworthy)),
        )

        self.assertIsInstance(result, SCDEEDResult)
        np.testing.assert_allclose(
            result.rho_original,
            expected_original,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.rho_permuted,
            expected_permuted,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            result.dubious_threshold,
            expected_dubious_threshold,
        )
        self.assertAlmostEqual(
            result.trustworthy_threshold,
            expected_trustworthy_threshold,
        )
        np.testing.assert_array_equal(result.dubious_indices, expected_dubious)
        np.testing.assert_array_equal(
            result.trustworthy_indices,
            expected_trustworthy,
        )
        np.testing.assert_array_equal(
            result.intermediate_indices,
            expected_intermediate,
        )
        self.assertEqual(
            result.n_dubious + result.n_trustworthy + result.n_intermediate,
            X.shape[0],
        )
        self.assertAlmostEqual(
            result.dubious_fraction,
            result.n_dubious / X.shape[0],
        )

    def test_low_memory_matches_full_mode(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(31, 7))
        X_permuted = permute_features_across_samples(
            X,
            random_state=29,
            dtype=np.float64,
        )
        embedding = np.column_stack(
            (
                0.7 * X[:, 0] - 0.2 * X[:, 3],
                X[:, 1] + 0.4 * X[:, 5],
            )
        )
        embedding_permuted = np.column_stack(
            (
                0.7 * X_permuted[:, 0] - 0.2 * X_permuted[:, 3],
                X_permuted[:, 1] + 0.4 * X_permuted[:, 5],
            )
        )

        full = scdeed_from_embeddings(
            X,
            embedding,
            X_permuted,
            embedding_permuted,
            low_memory=False,
            dtype=np.float64,
        )
        low_memory = scdeed_from_embeddings(
            X,
            embedding,
            X_permuted,
            embedding_permuted,
            low_memory=True,
            low_memory_batch_size=4,
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            low_memory.rho_original,
            full.rho_original,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            low_memory.rho_permuted,
            full.rho_permuted,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            low_memory.dubious_indices,
            full.dubious_indices,
        )
        np.testing.assert_array_equal(
            low_memory.trustworthy_indices,
            full.trustworthy_indices,
        )
        np.testing.assert_array_equal(
            low_memory.intermediate_indices,
            full.intermediate_indices,
        )

    def test_low_memory_never_requests_a_full_pairwise_matrix(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(19, 5))
        X_permuted = permute_features_across_samples(
            X,
            random_state=31,
            dtype=np.float64,
        )
        embedding = X[:, :2] + 0.05 * rng.normal(size=(19, 2))
        embedding_permuted = (
            X_permuted[:, :2] + 0.05 * rng.normal(size=(19, 2))
        )
        scdeed_module = importlib.import_module("ibumap.evaluation.scdeed")
        original_pairwise_distances = scdeed_module.pairwise_distances
        requested_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def recording_pairwise_distances(left, right, *args, **kwargs):
            requested_shapes.append((np.shape(left), np.shape(right)))
            return original_pairwise_distances(left, right, *args, **kwargs)

        with patch.object(
            scdeed_module,
            "pairwise_distances",
            side_effect=recording_pairwise_distances,
        ):
            scdeed_from_embeddings(
                X,
                embedding,
                X_permuted,
                embedding_permuted,
                low_memory=True,
                low_memory_batch_size=3,
                dtype=np.float64,
            )

        self.assertGreater(len(requested_shapes), 0)
        for left_shape, right_shape in requested_shapes:
            self.assertLessEqual(left_shape[0], 3)
            self.assertEqual(right_shape[0], X.shape[0])
            self.assertNotEqual(left_shape, (X.shape[0], X.shape[0]))

    def test_high_level_api_refits_once_on_the_permuted_input(self):
        rng = np.random.default_rng(37)
        X = rng.normal(size=(24, 5))
        embedding = X[:, :2]
        callback_inputs: list[np.ndarray] = []

        def refit_embedding(X_permuted):
            callback_inputs.append(np.array(X_permuted, copy=True))
            return np.column_stack(
                (
                    X_permuted[:, 0] + 0.2 * X_permuted[:, 2],
                    X_permuted[:, 1] - 0.1 * X_permuted[:, 4],
                )
            )

        result = scdeed(
            X,
            embedding,
            refit_embedding=refit_embedding,
            permutation_random_state=41,
            low_memory=True,
            low_memory_batch_size=5,
            dtype=np.float64,
        )

        self.assertEqual(len(callback_inputs), 1)
        received = callback_inputs[0]
        self.assertEqual(received.shape, X.shape)
        for feature_index in range(X.shape[1]):
            np.testing.assert_array_equal(
                np.sort(received[:, feature_index]),
                np.sort(X[:, feature_index]),
            )
        self.assertEqual(result.permutation_random_state, 41)
        self.assertEqual(result.n_selected_neighbors, 12)
        self.assertEqual(result.labels.dtype, np.dtype(np.int8))

    def test_validation_rejects_invalid_inputs_and_undefined_scores(self):
        X = np.arange(20, dtype=np.float64).reshape(5, 4)
        embedding = X[:, :2]

        with self.assertRaisesRegex(ValueError, "fewer than two neighbors"):
            scdeed_from_embeddings(
                X,
                embedding,
                X,
                embedding,
                similarity_percent=0.3,
                dtype=np.float64,
            )

        with self.assertRaisesRegex(ValueError, "cutoffs must satisfy"):
            scdeed_from_embeddings(
                X,
                embedding,
                X,
                embedding,
                dubious_cutoff=0.9,
                trustworthy_cutoff=0.1,
                dtype=np.float64,
            )

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            scdeed(
                X,
                embedding,
                refit_embedding=lambda values: values[:-1, :2],
                dtype=np.float64,
            )

        rng = np.random.default_rng(43)
        non_degenerate_source = rng.normal(size=(12, 4))
        collapsed_embedding = np.zeros((12, 2), dtype=np.float64)
        with self.assertRaisesRegex(
            ValueError,
            "Pearson correlation is undefined",
        ):
            scdeed_from_embeddings(
                non_degenerate_source,
                collapsed_embedding,
                non_degenerate_source,
                collapsed_embedding,
                dtype=np.float64,
            )


if __name__ == "__main__":
    unittest.main()
