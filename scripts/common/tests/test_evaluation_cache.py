from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.evaluation_cache import (
    DEFAULT_METRICS,
    PAPER_PROTOCOL,
    TIMING_SCHEMA_VERSION,
    EvaluationCacheRequest,
    EvaluationCacheStore,
    cache_store_from_config,
    request_from_config,
    scalar_scores,
    summarize_evaluation_scores,
)


def _example_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    source = rng.normal(size=(34, 5)).astype(np.float32)
    source[:, 2:] *= 0.05  # nearly two-dimensional, so the 2-D embedding below is faithful
    embedding = (source[:, :2] + 0.04 * rng.normal(size=(34, 2))).astype(np.float32)
    return source, embedding


def _request(**overrides) -> EvaluationCacheRequest:
    params = dict(metrics=DEFAULT_METRICS, n_neighbors=5, rta_n_triplets=200, rta_random_state=7,
                  distance_spearman_n_pairs=200, distance_spearman_random_state=11)
    params.update(overrides)
    return EvaluationCacheRequest(**params)


def _assert_nonnegative(values: dict[str, float]) -> None:
    assert all(value >= 0.0 for value in values.values())


def test_default_metrics_are_the_paper_metrics() -> None:
    assert DEFAULT_METRICS == ("trustworthiness", "continuity", "neighborhood_preservation", "rta",
                               "distance_spearman")
    request = request_from_config({})
    assert request.metrics == DEFAULT_METRICS
    for key in ("n_neighbors", "rta_n_triplets", "rta_random_state", "distance_spearman_n_pairs",
                "distance_spearman_random_state", "source_metric", "dtype"):
        assert getattr(request, key) == PAPER_PROTOCOL[key], key


@pytest.mark.parametrize("metric", ["geodesic", "persistent_homology", "unknown"])
def test_unsupported_metrics_are_rejected(metric: str) -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation metric"):
        EvaluationCacheRequest(metrics=("trustworthiness", metric))
    with pytest.raises(ValueError, match="Unsupported evaluation metric"):
        request_from_config({"metrics": [metric]})


def test_cache_store_records_cold_and_warm_lifecycle_timings(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    store = EvaluationCacheStore(tmp_path / "cache", record_timings=True)
    cold = store.evaluate_embedding(source, embedding, source_key="source", embedding_key="embedding",
                                    request=_request())

    assert cold.timings["schema_version"] == TIMING_SCHEMA_VERSION
    assert cold.timings["source"]["cache_mode"] == "miss"
    assert cold.timings["embedding"]["cache_mode"] == "miss"
    assert set(cold.timings["source"]["compute_components"]) == {
        "neighbors_seconds", "rta_seconds", "distance_spearman_seconds"}
    assert set(cold.timings["embedding"]["compute_components"]) == {"neighbors_seconds"}
    assert set(cold.timings["scoring"]["components"]) == {
        "trustworthiness_seconds", "continuity_seconds", "neighborhood_preservation_seconds", "rta_seconds",
        "distance_spearman_seconds"}
    _assert_nonnegative(cold.timings["source"]["compute_components"])
    _assert_nonnegative(cold.timings["embedding"]["compute_components"])
    _assert_nonnegative(cold.timings["scoring"]["components"])
    assert cold.timings["total_seconds"] >= 0.0

    warm = store.evaluate_embedding(source, embedding, source_key="source", embedding_key="embedding",
                                    request=_request())
    assert warm.timings["source"]["cache_mode"] == "hit"
    assert warm.timings["embedding"]["cache_mode"] == "hit"
    assert set(warm.timings["source"]["load_components"]) == {
        "neighbors_seconds", "rta_seconds", "distance_spearman_seconds"}
    assert set(warm.timings["embedding"]["load_components"]) == {"neighbors_seconds"}
    assert summarize_evaluation_scores(warm.scores) == summarize_evaluation_scores(cold.scores)


def test_scalar_scores_give_one_value_per_paper_metric(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    result = EvaluationCacheStore(tmp_path / "cache", use_cache=False).evaluate_embedding(
        source, embedding, source_key="source", embedding_key="embedding", request=_request())
    values = scalar_scores(summarize_evaluation_scores(result.scores))
    assert set(values) == set(DEFAULT_METRICS)
    assert all(0.0 <= values[m] <= 1.0 for m in ("trustworthiness", "continuity", "neighborhood_preservation",
                                                  "rta"))
    assert -1.0 <= values["distance_spearman"] <= 1.0
    # A near-copy of the first two coordinates keeps local structure well.
    assert values["trustworthiness"] > 0.8


def test_timing_collection_is_opt_in(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    result = EvaluationCacheStore(tmp_path / "cache", use_cache=False).evaluate_embedding(
        source, embedding, source_key="source", embedding_key="embedding",
        request=EvaluationCacheRequest(metrics=("trustworthiness",), n_neighbors=5))
    assert result.timings == {}
    assert result.source.timings == {}
    assert result.embedding.timings == {}


def test_cache_hit_accepts_numpy_array_metadata(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    request = EvaluationCacheRequest(metrics=("trustworthiness",), n_neighbors=5)
    store = EvaluationCacheStore(tmp_path / "cache", record_timings=True)
    metadata = {"dataset_id": "example", "sample_indices": np.arange(source.shape[0], dtype=np.int64)}
    cold = store.evaluate_embedding(source, embedding, source_key="source-with-array-metadata",
                                    embedding_key="embedding-cold", request=request, source_metadata=metadata)
    warm = store.evaluate_embedding(source, embedding, source_key="source-with-array-metadata",
                                    embedding_key="embedding-warm", request=request, source_metadata=metadata)
    assert cold.source.timings["cache_mode"] == "miss"
    assert warm.source.timings["cache_mode"] == "hit"


def test_global_metric_source_samples_are_cached_and_reused(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    request = _request(metrics=("rta", "distance_spearman"))
    store = EvaluationCacheStore(tmp_path / "cache", record_timings=True)
    cold = store.evaluate_embedding(source, embedding, source_key="source", embedding_key="embedding-a",
                                    request=request)
    warm = store.evaluate_embedding(source, embedding + 0.01, source_key="source", embedding_key="embedding-b",
                                    request=request)
    assert cold.source.timings["cache_mode"] == "miss"
    assert set(cold.source.timings["compute_components"]) == {"rta_seconds", "distance_spearman_seconds"}
    assert set(cold.timings["scoring"]["components"]) == {"rta_seconds", "distance_spearman_seconds"}
    assert warm.source.timings["cache_mode"] == "hit"
    assert set(warm.source.timings["load_components"]) == {"rta_seconds", "distance_spearman_seconds"}
    assert {"rta.npz", "distance_spearman.npz"}.issubset(set(cold.source.writes))
    assert {"rta.npz", "distance_spearman.npz"}.issubset(set(warm.source.hits))
    np.testing.assert_array_equal(cold.source.state.rta.anchor_indices, warm.source.state.rta.anchor_indices)
    np.testing.assert_array_equal(cold.source.state.distance_spearman.first_indices,
                                  warm.source.state.distance_spearman.first_indices)


def test_changed_sampling_seed_is_a_cache_miss(tmp_path: Path) -> None:
    source, embedding = _example_inputs()
    store = EvaluationCacheStore(tmp_path / "cache", record_timings=True)
    store.evaluate_embedding(source, embedding, source_key="source", embedding_key="e", request=_request())
    other = store.evaluate_embedding(source, embedding, source_key="source", embedding_key="e",
                                     request=_request(rta_random_state=8))
    assert other.source.timings["cache_mode"] == "miss"


def test_timing_config_and_source_cleanup_timing(tmp_path: Path) -> None:
    store = cache_store_from_config({"cache": {"root": "cache", "delete_source_after_dataset": True},
                                     "timing": {"enabled": True}}, base_dir=tmp_path)
    source_dir = store.source_dir("source")
    source_dir.mkdir(parents=True)
    cleanup_timings: dict[str, float] = {}
    deleted = store.cleanup_after_dataset("source", timings=cleanup_timings)
    assert store.record_timings is True
    assert deleted is True
    assert cleanup_timings["source_cache_delete_seconds"] >= 0.0
