"""Cached embedding-quality evaluation on top of :mod:`ibumap.evaluation`.

The paper scores every embedding with five metrics:

  local   trustworthiness, continuity, neighborhood_preservation  (k = 15 nearest neighbors)
  global  rta (random triplet accuracy), distance_spearman         (10^6 sampled triplets / pairs, seed 42)

Source-side state (source neighbors, sampled triplets and pairs with their source
comparisons/distances) depends only on the dataset and is computed once per
dataset; embedding-side state (embedding neighbors) once per embedding. Both are
stored under a cache root keyed by a hash of the request, so reruns and
additional embeddings of the same dataset reuse them.

Row subsampling of very large datasets (the paper uses at most 10^6 rows, seed 42)
is done by the caller, e.g. with :func:`common.dataset_io.select_row_indices`,
before the points are handed to the store.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from common.paths import ensure_ibumap_importable
from common.run_metadata import json_default, now_utc

DEFAULT_CACHE_VERSION = "ibumap_evaluation_v1"
TIMING_SCHEMA_VERSION = "evaluation_timing_v1"
LOCAL_METRICS = ("trustworthiness", "continuity", "neighborhood_preservation")
GLOBAL_METRICS = ("rta", "distance_spearman")
DEFAULT_METRICS = LOCAL_METRICS + GLOBAL_METRICS
SUPPORTED_METRICS = frozenset(DEFAULT_METRICS) | {"neighborhood", "random_triplet_accuracy"}
# Evaluation protocol of the paper (row subsampling is applied by the caller).
PAPER_PROTOCOL = {
    "metrics": list(DEFAULT_METRICS),
    "n_neighbors": 15,
    "source_metric": "euclidean",
    "dtype": "float32",
    "max_samples": 1_000_000,
    "sample_seed": 42,
    "rta_n_triplets": 1_000_000,
    "rta_random_state": 42,
    "distance_spearman_n_pairs": 1_000_000,
    "distance_spearman_random_state": 42,
}


@dataclass(frozen=True)
class EvaluationCacheRequest:
    metrics: tuple[str, ...] = DEFAULT_METRICS
    n_neighbors: int = 15
    source_metric: str = "euclidean"
    metric_params: dict[str, Any] | None = None
    dtype: str = "float32"
    device: str = "cpu"
    gpu_batch_size: int | None = None
    rank_low_memory: bool = True
    rank_low_memory_batch_size: int = 256
    rta_n_triplets: int = 1_000_000
    rta_random_state: int | None = 42
    rta_distance_batch_size: int = 65_536
    distance_spearman_n_pairs: int | None = 1_000_000
    distance_spearman_random_state: int | None = 42
    distance_spearman_distance_batch_size: int = 65_536

    def __post_init__(self) -> None:
        unknown = sorted(set(self.metrics) - SUPPORTED_METRICS)
        if unknown:
            raise ValueError(f"Unsupported evaluation metric(s) {unknown}; supported: {sorted(SUPPORTED_METRICS)}")

    @property
    def metric_names(self) -> set[str]:
        return set(self.metrics)

    @property
    def needs_source_neighbors(self) -> bool:
        return bool({"continuity", "neighborhood", "neighborhood_preservation"} & self.metric_names)

    @property
    def needs_embedding_neighbors(self) -> bool:
        return bool({"trustworthiness", "neighborhood", "neighborhood_preservation"} & self.metric_names)

    @property
    def needs_neighbors(self) -> bool:
        return self.needs_source_neighbors or self.needs_embedding_neighbors

    @property
    def needs_rta(self) -> bool:
        return bool({"rta", "random_triplet_accuracy"} & self.metric_names)

    @property
    def needs_distance_spearman(self) -> bool:
        return "distance_spearman" in self.metric_names

    def source_state_metadata(self) -> dict[str, Any]:
        device = evaluation_device_status(self.device)
        return {
            "dtype": self.dtype,
            "device": self.device,
            "resolved_device": device.get("resolved"),
            "gpu_batch_size": self.gpu_batch_size,
            "neighbors": {
                "enabled": self.needs_source_neighbors,
                "n_neighbors": int(self.n_neighbors),
                "metric": self.source_metric,
                "metric_params": self.metric_params or {},
            },
            "rta": {
                "enabled": self.needs_rta,
                "n_triplets": int(self.rta_n_triplets),
                "random_state": self.rta_random_state,
                "distance_batch_size": int(self.rta_distance_batch_size),
                "metric": self.source_metric,
                "metric_params": self.metric_params or {},
            },
            "distance_spearman": {
                "enabled": self.needs_distance_spearman,
                "n_pairs": self.distance_spearman_n_pairs,
                "random_state": self.distance_spearman_random_state,
                "distance_batch_size": int(self.distance_spearman_distance_batch_size),
                "metric": self.source_metric,
                "metric_params": self.metric_params or {},
            },
        }

    def embedding_state_metadata(self) -> dict[str, Any]:
        device = evaluation_device_status(self.device)
        return {
            "dtype": self.dtype,
            "device": self.device,
            "resolved_device": device.get("resolved"),
            "gpu_batch_size": self.gpu_batch_size,
            "neighbors": {
                "enabled": self.needs_embedding_neighbors,
                "n_neighbors": int(self.n_neighbors),
                "metric": "euclidean",
                "metric_params": {},
            },
        }

    def scoring_metadata(self) -> dict[str, Any]:
        device = evaluation_device_status(self.device)
        return {
            "metrics": list(self.metrics),
            "rank_low_memory": bool(self.rank_low_memory),
            "rank_low_memory_batch_size": int(self.rank_low_memory_batch_size),
            "rta_n_triplets": int(self.rta_n_triplets),
            "rta_random_state": self.rta_random_state,
            "rta_distance_batch_size": int(self.rta_distance_batch_size),
            "distance_spearman_n_pairs": self.distance_spearman_n_pairs,
            "distance_spearman_random_state": self.distance_spearman_random_state,
            "distance_spearman_distance_batch_size": int(self.distance_spearman_distance_batch_size),
            "device": self.device,
            "resolved_device": device.get("resolved"),
            "gpu_batch_size": self.gpu_batch_size,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "n_neighbors": int(self.n_neighbors),
            "source_metric": self.source_metric,
            "metric_params": self.metric_params or {},
            "dtype": self.dtype,
            "device": self.device,
            "gpu_batch_size": self.gpu_batch_size,
            "rank_low_memory": bool(self.rank_low_memory),
            "rank_low_memory_batch_size": int(self.rank_low_memory_batch_size),
            "rta_n_triplets": int(self.rta_n_triplets),
            "rta_random_state": self.rta_random_state,
            "rta_distance_batch_size": int(self.rta_distance_batch_size),
            "distance_spearman_n_pairs": self.distance_spearman_n_pairs,
            "distance_spearman_random_state": self.distance_spearman_random_state,
            "distance_spearman_distance_batch_size": int(self.distance_spearman_distance_batch_size),
            "source_state": self.source_state_metadata(),
            "embedding_state": self.embedding_state_metadata(),
            "scoring": self.scoring_metadata(),
        }


@dataclass(frozen=True)
class EvaluationCacheResult:
    state: Any
    cache_dir: Path
    used_cache: bool
    hits: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    timings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRunResult:
    scores: dict[str, Any]
    source: EvaluationCacheResult
    embedding: EvaluationCacheResult
    cache: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)


def _evaluation():
    ensure_ibumap_importable()
    import ibumap.evaluation as evaluation

    return evaluation


def _synchronize_gpu_for_timing() -> None:
    try:
        ensure_ibumap_importable()
        from ibumap.evaluation._device import synchronize_gpu_if_loaded
    except Exception:
        return
    synchronize_gpu_if_loaded()


@contextmanager
def _record_timing(timings: MutableMapping[str, Any] | None, name: str, *,
                   synchronize_gpu: bool = False) -> Iterator[None]:
    if timings is None:
        yield
        return
    if synchronize_gpu:
        _synchronize_gpu_for_timing()
    started = time.perf_counter()
    try:
        yield
    finally:
        if synchronize_gpu:
            _synchronize_gpu_for_timing()
        timings[name] = float(timings.get(name, 0.0)) + float(time.perf_counter() - started)


def _new_cache_timings() -> dict[str, Any]:
    return {
        "cache_mode": None,
        "metadata_lookup_seconds": 0.0,
        "load_seconds": 0.0,
        "prepare_seconds": 0.0,
        "serialization_seconds": 0.0,
        "total_seconds": 0.0,
        "load_components": {},
        "compute_components": {},
        "write_components": {},
    }


def _start_total_timing(enabled: bool) -> float | None:
    if not enabled:
        return None
    _synchronize_gpu_for_timing()
    return time.perf_counter()


def _finish_total_timing(timings: dict[str, Any] | None, started: float | None) -> dict[str, Any]:
    if timings is None or started is None:
        return {}
    _synchronize_gpu_for_timing()
    timings["total_seconds"] = float(time.perf_counter() - started)
    return timings


def _cache_mode(timings, load_started, *, use_cache, recompute, metadata_matches) -> None:
    if timings is None:
        return
    if load_started is not None:
        timings["load_seconds"] = float(time.perf_counter() - load_started)
    if not use_cache:
        timings["cache_mode"] = "disabled"
    elif recompute:
        timings["cache_mode"] = "recompute"
    elif metadata_matches:
        timings["cache_mode"] = "partial_miss"
    else:
        timings["cache_mode"] = "miss"


def safe_path_component(value: str, *, max_length: int = 96) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    safe = safe.strip("._") or "unnamed"
    return safe[:max_length]


def stable_hash(payload: Mapping[str, Any] | Sequence[Any] | str, *, length: int = 16) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def array_fingerprint(array: Any, *, max_bytes: int = 1_000_000) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    raw = np.ascontiguousarray(values).view(np.uint8)
    if raw.nbytes <= max_bytes:
        digest.update(raw.tobytes())
    else:
        digest.update(raw[: max_bytes // 2].tobytes())
        digest.update(raw[-max_bytes // 2:].tobytes())
        digest.update(str(raw.nbytes).encode("utf-8"))
    return digest.hexdigest()[:16]


def source_cache_key(*, dataset_id: str, n_rows: int, n_features: int, request: EvaluationCacheRequest,
                     sample_indices: Any | None = None, extra: Mapping[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "request": request.source_state_metadata(),
    }
    if sample_indices is not None:
        payload["sample_indices_hash"] = array_fingerprint(sample_indices)
    if extra:
        payload["extra"] = dict(extra)
    return f"{safe_path_component(dataset_id)}__{stable_hash(payload)}"


def embedding_cache_key(*, embedding_id: str, embedding: Any, request: EvaluationCacheRequest,
                        extra: Mapping[str, Any] | None = None) -> str:
    values = np.asarray(embedding)
    payload: dict[str, Any] = {
        "embedding_id": embedding_id,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "embedding_hash": array_fingerprint(values),
        "request": request.embedding_state_metadata(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return f"{safe_path_component(embedding_id)}__{stable_hash(payload)}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _resolve_cache_root(value: Any, *, base_dir: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (Path(base_dir) / path).resolve()


def _remove_cache_dir(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
                        encoding="utf-8")
    tmp_path.replace(path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def _metadata_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    existing = read_json(path)
    # Compare after the same JSON round trip that the stored metadata went through
    # (NumPy values, tuples and Paths become JSON values).
    canonical_expected = json.loads(json.dumps(dict(expected), sort_keys=True, default=json_default))
    return bool(existing.get("expected") == canonical_expected)


def _base_expected(*, key: str, kind: str, request: EvaluationCacheRequest,
                   metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if kind == "source":
        request_metadata = request.source_state_metadata()
    elif kind == "embedding":
        request_metadata = request.embedding_state_metadata()
    else:
        request_metadata = request.metadata()
    return {"key": key, "kind": kind, "request": request_metadata, "metadata": dict(metadata or {})}


def save_neighbor_result(path: Path, result: Any) -> None:
    arrays = {"indices": result.indices}
    if result.distances is not None:
        arrays["distances"] = result.distances
    atomic_save_npz(path, **arrays)


def load_neighbor_result(path: Path, *, n_neighbors: int, metric: Any, metric_params: dict[str, Any] | None) -> Any:
    payload = np.load(path)
    distances = np.asarray(payload["distances"]) if "distances" in payload.files else None
    return _evaluation().NeighborSearchResult(indices=np.asarray(payload["indices"], dtype=np.int64),
                                              distances=distances, n_neighbors=int(n_neighbors), metric=metric,
                                              metric_params=metric_params)


def save_random_triplet_source_state(path: Path, state: Any) -> None:
    atomic_save_npz(path, anchor_indices=state.anchor_indices, first_indices=state.first_indices,
                    second_indices=state.second_indices, source_comparisons=state.source_comparisons)


def load_random_triplet_source_state(path: Path, *, n_samples: int, n_triplets_requested: int, metric: Any,
                                     metric_params: dict[str, Any] | None) -> Any:
    payload = np.load(path)
    return _evaluation().RandomTripletSourceState(
        anchor_indices=np.asarray(payload["anchor_indices"], dtype=np.int64),
        first_indices=np.asarray(payload["first_indices"], dtype=np.int64),
        second_indices=np.asarray(payload["second_indices"], dtype=np.int64),
        source_comparisons=np.asarray(payload["source_comparisons"], dtype=np.int8),
        n_samples=int(n_samples), n_triplets_requested=int(n_triplets_requested), metric=metric,
        metric_params=metric_params)


def save_distance_spearman_source_state(path: Path, state: Any) -> None:
    atomic_save_npz(path, first_indices=state.first_indices, second_indices=state.second_indices,
                    source_distances=state.source_distances)


def load_distance_spearman_source_state(path: Path, *, n_samples: int, n_pairs_requested: int | None, metric: Any,
                                        metric_params: dict[str, Any] | None) -> Any:
    payload = np.load(path)
    return _evaluation().DistanceSpearmanSourceState(
        first_indices=np.asarray(payload["first_indices"], dtype=np.int64),
        second_indices=np.asarray(payload["second_indices"], dtype=np.int64),
        source_distances=np.asarray(payload["source_distances"], dtype=np.float64),
        n_samples=int(n_samples), n_pairs_requested=n_pairs_requested, metric=metric, metric_params=metric_params)


class EvaluationCacheStore:
    """Filesystem cache for reusable evaluation source/embedding states.

    The store is experiment-agnostic: callers provide stable keys and metadata;
    the store validates metadata and serializes the ``ibumap.evaluation`` states.
    """

    def __init__(self, root: str | Path, *, version: str = DEFAULT_CACHE_VERSION, use_cache: bool = True,
                 recompute: bool = False, delete_embedding_after_score: bool = False,
                 delete_source_after_dataset: bool = False, record_timings: bool = False) -> None:
        self.root = Path(root)
        self.version = str(version)
        self.use_cache = bool(use_cache)
        self.recompute = bool(recompute)
        self.delete_embedding_after_score = bool(delete_embedding_after_score)
        self.delete_source_after_dataset = bool(delete_source_after_dataset)
        self.record_timings = bool(record_timings)

    @property
    def version_root(self) -> Path:
        return self.root / self.version

    def source_dir(self, key: str) -> Path:
        return self.version_root / "source" / safe_path_component(key, max_length=160)

    def embedding_dir(self, source_key: str, embedding_key: str) -> Path:
        return (self.version_root / "embedding" / safe_path_component(source_key, max_length=120)
                / safe_path_component(embedding_key, max_length=160))

    def delete_source_cache(self, key: str) -> bool:
        return _remove_cache_dir(self.source_dir(key))

    def delete_embedding_cache(self, source_key: str, embedding_key: str) -> bool:
        return _remove_cache_dir(self.embedding_dir(source_key, embedding_key))

    def cleanup_after_dataset(self, source_key: str, *, timings: MutableMapping[str, Any] | None = None) -> bool:
        if not self.delete_source_after_dataset:
            return False
        with _record_timing(timings, "source_cache_delete_seconds"):
            return self.delete_source_cache(source_key)

    def load_or_prepare_source_state(self, points: Any, *, key: str, request: EvaluationCacheRequest,
                                     metadata: Mapping[str, Any] | None = None) -> EvaluationCacheResult:
        evaluation = _evaluation()
        cache_dir = self.source_dir(key)
        meta_path = cache_dir / "source_state.json"
        expected = _base_expected(key=key, kind="source", request=request, metadata=metadata)
        points_array = np.asarray(points, dtype=np.dtype(request.dtype))
        hits: list[str] = []
        writes: list[str] = []
        timings = _new_cache_timings() if self.record_timings else None
        total_started = _start_total_timing(self.record_timings)
        load_components = timings["load_components"] if timings is not None else None
        compute_components = timings["compute_components"] if timings is not None else None
        write_components = timings["write_components"] if timings is not None else None
        metadata_matches = False

        with _record_timing(timings, "metadata_lookup_seconds"):
            if self.use_cache and not self.recompute:
                metadata_matches = _metadata_matches(meta_path, expected)
        load_started = time.perf_counter() if timings is not None and metadata_matches else None

        if metadata_matches:
            neighbors = rta = distance_spearman = None
            if request.needs_source_neighbors and (cache_dir / "neighbors.npz").exists():
                with _record_timing(load_components, "neighbors_seconds"):
                    neighbors = load_neighbor_result(cache_dir / "neighbors.npz", n_neighbors=request.n_neighbors,
                                                     metric=request.source_metric,
                                                     metric_params=request.metric_params)
                hits.append("neighbors.npz")
            if request.needs_rta and (cache_dir / "rta.npz").exists():
                with _record_timing(load_components, "rta_seconds"):
                    rta = load_random_triplet_source_state(cache_dir / "rta.npz", n_samples=points_array.shape[0],
                                                           n_triplets_requested=request.rta_n_triplets,
                                                           metric=request.source_metric,
                                                           metric_params=request.metric_params)
                hits.append("rta.npz")
            if request.needs_distance_spearman and (cache_dir / "distance_spearman.npz").exists():
                with _record_timing(load_components, "distance_spearman_seconds"):
                    distance_spearman = load_distance_spearman_source_state(
                        cache_dir / "distance_spearman.npz", n_samples=points_array.shape[0],
                        n_pairs_requested=request.distance_spearman_n_pairs, metric=request.source_metric,
                        metric_params=request.metric_params)
                hits.append("distance_spearman.npz")
            required = sum(int(flag) for flag in (request.needs_source_neighbors, request.needs_rta,
                                                  request.needs_distance_spearman))
            if len(hits) == required:
                if timings is not None and load_started is not None:
                    timings["load_seconds"] = float(time.perf_counter() - load_started)
                    timings["cache_mode"] = "hit"
                state = evaluation.SourceEvaluationState(
                    points=points_array, n_neighbors=int(request.n_neighbors), metric=request.source_metric,
                    metric_params=request.metric_params, neighbors=neighbors, rta=rta,
                    distance_spearman=distance_spearman)
                return EvaluationCacheResult(state=state, cache_dir=cache_dir, used_cache=bool(hits),
                                             hits=tuple(hits), writes=(),
                                             timings=_finish_total_timing(timings, total_started))

        _cache_mode(timings, load_started, use_cache=self.use_cache, recompute=self.recompute,
                    metadata_matches=metadata_matches)
        with _record_timing(timings, "prepare_seconds", synchronize_gpu=True):
            state = evaluation.prepare_source_state(
                points_array, n_neighbors=request.n_neighbors, metric=request.source_metric,
                metric_params=request.metric_params, compute_neighbor_state=request.needs_source_neighbors,
                compute_geodesic_state=False, compute_persistence_state=False,
                compute_rta_state=request.needs_rta, rta_n_triplets=request.rta_n_triplets,
                rta_random_state=request.rta_random_state, rta_distance_batch_size=request.rta_distance_batch_size,
                compute_distance_spearman_state=request.needs_distance_spearman,
                distance_spearman_n_pairs=request.distance_spearman_n_pairs,
                distance_spearman_random_state=request.distance_spearman_random_state,
                distance_spearman_distance_batch_size=request.distance_spearman_distance_batch_size,
                dtype=request.dtype, device=request.device, gpu_batch_size=request.gpu_batch_size,
                timings=compute_components)

        if self.use_cache:
            with _record_timing(timings, "serialization_seconds"):
                cache_dir.mkdir(parents=True, exist_ok=True)
                if state.neighbors is not None:
                    with _record_timing(write_components, "neighbors_seconds"):
                        save_neighbor_result(cache_dir / "neighbors.npz", state.neighbors)
                    writes.append("neighbors.npz")
                if state.rta is not None:
                    with _record_timing(write_components, "rta_seconds"):
                        save_random_triplet_source_state(cache_dir / "rta.npz", state.rta)
                    writes.append("rta.npz")
                if state.distance_spearman is not None:
                    with _record_timing(write_components, "distance_spearman_seconds"):
                        save_distance_spearman_source_state(cache_dir / "distance_spearman.npz",
                                                            state.distance_spearman)
                    writes.append("distance_spearman.npz")
                with _record_timing(write_components, "metadata_seconds"):
                    atomic_write_json(meta_path, {"created_at": now_utc(), "expected": expected, "files": writes})
                writes.append("source_state.json")

        return EvaluationCacheResult(state=state, cache_dir=cache_dir, used_cache=False, hits=(),
                                     writes=tuple(writes), timings=_finish_total_timing(timings, total_started))

    def load_or_prepare_embedding_state(self, embedding: Any, *, source_key: str, key: str, source_state: Any,
                                        request: EvaluationCacheRequest,
                                        metadata: Mapping[str, Any] | None = None) -> EvaluationCacheResult:
        evaluation = _evaluation()
        cache_dir = self.embedding_dir(source_key, key)
        meta_path = cache_dir / "embedding_state.json"
        expected = _base_expected(key=key, kind="embedding", request=request, metadata=metadata)
        embedding_array = np.asarray(embedding, dtype=np.dtype(request.dtype))
        hits: list[str] = []
        writes: list[str] = []
        timings = _new_cache_timings() if self.record_timings else None
        total_started = _start_total_timing(self.record_timings)
        load_components = timings["load_components"] if timings is not None else None
        compute_components = timings["compute_components"] if timings is not None else None
        write_components = timings["write_components"] if timings is not None else None
        metadata_matches = False

        with _record_timing(timings, "metadata_lookup_seconds"):
            if self.use_cache and not self.recompute:
                metadata_matches = _metadata_matches(meta_path, expected)
        load_started = time.perf_counter() if timings is not None and metadata_matches else None

        if metadata_matches:
            neighbors = None
            if request.needs_embedding_neighbors and (cache_dir / "neighbors.npz").exists():
                with _record_timing(load_components, "neighbors_seconds"):
                    neighbors = load_neighbor_result(cache_dir / "neighbors.npz", n_neighbors=request.n_neighbors,
                                                     metric="euclidean", metric_params=None)
                hits.append("neighbors.npz")
            if len(hits) == int(request.needs_embedding_neighbors):
                if timings is not None and load_started is not None:
                    timings["load_seconds"] = float(time.perf_counter() - load_started)
                    timings["cache_mode"] = "hit"
                state = evaluation.EmbeddingEvaluationState(points=embedding_array, neighbors=neighbors)
                return EvaluationCacheResult(state=state, cache_dir=cache_dir, used_cache=bool(hits),
                                             hits=tuple(hits), writes=(),
                                             timings=_finish_total_timing(timings, total_started))

        _cache_mode(timings, load_started, use_cache=self.use_cache, recompute=self.recompute,
                    metadata_matches=metadata_matches)
        with _record_timing(timings, "prepare_seconds", synchronize_gpu=True):
            state = evaluation.prepare_embedding_state(
                embedding_array, source_state, compute_neighbor_state=request.needs_embedding_neighbors,
                compute_geodesic_state=False, compute_persistence_state=False, dtype=request.dtype,
                device=request.device, gpu_batch_size=request.gpu_batch_size, timings=compute_components)

        if self.use_cache:
            with _record_timing(timings, "serialization_seconds"):
                cache_dir.mkdir(parents=True, exist_ok=True)
                if state.neighbors is not None:
                    with _record_timing(write_components, "neighbors_seconds"):
                        save_neighbor_result(cache_dir / "neighbors.npz", state.neighbors)
                    writes.append("neighbors.npz")
                with _record_timing(write_components, "metadata_seconds"):
                    atomic_write_json(meta_path, {"created_at": now_utc(), "expected": expected, "files": writes})
                writes.append("embedding_state.json")

        return EvaluationCacheResult(state=state, cache_dir=cache_dir, used_cache=False, hits=(),
                                     writes=tuple(writes), timings=_finish_total_timing(timings, total_started))

    def evaluate_embedding(self, source_points: Any, embedding: Any, *, source_key: str, embedding_key: str,
                           request: EvaluationCacheRequest, source_metadata: Mapping[str, Any] | None = None,
                           embedding_metadata: Mapping[str, Any] | None = None) -> EvaluationRunResult:
        evaluation = _evaluation()
        run_timings: dict[str, Any] | None = None
        if self.record_timings:
            run_timings = {"schema_version": TIMING_SCHEMA_VERSION, "source": {}, "embedding": {},
                           "scoring": {"total_seconds": 0.0, "components": {}},
                           "cleanup": {"embedding_cache_delete_seconds": 0.0}, "total_seconds": 0.0}
        total_started = _start_total_timing(self.record_timings)

        source_result = self.load_or_prepare_source_state(source_points, key=source_key, request=request,
                                                          metadata=source_metadata)
        if run_timings is not None:
            run_timings["source"] = source_result.timings
        embedding_result = self.load_or_prepare_embedding_state(embedding, source_key=source_key, key=embedding_key,
                                                                source_state=source_result.state, request=request,
                                                                metadata=embedding_metadata)
        if run_timings is not None:
            run_timings["embedding"] = embedding_result.timings
        scoring_timings = run_timings["scoring"] if run_timings is not None else None
        scoring_components = scoring_timings["components"] if scoring_timings is not None else None
        with _record_timing(scoring_timings, "total_seconds", synchronize_gpu=True):
            scores = evaluation.evaluate_embedding_from_states(
                source_result.state, embedding_result.state, metrics=request.metrics,
                rank_low_memory=request.rank_low_memory, rank_low_memory_batch_size=request.rank_low_memory_batch_size,
                rta_n_triplets=request.rta_n_triplets, rta_random_state=request.rta_random_state,
                rta_distance_batch_size=request.rta_distance_batch_size,
                distance_spearman_n_pairs=request.distance_spearman_n_pairs,
                distance_spearman_random_state=request.distance_spearman_random_state,
                distance_spearman_distance_batch_size=request.distance_spearman_distance_batch_size,
                dtype=request.dtype, device=request.device, gpu_batch_size=request.gpu_batch_size,
                timings=scoring_components)
        embedding_deleted = False
        if self.delete_embedding_after_score:
            cleanup_timings = run_timings["cleanup"] if run_timings is not None else None
            with _record_timing(cleanup_timings, "embedding_cache_delete_seconds"):
                embedding_deleted = self.delete_embedding_cache(source_key, embedding_key)
        finalized_timings = _finish_total_timing(run_timings, total_started)
        return EvaluationRunResult(
            scores=scores, source=source_result, embedding=embedding_result,
            cache={
                "root": str(self.version_root),
                "source_dir": str(source_result.cache_dir),
                "embedding_dir": str(embedding_result.cache_dir),
                "source_hits": list(source_result.hits),
                "embedding_hits": list(embedding_result.hits),
                "source_writes": list(source_result.writes),
                "embedding_writes": list(embedding_result.writes),
                "source_used_cache": bool(source_result.used_cache),
                "embedding_used_cache": bool(embedding_result.used_cache),
                "device": evaluation_device_status(request.device),
                "gpu_batch_size": request.gpu_batch_size,
                "delete_embedding_after_score": bool(self.delete_embedding_after_score),
                "delete_source_after_dataset": bool(self.delete_source_after_dataset),
                "embedding_deleted_after_score": embedding_deleted,
                "record_timings": bool(self.record_timings),
            },
            timings=finalized_timings)


def cache_store_from_config(config: Mapping[str, Any], *, base_dir: str | Path,
                            default_root: str | Path = "cache/evaluation") -> EvaluationCacheStore:
    """Build a store from an evaluation config (``cache:`` and ``timing:`` sections)."""
    cache_config = config.get("cache")
    if not isinstance(cache_config, Mapping):
        cache_config = {}
    timing_config = config.get("timing")
    if not isinstance(timing_config, Mapping):
        timing_config = {}
    root_value = cache_config.get("root", config.get("evaluation_cache_root", default_root))
    return EvaluationCacheStore(
        _resolve_cache_root(root_value, base_dir=base_dir),
        version=str(cache_config.get("version", config.get("cache_version", DEFAULT_CACHE_VERSION))),
        use_cache=_as_bool(cache_config.get("enabled", config.get("use_cache", True)), default=True),
        recompute=_as_bool(cache_config.get("recompute", config.get("recompute_cache", False)), default=False),
        delete_embedding_after_score=_as_bool(
            cache_config.get("delete_embedding_after_score", config.get("delete_embedding_after_score", False)),
            default=False),
        delete_source_after_dataset=_as_bool(
            cache_config.get("delete_source_after_dataset", config.get("delete_source_after_dataset", False)),
            default=False),
        record_timings=_as_bool(timing_config.get("enabled", config.get("record_timings", False)), default=False),
    )


def evaluation_device_status(device: Any = "cpu") -> dict[str, Any]:
    requested = str(device)
    try:
        ensure_ibumap_importable()
        from ibumap.evaluation._device import gpu_available, resolve_device
    except Exception as exc:
        return {"requested": requested, "resolved": requested, "gpu_available": False,
                "reason": f"Could not import evaluation device helpers: {exc}"}
    available, reason = gpu_available()
    try:
        resolved = resolve_device(requested)
    except Exception as exc:
        return {"requested": requested, "resolved": None, "gpu_available": available, "reason": str(exc)}
    return {"requested": requested, "resolved": resolved, "gpu_available": available, "reason": reason}


def summarize_evaluation_scores(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe score summary without large per-point arrays."""
    summary: dict[str, Any] = {}
    for name, value in scores.items():
        if isinstance(value, (int, float, np.generic)):
            summary[name] = float(value)
        elif name == "neighborhood_preservation" and hasattr(value, "score"):
            local_scores = np.asarray(value.local_scores)
            summary[name] = {
                "score": float(value.score),
                "n_neighbors": int(value.n_neighbors),
                "local_scores_count": int(local_scores.shape[0]),
                "local_scores_min": float(np.min(local_scores)) if local_scores.size else None,
                "local_scores_max": float(np.max(local_scores)) if local_scores.size else None,
            }
        elif name == "rta" and hasattr(value, "n_valid_triplets"):
            summary[name] = {
                "score": float(value.score),
                "n_triplets": int(value.n_triplets),
                "n_valid_triplets": int(value.n_valid_triplets),
                "source_tie_count": int(value.source_tie_count),
                "embedding_tie_count": int(value.embedding_tie_count),
            }
        elif name == "distance_spearman" and hasattr(value, "spearman_correlation"):
            summary[name] = {
                "spearman_correlation": float(value.spearman_correlation),
                "pvalue": float(value.pvalue),
                "n_pairs": int(value.n_pairs),
            }
        else:
            summary[name] = value
    return summary


def scalar_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    """One float per metric from :func:`summarize_evaluation_scores` output (the values the paper reports)."""
    output: dict[str, float] = {}
    for name, value in summary.items():
        if isinstance(value, Mapping):
            value = value.get("score", value.get("spearman_correlation"))
        if value is not None:
            output["neighborhood_preservation" if name == "neighborhood" else name] = float(value)
    return output


def request_from_config(config: Mapping[str, Any]) -> EvaluationCacheRequest:
    """Build a request from an evaluation config; unspecified values follow :data:`PAPER_PROTOCOL`."""
    metrics = tuple(str(metric) for metric in config.get("metrics", DEFAULT_METRICS))
    return EvaluationCacheRequest(
        metrics=metrics,
        n_neighbors=int(config.get("n_neighbors", config.get("knn_k", 15))),
        source_metric=str(config.get("source_metric", config.get("distance_metric", "euclidean"))),
        metric_params=config.get("metric_params"),
        dtype=str(config.get("dtype", "float32")),
        device=str(config.get("device", config.get("evaluation_device", "cpu"))),
        gpu_batch_size=_optional_int(config.get("gpu_batch_size")),
        rank_low_memory=_as_bool(config.get("rank_low_memory", True), default=True),
        rank_low_memory_batch_size=int(config.get("rank_low_memory_batch_size", 256)),
        rta_n_triplets=int(config.get("rta_n_triplets", config.get("rta_sample_size", 1_000_000))),
        rta_random_state=_optional_int(config.get("rta_random_state", config.get("rta_seed", 42))),
        rta_distance_batch_size=int(config.get("rta_distance_batch_size", 65_536)),
        distance_spearman_n_pairs=_optional_int(
            config.get("distance_spearman_n_pairs", config.get("distance_spearman_sample_size", 1_000_000))),
        distance_spearman_random_state=_optional_int(
            config.get("distance_spearman_random_state", config.get("distance_spearman_seed", 42))),
        distance_spearman_distance_batch_size=int(config.get("distance_spearman_distance_batch_size", 65_536)),
    )
