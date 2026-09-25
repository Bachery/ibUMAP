"""Shared contract and adapters of the end-to-end benchmark (paper Section 5, Appendix C).

Every run record carries hashes of the configuration, of the measured source
code (controller, adapters, shared helpers and, for ibUMAP, ``src/ibumap``), of
the input features and of the embedding, plus package/runtime and hardware
signatures. Resuming, evaluation and summarization accept a record only when all
of them still match, so results produced by different code, parameters,
environments or machines are never mixed silently.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import inspect
import json
import os
import resource
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
IBUMAP_SRC = REPO_ROOT / "src"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.paths import ensure_ibumap_importable, repo_relative  # noqa: E402

ensure_ibumap_importable()

from common.callable_utils import filtered_kwargs  # noqa: E402
from common.config import load_yaml, resolve_path  # noqa: E402
from common.dataset_io import ProcessedDataset, load_processed_dataset  # noqa: E402
from common.embedding_diagnostics import bbox_summary  # noqa: E402
from common.gpu_runtime import (  # noqa: E402
    machine_info,
    synchronize_gpu,
    to_numpy_array,
)
from common.run_metadata import (  # noqa: E402
    git_commit,
    json_default,
    now_utc,
    package_versions,
)
from common.task_grid import entry_name, filter_entries  # noqa: E402


CONFIG_FILES = ("paths", "datasets", "algorithms", "evaluation", "experiment")
PROTOCOL_VERSION = "20260806.1"
DEFAULT_STABILITY_MAX_POINTS = 10000
DEFAULT_STABILITY_PAIRS = 20000
DEFAULT_BOOTSTRAP_SAMPLES = 500
RESOURCE_SKIP_ALGORITHMS = (
    "umap_learn_seed_42",
    "umap_learn_seed_none",
    "ibumap_cpu_seed_42",
    "ibumap_cpu_seed_none",
    "torchdr_umap_seed_42",
    "torchdr_umap_seed_none",
)
RESOURCE_SKIP_DATASETS = (
    "gist_960_euclidean",
    "whole_mouse_brain_merfish_animal2_coronal",
    "whole_mouse_brain_merfish_animal3_sagittal",
    "google_news_300d",
    "whole_mouse_brain_merfish_animal1_coronal",
)
RESOURCE_SKIP_REASON = (
    "known host/GPU memory exhaustion for umap-learn, ibUMAP CPU, and "
    "TorchDR on the five datasets in the resource skip list"
)
PROTOCOL_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".py",
    ".pyx",
}
# Cython writes this ignored platform-specific translation next to its tracked
# .pyx source. Its contents and presence depend on the build host, so it is not
# part of the portable source contract.
GENERATED_PROTOCOL_SOURCES = {
    "src/ibumap/graph/_cuml_graph_ext.cpp",
}
SHARED_PROTOCOL_SOURCES = (
    SCRIPTS_ROOT / "common" / "algorithm_adapters.py",
    SCRIPTS_ROOT / "common" / "callable_utils.py",
    SCRIPTS_ROOT / "common" / "dataset_io.py",
    SCRIPTS_ROOT / "common" / "fixed_inputs.py",
    SCRIPTS_ROOT / "common" / "gpu_runtime.py",
    SCRIPTS_ROOT / "common" / "run_metadata.py",
)
ALGORITHM_META_KEYS = {
    "name",
    "id",
    "family",
    "algorithm",
    "device",
    "enabled",
    "baseline",
    "seed_mode",
    "execution_profile",
    "notes",
    "description",
    "params",
}
RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
)


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_root: Path
    repo_root: Path
    config_dir: Path
    processed_root: Path
    results_root: Path
    embeddings_root: Path
    evaluation_root: Path
    summaries_root: Path
    embedding_plots_root: Path
    time_plots_root: Path
    metric_plots_root: Path
    logs_root: Path


@dataclass(frozen=True)
class BenchmarkTask:
    dataset_id: str
    algorithm_id: str
    repeat: int
    n_neighbors: int
    execution_index: int
    algorithm_entry: Mapping[str, Any]

    @property
    def run_id(self) -> str:
        return (
            f"{self.algorithm_id}__nn_{self.n_neighbors}"
            f"__repeat_{self.repeat:02d}"
        )

    @property
    def execution_profile(self) -> str:
        return str(self.algorithm_entry["execution_profile"])


def load_configs(config_dir: Path) -> dict[str, Any]:
    return {
        name: load_yaml(config_dir / f"{name}.yaml") or {}
        for name in CONFIG_FILES
    }


def _configured_path(
    config: Mapping[str, Any], key: str, *, config_dir: Path, default: Path
) -> Path:
    resolved = resolve_path(config.get(key), base_dir=config_dir)
    return resolved or default


def experiment_paths(config_dir: Path, configs: Mapping[str, Any]) -> ExperimentPaths:
    path_config = configs.get("paths") or {}
    results_root = _configured_path(
        path_config, "results_root", config_dir=config_dir, default=EXPERIMENT_ROOT / "results"
    )
    return ExperimentPaths(
        experiment_root=EXPERIMENT_ROOT,
        repo_root=REPO_ROOT,
        config_dir=config_dir,
        processed_root=_configured_path(
            path_config,
            "processed_root",
            config_dir=config_dir,
            default=REPO_ROOT / "datasets" / "processed",
        ),
        results_root=results_root,
        embeddings_root=_configured_path(
            path_config,
            "embeddings_root",
            config_dir=config_dir,
            default=results_root / "01_embeddings",
        ),
        evaluation_root=_configured_path(
            path_config,
            "evaluation_root",
            config_dir=config_dir,
            default=results_root / "02_evaluations",
        ),
        summaries_root=_configured_path(
            path_config,
            "summaries_root",
            config_dir=config_dir,
            default=results_root / "03_summaries",
        ),
        embedding_plots_root=_configured_path(
            path_config,
            "embedding_plots_root",
            config_dir=config_dir,
            default=results_root / "04_embedding_plots",
        ),
        time_plots_root=_configured_path(
            path_config,
            "time_plots_root",
            config_dir=config_dir,
            default=results_root / "05_time_plots",
        ),
        metric_plots_root=_configured_path(
            path_config,
            "metric_plots_root",
            config_dir=config_dir,
            default=results_root / "06_metrics_plots",
        ),
        logs_root=_configured_path(
            path_config,
            "logs_root",
            config_dir=config_dir,
            default=EXPERIMENT_ROOT / "logs",
        ),
    )


def ensure_experiment_dirs(paths: ExperimentPaths) -> None:
    directories = (
        paths.results_root,
        paths.embeddings_root / "per_run",
        paths.embeddings_root / "metadata",
        paths.embeddings_root / "failures",
        paths.embeddings_root / "warmups",
        paths.evaluation_root / "per_run",
        paths.evaluation_root / "failures",
        paths.evaluation_root / "metric_cache",
        paths.evaluation_root / "sample_indices",
        paths.summaries_root,
        paths.embedding_plots_root / "per_run",
        paths.embedding_plots_root / "per_dataset",
        paths.time_plots_root,
        paths.metric_plots_root,
        paths.logs_root,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(EXPERIMENT_ROOT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT_ROOT / ".mplconfig"))


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configuration_hash(configs: Mapping[str, Any]) -> str:
    return canonical_hash({name: configs.get(name) for name in CONFIG_FILES})


def _relative_source_name(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_protocol_source(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in PROTOCOL_SOURCE_SUFFIXES
        and _relative_source_name(path) not in GENERATED_PROTOCOL_SOURCES
    )


@functools.lru_cache(maxsize=None)
def protocol_source_manifest(family: str) -> dict[str, str]:
    roots = [
        Path(__file__).resolve(),
        EXPERIMENT_ROOT / "scripts" / "01_run_e2e_benchmark.py",
        *SHARED_PROTOCOL_SOURCES,
    ]
    if family == "ibumap":
        roots.append(IBUMAP_SRC / "ibumap")
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if _is_protocol_source(path)
            )
        elif root.is_file():
            files.append(root)
        else:
            raise FileNotFoundError(f"Protocol source is missing: {root}")
    return {
        _relative_source_name(path): file_sha256(path)
        for path in sorted(set(files), key=_relative_source_name)
    }


@functools.lru_cache(maxsize=1)
def summary_source_manifest() -> dict[str, str]:
    roots = [
        Path(__file__).resolve(),
        EXPERIMENT_ROOT / "scripts" / "02_run_evaluations.py",
        EXPERIMENT_ROOT / "scripts" / "03_summarize_results.py",
        SCRIPTS_ROOT / "common" / "dataset_io.py",
        SCRIPTS_ROOT / "common" / "evaluation_cache.py",
        SCRIPTS_ROOT / "common" / "gpu_runtime.py",
        SCRIPTS_ROOT / "common" / "run_metadata.py",
        IBUMAP_SRC / "ibumap" / "evaluation",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if _is_protocol_source(path)
            )
        elif root.is_file():
            files.append(root)
        else:
            raise FileNotFoundError(f"Summary source is missing: {root}")
    return {
        _relative_source_name(path): file_sha256(path)
        for path in sorted(set(files), key=_relative_source_name)
    }


def benchmark_policy(configs: Mapping[str, Any]) -> dict[str, Any]:
    experiment = dict(configs.get("experiment") or {})
    configured_version = str(experiment.get("protocol_version") or "")
    return {
        "protocol_version": configured_version,
        "warmup_runs": int(experiment.get("warmup_runs", 1)),
        "timing": experiment.get("timing") or {},
        "recording": experiment.get("recording") or {},
        "environment_requirements": experiment.get("environment_requirements") or {},
        "resource_skip_policy": {
            "algorithms": list(RESOURCE_SKIP_ALGORITHMS),
            "datasets": list(RESOURCE_SKIP_DATASETS),
            "reason": RESOURCE_SKIP_REASON,
        },
    }


def benchmark_configuration_hash(configs: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "datasets": configs.get("datasets") or {},
            "algorithms": configs.get("algorithms") or {},
            "benchmark_policy": benchmark_policy(configs),
        }
    )


def benchmark_protocol_hash(
    configs: Mapping[str, Any],
    algorithm_entry: Mapping[str, Any],
) -> str:
    family = str(algorithm_entry.get("family"))
    return canonical_hash(
        {
            "benchmark_policy": benchmark_policy(configs),
            "family": family,
            "source_manifest": protocol_source_manifest(family),
        }
    )


def evaluation_request_hash(
    evaluation_config: Mapping[str, Any],
    *,
    max_samples: int | None,
    sample_seed: int,
) -> str:
    scoring_config = {
        str(key): value
        for key, value in evaluation_config.items()
        if str(key) != "cache"
    }
    return canonical_hash(
        {
            "evaluation": scoring_config,
            "sampling": {
                "max_samples": max_samples,
                "sample_seed": int(sample_seed),
            },
        }
    )


def configured_evaluation_request_hash(configs: Mapping[str, Any]) -> str:
    evaluation = dict(configs.get("evaluation") or {})
    configured_max = evaluation.get("max_samples")
    return evaluation_request_hash(
        evaluation,
        max_samples=None if configured_max is None else int(configured_max),
        sample_seed=int(evaluation.get("sample_seed", 42)),
    )


def summary_policy() -> dict[str, Any]:
    return {
        "stability_max_points": DEFAULT_STABILITY_MAX_POINTS,
        "stability_pairs": DEFAULT_STABILITY_PAIRS,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "skip_stability": False,
    }


def summary_configuration_hash(configs: Mapping[str, Any]) -> str:
    experiment = dict(configs.get("experiment") or {})
    return canonical_hash(
        {
            "datasets": configs.get("datasets") or {},
            "algorithms": configs.get("algorithms") or {},
            "evaluation_request_hash": configured_evaluation_request_hash(configs),
            "summary_policy": summary_policy(),
            "task_grid": {
                "repeats": int(experiment.get("repeats", 1)),
                "order_seed": int(experiment.get("order_seed", 20260630)),
                "shuffle_within_repeat_blocks": bool(
                    experiment.get("shuffle_within_repeat_blocks", True)
                ),
                "resource_skip_algorithms": list(RESOURCE_SKIP_ALGORITHMS),
                "resource_skip_datasets": list(RESOURCE_SKIP_DATASETS),
            },
            "benchmark_policy": benchmark_policy(configs),
            "validation": experiment.get("validation") or {},
            "summary_source_manifest": summary_source_manifest(),
            "benchmark_protocol_hashes": {
                entry_name(entry): benchmark_protocol_hash(configs, entry)
                for entry in (configs.get("algorithms") or {}).get(
                    "algorithms",
                    [],
                )
                if bool(entry.get("enabled", True))
            },
        }
    )


def hardware_signature(environment: Mapping[str, Any]) -> str:
    gpu = environment.get("gpu") if isinstance(environment.get("gpu"), Mapping) else {}
    devices = gpu.get("devices") if isinstance(gpu.get("devices"), Sequence) else []
    normalized_devices = [
        {
            "name": device.get("name"),
            "total_memory_bytes": device.get(
                "total_memory_bytes",
                device.get("runtime_total_memory_bytes"),
            ),
        }
        for device in devices
        if isinstance(device, Mapping)
    ]
    return canonical_hash(
        {
            "platform": environment.get("platform"),
            "cpu_model": environment.get("cpu_model"),
            "gpu_devices": normalized_devices,
        }
    )


def runtime_signature(
    algorithm_entry: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    family = str(algorithm_entry.get("family"))
    packages = (
        environment.get("packages")
        if isinstance(environment.get("packages"), Mapping)
        else {}
    )
    package_names = {
        "umap_learn": (
            "joblib",
            "llvmlite",
            "numba",
            "numpy",
            "pynndescent",
            "scikit-learn",
            "scipy",
            "threadpoolctl",
            "umap-learn",
        ),
        "cuml_umap": (
            "cuml",
            "cuml-cu12",
            "cuvs",
            "cuvs-cu12",
            "cupy",
            "cupy-cuda12x",
            "llvmlite",
            "numba",
            "numpy",
            "rmm",
            "rmm-cu12",
            "scikit-learn",
            "scipy",
        ),
        "ibumap": (
            (
                "cuml",
                "cuml-cu12",
                "cuvs",
                "cuvs-cu12",
                "cupy",
                "cupy-cuda12x",
                "llvmlite",
                "numba",
                "numpy",
                "pynndescent",
                "pyfftw",
                "rmm",
                "rmm-cu12",
                "scikit-learn",
                "scipy",
                "threadpoolctl",
                "umap-learn",
            )
            if str(algorithm_entry.get("device")) == "cuda"
            else (
                "joblib",
                "llvmlite",
                "numba",
                "numpy",
                "pynndescent",
                "pyfftw",
                "scikit-learn",
                "scipy",
                "threadpoolctl",
                "umap-learn",
            )
        ),
        "torchdr_umap": (
            "faiss-cpu",
            "faiss-gpu",
            "llvmlite",
            "numba",
            "numpy",
            "pyfftw",
            "scikit-learn",
            "scipy",
            "torch",
            "torchdr",
        ),
    }.get(family, tuple(sorted(str(key) for key in packages)))
    return canonical_hash(
        {
            "family": family,
            "device": algorithm_entry.get("device"),
            "python_version": environment.get("python_version"),
            "packages": {name: packages.get(name) for name in package_names},
            "runtime_env": environment.get("runtime_env") or {},
            "hardware_signature": hardware_signature(environment),
        }
    )


def evaluation_runtime_signature(
    environment: Mapping[str, Any],
    device_status: Mapping[str, Any],
) -> str:
    packages = (
        environment.get("packages")
        if isinstance(environment.get("packages"), Mapping)
        else {}
    )
    package_names = (
        "cuml",
        "cuml-cu12",
        "cupy",
        "cupy-cuda12x",
        "joblib",
        "llvmlite",
        "numba",
        "numpy",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
    )
    return canonical_hash(
        {
            "python_version": environment.get("python_version"),
            "packages": {name: packages.get(name) for name in package_names},
            "runtime_env": environment.get("runtime_env") or {},
            "hardware_signature": hardware_signature(environment),
            "device": {
                key: device_status.get(key)
                for key in ("requested", "resolved", "gpu_available")
            },
        }
    )


def published_summary_root(
    paths: ExperimentPaths,
    configs: Mapping[str, Any],
) -> Path:
    pointer_path = paths.summaries_root / "published.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No strict summary bundle is published: {pointer_path}. "
            "Run 03_summarize_results.py --strict first."
        )
    pointer = read_json(pointer_path)
    if (
        pointer.get("strict_passed") is not True
        or pointer.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise RuntimeError(f"Published summary pointer is not strict: {pointer_path}")
    published_hash = pointer.get("summary_configuration_hash")
    if not published_hash:
        raise RuntimeError(f"Published summary pointer has no identity: {pointer_path}")
    relative_bundle = Path(str(pointer.get("bundle") or ""))
    bundle = (paths.summaries_root / relative_bundle).resolve()
    summary_root = paths.summaries_root.resolve()
    if (
        relative_bundle.is_absolute()
        or not relative_bundle.parts
        or not bundle.is_relative_to(summary_root)
        or not bundle.is_dir()
    ):
        raise RuntimeError(f"Invalid published summary bundle: {relative_bundle}")
    completeness = read_json(bundle / "completeness.json")
    expected_scope = {
        "datasets": [entry_name(entry) for entry in dataset_entries(configs)],
        "algorithms": [entry_name(entry) for entry in algorithm_entries(configs)],
        "repeats": repeat_values(configs),
        "n_neighbors": n_neighbors_values(configs)[0],
        "analysis": {**summary_policy(), "standard_policy": True},
        "resource_skip_rules": {
            "algorithms": list(RESOURCE_SKIP_ALGORITHMS),
            "datasets": list(RESOURCE_SKIP_DATASETS),
            "reason": RESOURCE_SKIP_REASON,
        },
    }
    if (
        completeness.get("strict_passed") is not True
        or completeness.get("validation_passed") is not True
        or completeness.get("full_scope") is not True
        or completeness.get("published") is not True
        or completeness.get("protocol_version") != PROTOCOL_VERSION
        or completeness.get("summary_configuration_hash") != published_hash
    ):
        raise RuntimeError(f"Published summary manifest is invalid: {bundle}")
    if (
        completeness.get("benchmark_configuration_hash")
        != benchmark_configuration_hash(configs)
        or completeness.get("evaluation_request_hash")
        != configured_evaluation_request_hash(configs)
        or completeness.get("scope") != expected_scope
    ):
        raise RuntimeError(
            "Published summaries do not match the current experiment semantics"
        )
    return bundle


def file_sha256(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def recorded_dataset_id(record: Mapping[str, Any]) -> Any:
    dataset = record.get("dataset")
    if isinstance(dataset, Mapping):
        return dataset.get("dataset_id")
    return dataset


def atomic_save_npy(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npy")
    try:
        np.save(temporary, value)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_dataset_path(entry: Mapping[str, Any], paths: ExperimentPaths) -> Path:
    raw = Path(str(entry.get("path") or entry_name(entry)))
    if raw.is_absolute():
        return raw
    processed_candidate = paths.processed_root / raw
    if processed_candidate.exists():
        return processed_candidate
    return processed_candidate


def load_dataset(
    entry: Mapping[str, Any], paths: ExperimentPaths, *, mmap: bool
) -> tuple[str, np.ndarray, ProcessedDataset, dict[str, Any]]:
    dataset = load_processed_dataset(resolve_dataset_path(entry, paths), processed_root=paths.processed_root)
    features = dataset.load_features(
        mmap=mmap,
        dtype=None if mmap else np.float32,
        order=None if mmap else "C",
    )
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D after flattening: {dataset.features_path}")
    metadata = {
        "dataset_id": dataset.dataset_id,
        "dataset_dir": repo_relative(dataset.dataset_dir),
        "features_path": repo_relative(dataset.features_path),
        "feature_shape": [int(value) for value in features.shape],
        "feature_dtype": str(np.dtype(np.float32)),
        "config_entry": dict(entry),
    }
    return dataset.dataset_id, features, dataset, metadata


def dataset_entries(configs: Mapping[str, Any], filters: Sequence[str] | None = None) -> list[Mapping[str, Any]]:
    entries = list((configs.get("datasets") or {}).get("datasets") or [])
    return filter_entries(entries, filters, kind="dataset")


def algorithm_entries(
    configs: Mapping[str, Any],
    filters: Sequence[str] | None = None,
    *,
    profiles: Sequence[str] | None = None,
    include_disabled: bool = False,
) -> list[Mapping[str, Any]]:
    entries = list((configs.get("algorithms") or {}).get("algorithms") or [])
    if not include_disabled:
        entries = [entry for entry in entries if bool(entry.get("enabled", True))]
    entries = filter_entries(entries, filters, kind="algorithm")
    if profiles:
        wanted = set(profiles)
        unknown = wanted - {"seeded", "unseeded"}
        if unknown:
            raise ValueError(f"Unknown execution profile(s): {', '.join(sorted(unknown))}")
        entries = [entry for entry in entries if entry.get("execution_profile") in wanted]
    return entries


def common_params(configs: Mapping[str, Any], *, n_neighbors: int | None = None) -> dict[str, Any]:
    params = dict((configs.get("algorithms") or {}).get("common_umap_params") or {})
    if n_neighbors is not None:
        params["n_neighbors"] = int(n_neighbors)
    return params


def n_neighbors_values(
    configs: Mapping[str, Any], override: Sequence[int] | None = None
) -> list[int]:
    if override:
        return [int(value) for value in override]
    return [int(common_params(configs).get("n_neighbors", 15))]


def repeat_values(configs: Mapping[str, Any], override: Sequence[int] | None = None) -> list[int]:
    if override:
        values = sorted({int(value) for value in override})
    else:
        values = list(range(1, int((configs.get("experiment") or {}).get("repeats", 1)) + 1))
    if not values or min(values) < 1:
        raise ValueError("Repeat indices must be positive integers")
    return values


def algorithm_params(
    configs: Mapping[str, Any], entry: Mapping[str, Any], *, n_neighbors: int
) -> dict[str, Any]:
    params = common_params(configs, n_neighbors=n_neighbors)
    params.update(dict(entry.get("params") or {}))
    return params


def task_signature(
    configs: Mapping[str, Any],
    task: BenchmarkTask,
    *,
    feature_sha256: str | None,
    protocol_hash: str,
) -> str:
    return canonical_hash(
        {
            "protocol_hash": protocol_hash,
            "dataset_id": task.dataset_id,
            "algorithm_id": task.algorithm_id,
            "algorithm_entry": task.algorithm_entry,
            "repeat": int(task.repeat),
            "n_neighbors": int(task.n_neighbors),
            "requested_params": algorithm_params(
                configs,
                task.algorithm_entry,
                n_neighbors=task.n_neighbors,
            ),
            "feature_sha256": feature_sha256,
        }
    )


def requires_cuda_graph_extension(
    configs: Mapping[str, Any],
    algorithm_entry: Mapping[str, Any],
) -> bool:
    requirements = (
        (configs.get("experiment") or {}).get("environment_requirements") or {}
    )
    return bool(requirements.get("require_cuml_graph_extension_for_cuda_ibumap", True)) and (
        str(algorithm_entry.get("family")) == "ibumap"
        and str(algorithm_entry.get("device")) == "cuda"
    )


def cuda_graph_extension_status() -> dict[str, Any]:
    try:
        from ibumap.graph import _cuml_graph_ext  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "module": getattr(_cuml_graph_ext, "__name__", "ibumap.graph._cuml_graph_ext"),
        "module_file": getattr(_cuml_graph_ext, "__file__", None),
    }


def enforce_required_backend(
    configs: Mapping[str, Any],
    algorithm_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if not requires_cuda_graph_extension(configs, algorithm_entry):
        return {"required": False, "available": None}
    if os.environ.get("IBUMAP_DISABLE_CUML_GRAPH_EXT") == "1":
        raise RuntimeError(
            "CUDA ibUMAP requires the cuML graph extension, but "
            "IBUMAP_DISABLE_CUML_GRAPH_EXT=1 is set"
        )
    os.environ["IBUMAP_REQUIRE_CUML_GRAPH_EXT"] = "1"
    status = cuda_graph_extension_status()
    if not status.get("available"):
        raise RuntimeError(
            "CUDA ibUMAP requires the cuML graph extension, but its runtime "
            f"preflight failed: {status.get('error')}"
        )
    return {"required": True, **status}


def validate_configs(configs: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    datasets = list((configs.get("datasets") or {}).get("datasets") or [])
    algorithm_config = configs.get("algorithms") or {}
    common = dict(algorithm_config.get("common_umap_params") or {})
    algorithms = list(algorithm_config.get("algorithms") or [])
    if common.get("n_epochs") != 200:
        issues.append("algorithms.common_umap_params.n_epochs must remain 200")
    if "fixed_inputs" in algorithm_config:
        issues.append("algorithms.fixed_inputs is not used in the end-to-end benchmark")
    for kind, entries in (("dataset", datasets), ("algorithm", algorithms)):
        names = [entry_name(entry) for entry in entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            issues.append(f"Duplicate {kind} names: {', '.join(duplicates)}")
    if int((configs.get("experiment") or {}).get("repeats", 0)) < 1:
        issues.append("experiment.repeats must be >= 1")
    configured_protocol = str(
        (configs.get("experiment") or {}).get("protocol_version") or ""
    )
    if configured_protocol != PROTOCOL_VERSION:
        issues.append(
            f"experiment.protocol_version must be {PROTOCOL_VERSION!r}, "
            f"got {configured_protocol!r}"
        )
    if int((configs.get("experiment") or {}).get("warmup_runs", 0)) < 0:
        issues.append("experiment.warmup_runs must be >= 0")
    for entry in algorithms:
        name = entry_name(entry)
        params = dict(entry.get("params") or {})
        seeded = entry.get("seed_mode") == "fixed"
        expected_profile = "seeded" if seeded else "unseeded"
        if entry.get("execution_profile") != expected_profile:
            issues.append(f"{name}: execution_profile disagrees with seed_mode")
        random_state = params.get("random_state")
        if seeded != (random_state is not None):
            issues.append(f"{name}: seed_mode disagrees with random_state")
        family = str(entry.get("family"))
        if family in {"umap_learn", "cuml_umap"} and bool(
            entry.get("baseline", False)
        ):
            extra_baseline_params = sorted(set(params) - {"random_state"})
            if extra_baseline_params:
                issues.append(
                    f"{name}: standard baseline may only override random_state; "
                    f"got {', '.join(extra_baseline_params)}"
                )
        if family in {"ibumap", "torchdr_umap"}:
            if bool(params.get("deterministic", False)) != seeded:
                issues.append(f"{name}: deterministic must match the seeded profile")
        elif "deterministic" in params:
            issues.append(f"{name}: deterministic is only an ibUMAP or TorchDR parameter")
        if family == "torchdr_umap":
            if str(entry.get("device")) != "cuda":
                issues.append(f"{name}: TorchDR variants must use CUDA")
            if params.get("backend") != "faiss":
                issues.append(f"{name}: TorchDR backend must be explicitly set to faiss")
            if params.get("device") != "cuda":
                issues.append(f"{name}: TorchDR runtime device must be explicitly set to cuda")
            if params.get("distributed") is not False:
                issues.append(f"{name}: distributed must be explicitly false")
            if params.get("compile") is not False:
                issues.append(f"{name}: compile must be explicitly false")
        if family not in {"umap_learn", "cuml_umap", "ibumap", "torchdr_umap"}:
            issues.append(f"{name}: unsupported family {family!r}")
    return issues


def is_resource_skipped(dataset_id: str, algorithm_id: str) -> bool:
    return (
        dataset_id in RESOURCE_SKIP_DATASETS
        and algorithm_id in RESOURCE_SKIP_ALGORITHMS
    )


def resource_skipped_task_count(
    datasets: Sequence[Mapping[str, Any]],
    algorithms: Sequence[Mapping[str, Any]],
    repeats: Sequence[int],
) -> int:
    skipped_pairs = sum(
        is_resource_skipped(entry_name(dataset), entry_name(algorithm))
        for dataset in datasets
        for algorithm in algorithms
    )
    return int(skipped_pairs * len(repeats))


def build_tasks(
    configs: Mapping[str, Any],
    datasets: Sequence[Mapping[str, Any]],
    algorithms: Sequence[Mapping[str, Any]],
    repeats: Sequence[int],
    *,
    n_neighbors: int,
) -> list[BenchmarkTask]:
    order_seed = int((configs.get("experiment") or {}).get("order_seed", 20260630))
    shuffle = bool((configs.get("experiment") or {}).get("shuffle_within_repeat_blocks", True))
    tasks: list[BenchmarkTask] = []
    execution_index = 0
    for repeat in repeats:
        for dataset in datasets:
            dataset_id = entry_name(dataset)
            ordered = list(algorithms)
            if shuffle:
                seed_material = f"{order_seed}:{repeat}:{dataset_id}".encode("utf-8")
                local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
                np.random.default_rng(local_seed).shuffle(ordered)
            for algorithm in ordered:
                algorithm_id = entry_name(algorithm)
                if is_resource_skipped(dataset_id, algorithm_id):
                    continue
                execution_index += 1
                tasks.append(
                    BenchmarkTask(
                        dataset_id=dataset_id,
                        algorithm_id=algorithm_id,
                        repeat=int(repeat),
                        n_neighbors=int(n_neighbors),
                        execution_index=execution_index,
                        algorithm_entry=algorithm,
                    )
                )
    return tasks


def embedding_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "per_run" / task.dataset_id / f"{task.run_id}.npy"


def run_record_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "metadata" / task.dataset_id / f"{task.run_id}.json"


def run_error_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "failures" / task.dataset_id / f"{task.run_id}.error.json"


def evaluation_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.evaluation_root / "per_run" / task.dataset_id / f"{task.run_id}.json"


def evaluation_error_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.evaluation_root / "failures" / task.dataset_id / f"{task.run_id}.error.json"


def array_stats(array: Any) -> dict[str, Any]:
    values = np.asarray(array)
    finite = bool(np.isfinite(values).all())
    result: dict[str, Any] = {
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "finite": finite,
    }
    if values.size and finite:
        result.update(
            {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values, dtype=np.float64)),
                "l2_norm": float(np.linalg.norm(values)),
            }
        )
    return result


def feature_stats(array: Any) -> dict[str, Any]:
    values = np.asarray(array)
    result: dict[str, Any] = {
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "finite": True,
    }
    if not values.size:
        return result

    # Full-array np.isfinite/nanmean calls can allocate masks and copies as
    # large as the dataset. Keep temporary memory bounded while collecting all
    # statistics in a single pass.
    target_chunk_bytes = 64 * 1024 * 1024
    if values.ndim == 0:
        chunks = (values.reshape(1),)
    else:
        row_bytes = max(1, int(values[:1].nbytes))
        rows_per_chunk = max(1, target_chunk_bytes // row_bytes)
        chunks = (
            values[start : start + rows_per_chunk]
            for start in range(0, int(values.shape[0]), rows_per_chunk)
        )

    finite_count = 0
    finite_sum = 0.0
    finite_min: float | None = None
    finite_max: float | None = None
    for chunk in chunks:
        finite_mask = np.isfinite(chunk)
        chunk_is_finite = bool(finite_mask.all())
        result["finite"] = bool(result["finite"] and chunk_is_finite)
        finite_values = chunk if chunk_is_finite else chunk[finite_mask]
        if not finite_values.size:
            continue
        chunk_min = float(np.min(finite_values))
        chunk_max = float(np.max(finite_values))
        finite_min = chunk_min if finite_min is None else min(finite_min, chunk_min)
        finite_max = chunk_max if finite_max is None else max(finite_max, chunk_max)
        finite_sum += float(np.sum(finite_values, dtype=np.float64))
        finite_count += int(finite_values.size)

    result.update(
        {
            "min": finite_min,
            "max": finite_max,
            "mean": finite_sum / finite_count if finite_count else None,
        }
    )
    return result


def embedding_sanity(embedding: Any, *, expected_rows: int) -> dict[str, Any]:
    values = np.asarray(embedding)
    result = array_stats(values)
    result["expected_rows"] = int(expected_rows)
    result["row_count_matches"] = bool(values.ndim == 2 and values.shape[0] == expected_rows)
    if values.ndim == 2 and values.shape[1] >= 2 and values.shape[0] > 0 and result["finite"]:
        result.update(bbox_summary(values[:, :2]))
    return result


def resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes: int | None = None
    try:
        statm_fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        rss_bytes = int(statm_fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError):
        try:
            import psutil  # type: ignore

            rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    return {
        "ru_maxrss": int(usage.ru_maxrss),
        "rss_bytes": rss_bytes,
        "ru_utime": float(usage.ru_utime),
        "ru_stime": float(usage.ru_stime),
    }


def gpu_memory_snapshot() -> dict[str, Any]:
    try:
        import cupy as cp

        synchronize_gpu()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {
            "available": True,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }
    except Exception as cupy_exc:
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "available": False,
                    "cupy_error": f"{type(cupy_exc).__name__}: {cupy_exc}",
                    "torch_error": "CUDA-enabled PyTorch is unavailable",
                }
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {
                "available": True,
                "backend": "torch",
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "cupy_error": f"{type(cupy_exc).__name__}: {cupy_exc}",
            }
        except Exception as torch_exc:
            return {
                "available": False,
                "cupy_error": f"{type(cupy_exc).__name__}: {cupy_exc}",
                "torch_error": f"{type(torch_exc).__name__}: {torch_exc}",
            }


def environment_snapshot() -> dict[str, Any]:
    machine = machine_info()
    return {
        "created_at": now_utc(),
        "platform": machine["platform"],
        "python_version": machine["python_version"],
        "machine": machine["machine"],
        "cpu_model": machine["cpu_model"],
        "logical_cpus": machine["logical_cpus"],
        "memory_total_bytes": machine["memory_total_bytes"],
        "git_commit": git_commit(REPO_ROOT),
        "runtime_env": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
        "packages": package_versions(
            (
                "numpy",
                "scipy",
                "scikit-learn",
                "umap-learn",
                "pynndescent",
                "joblib",
                "llvmlite",
                "numba",
                "pyfftw",
                "threadpoolctl",
                "cupy",
                "cupy-cuda12x",
                "cuml",
                "cuml-cu12",
                "cuvs",
                "cuvs-cu12",
                "rmm",
                "rmm-cu12",
                "torch",
                "torchdr",
                "faiss-cpu",
                "faiss-gpu",
            )
        ),
        "gpu": machine["gpu"],
    }


def _ignored_params(params: Mapping[str, Any], conceptual_used: set[str]) -> list[str]:
    return sorted(key for key in params if key not in conceptual_used)


def clean_runtime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in params.items() if value is not None}


def callable_exposes_parameter(callable_obj: Any, parameter: str) -> bool | None:
    """Report explicit public parameter support without changing call semantics."""
    try:
        return parameter in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return None


def coerce_time_costs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is None or isinstance(item, (str, bool)):
            result[str(key)] = item
            continue
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            result[str(key)] = item
    return result


def run_umap_learn_e2e(X: Any, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import umap  # type: ignore

    raw_seed = params.get("random_state")
    kwargs = {
        "n_neighbors": int(params.get("n_neighbors", 15)),
        "n_components": int(params.get("n_components", 2)),
        "metric": str(params.get("metric", "euclidean")),
        "metric_kwds": params.get("metric_kwds") or None,
        "min_dist": float(params.get("min_dist", 0.1)),
        "spread": float(params.get("spread", 1.0)),
        "n_epochs": int(params.get("n_epochs", 200)),
        "learning_rate": float(params.get("learning_rate", 1.0)),
        "repulsion_strength": float(params.get("repulsion_strength", 1.0)),
        "negative_sample_rate": int(params.get("negative_sample_rate", 5)),
        "set_op_mix_ratio": float(params.get("set_op_mix_ratio", 1.0)),
        "local_connectivity": float(params.get("local_connectivity", 1.0)),
        "low_memory": bool(params.get("low_memory", True)),
        "random_state": raw_seed,
        "n_jobs": int(params.get("n_jobs", -1)),
        "verbose": bool(params.get("verbose", False)),
        "init": params.get("init", "spectral"),
        "densmap": False,
        "output_dens": False,
    }
    effective_kwargs = filtered_kwargs(umap.UMAP, clean_runtime_params(kwargs))
    model = umap.UMAP(**effective_kwargs)
    embedding = model.fit_transform(X)
    conceptual_used = {
        "n_neighbors",
        "n_components",
        "metric",
        "metric_kwds",
        "min_dist",
        "spread",
        "n_epochs",
        "learning_rate",
        "repulsion_strength",
        "negative_sample_rate",
        "set_op_mix_ratio",
        "local_connectivity",
        "low_memory",
        "random_state",
        "n_jobs",
        "verbose",
        "init",
    }
    return np.asarray(embedding, dtype=np.float32, order="C"), {
        "effective_params": {
            **effective_kwargs,
            "parallel": raw_seed is None,
            "resolved_n_jobs": getattr(model, "n_jobs", effective_kwargs.get("n_jobs")),
        },
        "ignored_or_unsupported_params": _ignored_params(params, conceptual_used),
        "time_costs": {},
    }


def run_cuml_e2e(X: Any, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import cupy as cp
    from cuml.manifold import UMAP as CumlUMAP  # type: ignore

    raw_seed = params.get("random_state")
    force_serial_requested = params.get("force_serial_epochs")
    kwargs = {
        "n_neighbors": int(params.get("n_neighbors", 15)),
        "n_components": int(params.get("n_components", 2)),
        "metric": str(params.get("metric", "euclidean")),
        "min_dist": float(params.get("min_dist", 0.1)),
        "spread": float(params.get("spread", 1.0)),
        "n_epochs": int(params.get("n_epochs", 200)),
        "learning_rate": float(params.get("learning_rate", 1.0)),
        "repulsion_strength": float(params.get("repulsion_strength", 1.0)),
        "negative_sample_rate": int(params.get("negative_sample_rate", 5)),
        "set_op_mix_ratio": float(params.get("set_op_mix_ratio", 1.0)),
        "local_connectivity": float(params.get("local_connectivity", 1.0)),
        "random_state": raw_seed,
        "verbose": bool(params.get("verbose", False)),
        "output_type": "numpy",
    }
    if force_serial_requested is not None:
        kwargs["force_serial_epochs"] = bool(force_serial_requested)
    if params.get("metric_kwds"):
        kwargs["metric_kwds"] = params.get("metric_kwds")
    effective_kwargs = filtered_kwargs(CumlUMAP, clean_runtime_params(kwargs))
    force_serial_supported = callable_exposes_parameter(
        CumlUMAP,
        "force_serial_epochs",
    )
    force_serial_passed = "force_serial_epochs" in effective_kwargs
    model = CumlUMAP(**effective_kwargs)
    synchronize_gpu()
    embedding = model.fit_transform(cp.asarray(X, dtype=cp.float32, order="C"))
    synchronize_gpu()
    conceptual_used = {
        "n_neighbors",
        "n_components",
        "metric",
        "metric_kwds",
        "min_dist",
        "spread",
        "n_epochs",
        "learning_rate",
        "repulsion_strength",
        "negative_sample_rate",
        "set_op_mix_ratio",
        "local_connectivity",
        "random_state",
        "force_serial_epochs",
        "verbose",
    }
    return to_numpy_array(embedding).astype(np.float32, copy=False), {
        "effective_params": {
            "random_state": raw_seed,
            "random_state_keyword": "passed" if raw_seed is not None else "omitted",
            "seed_configured": raw_seed is not None,
            "force_serial_epochs_requested": force_serial_requested,
            "force_serial_epochs_keyword": (
                "passed" if force_serial_passed else "omitted"
            ),
            "force_serial_epochs_resolution": (
                "explicit" if force_serial_passed else "cuml_default"
            ),
            "force_serial_epochs_supported": force_serial_supported,
            "force_serial_epochs_applied": (
                effective_kwargs["force_serial_epochs"]
                if force_serial_passed
                else None
            ),
            "used_constructor_kwargs": effective_kwargs,
        },
        "ignored_or_unsupported_params": _ignored_params(params, conceptual_used),
        "time_costs": {},
    }


def run_ibumap_e2e(
    X: Any, params: Mapping[str, Any], *, device: str
) -> tuple[np.ndarray, dict[str, Any]]:
    from common.algorithm_adapters import run_ibumap

    embedding, extras = run_ibumap(X, params, algorithm="ibumap", device=device)
    extras = dict(extras)
    extras["effective_params"] = dict(extras.pop("used_params", {}))
    extras["ignored_or_unsupported_params"] = []
    extras["time_costs"] = coerce_time_costs(extras.get("time_costs"))
    return np.asarray(embedding, dtype=np.float32, order="C"), extras


def run_torchdr_e2e(
    X: Any, params: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    from common.algorithm_adapters import run_torchdr_umap

    embedding, extras = run_torchdr_umap(X, params)
    extras = dict(extras)
    used = dict(extras.pop("used_params", {}))
    effective = {
        **used,
        "deterministic": extras.get("deterministic"),
        "torch_runtime": extras.get("torch_runtime"),
    }
    conceptual_used = set(used) | {
        "n_epochs",
        "learning_rate",
        "random_state",
        "deterministic",
    }
    extras["effective_params"] = effective
    extras["ignored_or_unsupported_params"] = _ignored_params(
        params,
        conceptual_used,
    )
    extras["time_costs"] = {}
    return np.asarray(embedding, dtype=np.float32, order="C"), extras


def run_algorithm(
    X: Any,
    entry: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    family = str(entry.get("family"))
    if family == "umap_learn":
        return run_umap_learn_e2e(X, params)
    if family == "cuml_umap":
        return run_cuml_e2e(X, params)
    if family == "ibumap":
        return run_ibumap_e2e(X, params, device=str(entry.get("device", "cpu")))
    if family == "torchdr_umap":
        return run_torchdr_e2e(X, params)
    raise ValueError(f"Unsupported end-to-end family: {family}")


def warmup_subset(X: np.ndarray, *, max_rows: int = 512) -> np.ndarray:
    n = min(int(X.shape[0]), int(max_rows))
    if n < 3:
        raise ValueError("Warmup requires at least three rows")
    return np.asarray(X[:n], dtype=np.float32, order="C")


def failure_payload(
    *,
    stage: str,
    dataset_id: str | None,
    run_id: str | None,
    error: BaseException,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "dataset": dataset_id,
        "run_id": run_id,
        "created_at": now_utc(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "metadata": dict(metadata or {}),
    }


def setup_log_path(paths: ExperimentPaths, stage: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return paths.logs_root / f"{stage}_{stamp}.log"
