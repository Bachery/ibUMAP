"""Shared contracts of the interpolation-order sweep (paper appendix, tab:mechanism-psweep).

Eight optimizer variants (CPU/CUDA x fixed p = 1, 2, 3 and the p1->p2->p3
schedule) run ``IBUMAP.optimize_from_graph`` on one fixed kNN graph and spectral
initialization per dataset, with three paired seeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.paths import ensure_ibumap_importable  # noqa: E402

ensure_ibumap_importable()

from common.config import load_yaml, resolve_path  # noqa: E402
from common.dataset_io import ProcessedDataset, load_processed_dataset  # noqa: E402
from common.fixed_inputs import fixed_input_paths  # noqa: E402
from common.gpu_runtime import machine_info  # noqa: E402
from common.run_metadata import (  # noqa: E402
    git_commit,
    json_default,
    now_utc,
    package_versions,
    write_json,
)


CONFIG_NAMES = ("paths", "datasets", "algorithms", "evaluation", "experiment")
VARIANT_ORDER = ("p1", "p2", "p3", "schedule")
DEVICE_ORDER = ("cpu", "cuda")


@dataclass(frozen=True)
class ExperimentPaths:
    processed_root: Path
    catalog_path: Path
    results_root: Path
    fixed_inputs_root: Path
    embeddings_root: Path
    evaluations_root: Path
    summaries_root: Path
    plots_root: Path
    logs_root: Path


@dataclass(frozen=True)
class BenchmarkTask:
    dataset_entry: Mapping[str, Any]
    algorithm_entry: Mapping[str, Any]
    repeat: int
    seed: int
    execution_index: int

    @property
    def dataset_id(self) -> str:
        return str(self.dataset_entry["name"])

    @property
    def algorithm_id(self) -> str:
        return str(self.algorithm_entry["name"])

    @property
    def device(self) -> str:
        return str(self.algorithm_entry["device"])

    @property
    def variant(self) -> str:
        return str(self.algorithm_entry["variant"])

    @property
    def run_id(self) -> str:
        return f"{self.algorithm_id}__repeat_{self.repeat:02d}"


def load_configs(config_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(config_dir or EXPERIMENT_ROOT / "configs").resolve()
    return {name: load_yaml(root / f"{name}.yaml") or {} for name in CONFIG_NAMES}


def _path(config: Mapping[str, Any], key: str, default: Path, config_dir: Path) -> Path:
    value = resolve_path(config.get(key), base_dir=config_dir)
    return (value or default).resolve()


def experiment_paths(
    configs: Mapping[str, Any], config_dir: str | Path | None = None
) -> ExperimentPaths:
    root = Path(config_dir or EXPERIMENT_ROOT / "configs").resolve()
    config = configs.get("paths") or {}
    results = _path(config, "results_root", EXPERIMENT_ROOT / "results", root)
    return ExperimentPaths(
        processed_root=_path(config, "processed_root", REPO_ROOT / "datasets" / "processed", root),
        catalog_path=_path(config, "catalog_path", REPO_ROOT / "datasets" / "catalog.json", root),
        results_root=results,
        fixed_inputs_root=_path(config, "fixed_inputs_root", results / "01_fixed_inputs", root),
        embeddings_root=_path(config, "embeddings_root", results / "02_embeddings", root),
        evaluations_root=_path(config, "evaluations_root", results / "03_evaluations", root),
        summaries_root=_path(config, "summaries_root", results / "04_summaries", root),
        plots_root=_path(config, "plots_root", results / "05_plots", root),
        logs_root=_path(config, "logs_root", EXPERIMENT_ROOT / "logs", root),
    )


def ensure_directories(paths: ExperimentPaths) -> None:
    directories = (
        paths.fixed_inputs_root,
        paths.embeddings_root / "per_run",
        paths.embeddings_root / "metadata",
        paths.embeddings_root / "failures",
        paths.embeddings_root / "memory",
        paths.evaluations_root / "per_run",
        paths.evaluations_root / "failures",
        paths.evaluations_root / "sample_indices",
        paths.evaluations_root / "metric_cache",
        paths.summaries_root,
        paths.plots_root,
        paths.logs_root,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(EXPERIMENT_ROOT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT_ROOT / ".mplconfig"))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configuration_hash(configs: Mapping[str, Any]) -> str:
    return canonical_hash({name: configs.get(name) for name in CONFIG_NAMES})


def file_sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_save_npy(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.npy")
    try:
        np.save(temporary, value)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.npz")
    try:
        sparse.save_npz(temporary, value)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            if columns:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_entries(
    configs: Mapping[str, Any], filters: Sequence[str] | None = None
) -> list[Mapping[str, Any]]:
    entries = list((configs.get("datasets") or {}).get("datasets") or [])
    if filters:
        wanted = set(filters)
        entries = [entry for entry in entries if str(entry.get("name")) in wanted]
        missing = wanted - {str(entry.get("name")) for entry in entries}
        if missing:
            raise ValueError(f"Unknown dataset(s): {', '.join(sorted(missing))}")
    return entries


def algorithm_entries(
    configs: Mapping[str, Any],
    filters: Sequence[str] | None = None,
    *,
    devices: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    entries = [
        entry
        for entry in list((configs.get("algorithms") or {}).get("algorithms") or [])
        if bool(entry.get("enabled", True))
    ]
    if filters:
        wanted = set(filters)
        entries = [entry for entry in entries if str(entry.get("name")) in wanted]
        missing = wanted - {str(entry.get("name")) for entry in entries}
        if missing:
            raise ValueError(f"Unknown algorithm(s): {', '.join(sorted(missing))}")
    if devices:
        wanted_devices = set(devices)
        entries = [entry for entry in entries if str(entry.get("device")) in wanted_devices]
    if variants:
        wanted_variants = set(variants)
        entries = [entry for entry in entries if str(entry.get("variant")) in wanted_variants]
    return entries


def repeat_seeds(configs: Mapping[str, Any]) -> list[int]:
    values = list((configs.get("experiment") or {}).get("repeat_seeds") or [])
    if not values:
        raise ValueError("experiment.repeat_seeds must contain at least two seeds")
    return [int(value) for value in values]


def common_umap_params(configs: Mapping[str, Any]) -> dict[str, Any]:
    return dict((configs.get("algorithms") or {}).get("common_umap_params") or {})


def fixed_input_params(configs: Mapping[str, Any]) -> dict[str, Any]:
    params = common_umap_params(configs)
    params.update(dict((configs.get("algorithms") or {}).get("fixed_inputs") or {}))
    params["random_state"] = int(
        params.get("random_state", (configs.get("experiment") or {}).get("fixed_input_seed", 42))
    )
    return params


def build_tasks(
    configs: Mapping[str, Any],
    datasets: Sequence[Mapping[str, Any]],
    algorithms: Sequence[Mapping[str, Any]],
    repeats: Sequence[int] | None = None,
) -> list[BenchmarkTask]:
    seeds = repeat_seeds(configs)
    selected = list(repeats or range(1, len(seeds) + 1))
    if not selected or min(selected) < 1 or max(selected) > len(seeds):
        raise ValueError(f"Repeat indices must be between 1 and {len(seeds)}")
    experiment = configs.get("experiment") or {}
    shuffle = bool(experiment.get("shuffle_variants_within_dataset_repeat", True))
    order_seed = int(experiment.get("order_seed", 20260716))
    tasks: list[BenchmarkTask] = []
    index = 0
    for repeat in selected:
        for dataset in datasets:
            ordered = list(algorithms)
            if shuffle:
                material = f"{order_seed}:{repeat}:{dataset['name']}".encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
                np.random.default_rng(seed).shuffle(ordered)
            for algorithm in ordered:
                index += 1
                tasks.append(
                    BenchmarkTask(
                        dataset_entry=dataset,
                        algorithm_entry=algorithm,
                        repeat=int(repeat),
                        seed=seeds[int(repeat) - 1],
                        execution_index=index,
                    )
                )
    return tasks


def task_lookup(
    configs: Mapping[str, Any], dataset_id: str, algorithm_id: str, repeat: int
) -> BenchmarkTask:
    tasks = build_tasks(
        configs,
        dataset_entries(configs, [dataset_id]),
        algorithm_entries(configs, [algorithm_id]),
        [repeat],
    )
    if len(tasks) != 1:
        raise RuntimeError("Expected exactly one worker task")
    return tasks[0]


def load_dataset(
    entry: Mapping[str, Any], paths: ExperimentPaths, *, mmap: bool
) -> tuple[str, np.ndarray, ProcessedDataset]:
    raw = Path(str(entry.get("path") or entry["name"]))
    dataset = load_processed_dataset(raw, processed_root=paths.processed_root)
    values = dataset.load_features(
        mmap=mmap,
        dtype=None if mmap else np.float32,
        order=None if mmap else "C",
    )
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D features for {dataset.dataset_id}, got {values.shape}")
    return dataset.dataset_id, values, dataset


def fixed_identity(configs: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    params = fixed_input_params(configs)
    return {
        "dataset": str(entry["name"]),
        "expected_rows": int(entry["expected_rows"]),
        "params": params,
        "identity_hash": canonical_hash(
            {"dataset": str(entry["name"]), "expected_rows": int(entry["expected_rows"]), "params": params}
        ),
    }


def fixed_bundle_paths(
    configs: Mapping[str, Any], paths: ExperimentPaths, dataset_id: str
) -> dict[str, Path]:
    n_neighbors = int(fixed_input_params(configs).get("n_neighbors", 15))
    return fixed_input_paths(paths.fixed_inputs_root, dataset_id, n_neighbors)


def load_fixed_bundle(
    configs: Mapping[str, Any], paths: ExperimentPaths, entry: Mapping[str, Any]
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    files = fixed_bundle_paths(configs, paths, str(entry["name"]))
    for key in ("fuzzy_graph", "init_embedding", "metadata"):
        if not files[key].exists():
            raise FileNotFoundError(f"Missing fixed input {files[key]}; run 01_prepare_fixed_inputs.py --run")
    metadata = read_json(files["metadata"])
    expected = fixed_identity(configs, entry)
    if metadata.get("identity_hash") != expected["identity_hash"]:
        raise ValueError(f"Stale fixed inputs for {entry['name']}; regenerate with --overwrite")
    graph = sparse.load_npz(files["fuzzy_graph"]).tocsr().astype(np.float32, copy=False)
    init = np.asarray(np.load(files["init_embedding"]), dtype=np.float32, order="C")
    if graph.shape != (int(entry["expected_rows"]), int(entry["expected_rows"])):
        raise ValueError(f"Unexpected graph shape for {entry['name']}: {graph.shape}")
    if init.shape != (int(entry["expected_rows"]), 2):
        raise ValueError(f"Unexpected initialization shape for {entry['name']}: {init.shape}")
    return graph, init, metadata


def embedding_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "per_run" / task.dataset_id / f"{task.run_id}.npy"


def run_record_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "metadata" / task.dataset_id / f"{task.run_id}.json"


def run_error_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "failures" / task.dataset_id / f"{task.run_id}.json"


def memory_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.embeddings_root / "memory" / task.dataset_id / f"{task.run_id}.jsonl"


def evaluation_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.evaluations_root / "per_run" / task.dataset_id / f"{task.run_id}.json"


def evaluation_error_path(paths: ExperimentPaths, task: BenchmarkTask) -> Path:
    return paths.evaluations_root / "failures" / task.dataset_id / f"{task.run_id}.json"


def fft_config_for(entry: Mapping[str, Any]) -> Any:
    from ibumap import FFTConfig

    spec = dict(entry.get("fft") or {})
    return FFTConfig(**spec)


def optimizer_params(
    configs: Mapping[str, Any], task: BenchmarkTask, *, diagnostics_path: Path | None
) -> dict[str, Any]:
    params = common_umap_params(configs)
    params.update(dict(task.algorithm_entry.get("params") or {}))
    params["random_state"] = int(task.seed)
    params["fft"] = fft_config_for(task.algorithm_entry)
    if diagnostics_path is not None:
        memory = (configs.get("experiment") or {}).get("memory") or {}
        params["diagnostics_memory_path"] = str(diagnostics_path)
        params["diagnostics_memory_epoch_stride"] = int(memory.get("epoch_stride", 10))
        params["diagnostics_memory_sample_interval_ms"] = float(
            memory.get("sample_interval_ms", 20)
        )
    return params


def fft_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    config = fft_config_for(entry)
    schedule = config.interpolation_schedule
    return {
        "n_interpolation_points": int(config.n_interpolation_points),
        "p2m_mode": str(config.p2m_mode),
        "combine_stages": bool(config.combine_stages),
        "interpolation_schedule": (
            None
            if schedule is None
            else [
                {
                    "start_fraction": float(stage.start_fraction),
                    "n_interpolation_points": int(stage.n_interpolation_points),
                }
                for stage in schedule
            ]
        ),
    }


def parse_memory_diagnostics(path: str | Path, device: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)

    def values(*keys: str) -> list[int]:
        output: list[int] = []
        for row in rows:
            current: Any = row
            for key in keys:
                current = current.get(key) if isinstance(current, Mapping) else None
            if isinstance(current, (int, float)) and np.isfinite(current):
                output.append(int(current))
        return output

    def peak_delta(series: Sequence[int]) -> tuple[int | None, int | None, int | None]:
        if not series:
            return None, None, None
        baseline = int(series[0])
        peak = int(max(series))
        return baseline, peak, peak - baseline

    cpu_series = values("cpu", "rss_bytes")
    cpu_source = "rss_bytes"
    if not cpu_series:
        cpu_series = values("cpu", "hwm_bytes")
        cpu_source = "hwm_bytes"
    if not cpu_series:
        cpu_series = values("cpu", "ru_maxrss_bytes")
        cpu_source = "ru_maxrss_bytes"
    cpu_baseline, cpu_peak, cpu_delta = peak_delta(cpu_series)

    gpu_process = values("gpu", "nvml_process_used_bytes")
    gpu_pool_used = values("gpu", "cupy_pool_used_bytes")
    gpu_pool_total = values("gpu", "cupy_pool_total_bytes")
    gpu_device_used: list[int] = []
    totals = values("gpu", "cuda_total_bytes")
    frees = values("gpu", "cuda_free_bytes")
    if totals and len(totals) == len(frees):
        gpu_device_used = [total - free for total, free in zip(totals, frees)]
    process_baseline, process_peak, process_delta = peak_delta(gpu_process)
    pool_used_baseline, pool_used_peak, pool_used_delta = peak_delta(gpu_pool_used)
    pool_total_baseline, pool_total_peak, pool_total_delta = peak_delta(gpu_pool_total)
    device_baseline, device_peak, device_delta = peak_delta(gpu_device_used)

    if device == "cpu":
        primary_source = f"cpu.{cpu_source}"
        primary_baseline, primary_peak, primary_delta = cpu_baseline, cpu_peak, cpu_delta
    elif gpu_process:
        primary_source = "gpu.nvml_process_used_bytes"
        primary_baseline, primary_peak, primary_delta = process_baseline, process_peak, process_delta
    else:
        primary_source = "gpu.cupy_pool_total_bytes"
        primary_baseline, primary_peak, primary_delta = (
            pool_total_baseline,
            pool_total_peak,
            pool_total_delta,
        )
    return {
        "samples": len(rows),
        "duration_s": (
            float(rows[-1].get("time_since_start_s", 0.0)) if rows else None
        ),
        "primary_source": primary_source,
        "primary_baseline_bytes": primary_baseline,
        "primary_peak_bytes": primary_peak,
        "primary_peak_delta_bytes": primary_delta,
        "cpu_source": cpu_source,
        "cpu_baseline_bytes": cpu_baseline,
        "cpu_peak_bytes": cpu_peak,
        "cpu_peak_delta_bytes": cpu_delta,
        "gpu_process_baseline_bytes": process_baseline,
        "gpu_process_peak_bytes": process_peak,
        "gpu_process_peak_delta_bytes": process_delta,
        "gpu_pool_used_baseline_bytes": pool_used_baseline,
        "gpu_pool_used_peak_bytes": pool_used_peak,
        "gpu_pool_used_peak_delta_bytes": pool_used_delta,
        "gpu_pool_total_baseline_bytes": pool_total_baseline,
        "gpu_pool_total_peak_bytes": pool_total_peak,
        "gpu_pool_total_peak_delta_bytes": pool_total_delta,
        "gpu_device_baseline_bytes": device_baseline,
        "gpu_device_peak_bytes": device_peak,
        "gpu_device_peak_delta_bytes": device_delta,
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at": now_utc(),
        **machine_info(),
        "git_commit": git_commit(REPO_ROOT),
        "packages": package_versions(
            ("ibumap", "numpy", "scipy", "scikit-learn", "umap-learn", "pynndescent", "numba", "pyFFTW",
             "cupy-cuda12x")
        ),
    }


def failure_payload(
    *, stage: str, task: BenchmarkTask | None, error: BaseException, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "dataset": None if task is None else task.dataset_id,
        "run_id": None if task is None else task.run_id,
        "created_at": now_utc(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "metadata": dict(metadata or {}),
    }


def validate_static_contract(configs: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    datasets = dataset_entries(configs)
    algorithms = algorithm_entries(configs)
    contract = (configs.get("experiment") or {}).get("dataset_contract") or {}
    required_count = int(contract.get("required_count", 30))
    if len(datasets) != required_count:
        issues.append(f"Expected {required_count} datasets, found {len(datasets)}")
    names = [str(entry.get("name")) for entry in datasets]
    if len(names) != len(set(names)):
        issues.append("Dataset names are not unique")
    maximum_rows = int(contract.get("maximum_rows", 200000))
    over_limit = [str(entry["name"]) for entry in datasets if int(entry.get("expected_rows", 0)) > maximum_rows]
    if over_limit:
        issues.append(f"Datasets exceed {maximum_rows} rows: {', '.join(over_limit)}")
    families = {str(entry.get("family")) for entry in datasets}
    missing_families = set(contract.get("required_families") or []) - families
    if missing_families:
        issues.append(f"Missing dataset families: {', '.join(sorted(missing_families))}")
    bins = {str(entry.get("size_bin")) for entry in datasets}
    missing_bins = set(contract.get("required_size_bins") or []) - bins
    if missing_bins:
        issues.append(f"Missing size bins: {', '.join(sorted(missing_bins))}")

    combinations = {(str(entry.get("device")), str(entry.get("variant"))) for entry in algorithms}
    expected = {(device, variant) for device in DEVICE_ORDER for variant in VARIANT_ORDER}
    if combinations != expected:
        issues.append(f"Algorithm matrix mismatch: expected {sorted(expected)}, got {sorted(combinations)}")
    for entry in algorithms:
        try:
            metadata = fft_metadata(entry)
        except Exception as exc:
            issues.append(f"{entry.get('name')}: invalid FFT config: {type(exc).__name__}: {exc}")
            continue
        variant = str(entry.get("variant"))
        if metadata["combine_stages"]:
            issues.append(f"{entry.get('name')}: deprecated combine_stages must remain false")
        if variant == "schedule":
            expected_schedule = [(0.0, 1), (0.9, 2), (0.95, 3)]
            actual_schedule = [
                (stage["start_fraction"], stage["n_interpolation_points"])
                for stage in metadata["interpolation_schedule"] or []
            ]
            if actual_schedule != expected_schedule:
                issues.append(f"{entry.get('name')}: unexpected schedule {actual_schedule}")
        else:
            expected_p = int(variant[1:])
            if metadata["n_interpolation_points"] != expected_p or metadata["interpolation_schedule"] is not None:
                issues.append(f"{entry.get('name')}: fixed {variant} config is inconsistent")
    if len(repeat_seeds(configs)) < 3:
        issues.append("At least three paired repeat seeds are required for stability")
    return issues


def rows_from_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows
