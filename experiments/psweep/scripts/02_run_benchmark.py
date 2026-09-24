#!/usr/bin/env python3
"""Run every optimizer task in isolated child processes.

For each task the parent first runs a warmup worker (same variant, at most 512
rows), then a fresh measurement worker that times one ``optimize_from_graph``
call on the full fixed inputs while ibUMAP's memory diagnostics sample the
process every 20 ms.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    BenchmarkTask,
    algorithm_entries,
    atomic_save_npy,
    build_tasks,
    configuration_hash,
    dataset_entries,
    embedding_path,
    ensure_directories,
    environment_snapshot,
    experiment_paths,
    failure_payload,
    fft_metadata,
    file_sha256,
    load_configs,
    load_dataset,
    load_fixed_bundle,
    memory_path,
    optimizer_params,
    parse_memory_diagnostics,
    read_json,
    repeat_seeds,
    run_error_path,
    run_record_path,
    task_lookup,
)
from common.algorithm_adapters import run_ibumap_optimize_from_graph
from common.gpu_runtime import force_cleanup
from common.paths import repo_relative
from common.run_metadata import now_utc, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--algorithm", action="append", dest="algorithms")
    parser.add_argument("--device", action="append", choices=("cpu", "cuda"), dest="devices")
    parser.add_argument("--variant", action="append", choices=("p1", "p2", "p3", "schedule"), dest="variants")
    parser.add_argument("--repeat", action="append", type=int, dest="repeats")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execution-index", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def embedding_sanity(embedding: np.ndarray, rows: int) -> dict[str, Any]:
    finite = bool(np.isfinite(embedding).all())
    span = np.ptp(embedding, axis=0) if finite and embedding.size else np.array([np.nan, np.nan])
    return {
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "row_count_matches": int(embedding.shape[0]) == int(rows),
        "finite": finite,
        "bbox_area": float(np.prod(span)) if finite else None,
        "min": float(embedding.min()) if finite else None,
        "max": float(embedding.max()) if finite else None,
    }


def run_once(
    X: np.ndarray,
    graph: Any,
    init: np.ndarray,
    task: BenchmarkTask,
    configs: Mapping[str, Any],
    diagnostics: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return run_ibumap_optimize_from_graph(
        X,
        graph.copy(),
        init.copy(),
        optimizer_params(configs, task, diagnostics_path=diagnostics),
        algorithm="ibumap",
        device=task.device,
    )


def worker(args: argparse.Namespace, configs: Mapping[str, Any]) -> int:
    if not args.datasets or len(args.datasets) != 1 or not args.algorithms or len(args.algorithms) != 1 or not args.repeats or len(args.repeats) != 1:
        raise ValueError("Worker requires exactly one --dataset, --algorithm, and --repeat")
    task = task_lookup(configs, args.datasets[0], args.algorithms[0], args.repeats[0])
    if args.execution_index is not None:
        task = replace(task, execution_index=int(args.execution_index))
    paths = experiment_paths(configs, args.config_dir.resolve())
    ensure_directories(paths)
    config_hash = configuration_hash(configs)
    output_embedding = embedding_path(paths, task)
    output_record = run_record_path(paths, task)
    output_error = run_error_path(paths, task)
    diagnostics = memory_path(paths, task)
    started_at = now_utc()
    process_started = time.perf_counter()
    try:
        dataset_id, X, dataset = load_dataset(task.dataset_entry, paths, mmap=False)
        graph, init, fixed_metadata = load_fixed_bundle(configs, paths, task.dataset_entry)
        if X.shape[0] != graph.shape[0]:
            raise ValueError(f"feature/graph row mismatch: {X.shape[0]} != {graph.shape[0]}")
        warmup = (configs.get("experiment") or {}).get("warmup") or {}
        if args.warmup_only:
            n = min(int(warmup.get("max_rows", 512)), int(X.shape[0]))
            warm_started = time.perf_counter()
            warm_embedding, _ = run_once(
                np.asarray(X[:n], dtype=np.float32, order="C"),
                graph[:n, :n].tocsr(),
                np.asarray(init[:n], dtype=np.float32, order="C"),
                task,
                configs,
                None,
            )
            finite = bool(np.isfinite(warm_embedding).all())
            if not finite:
                raise ValueError("Warmup produced non-finite values")
            print(
                f"WARMUP dataset={dataset_id} algorithm={task.algorithm_id} "
                f"rows={n} wall_s={time.perf_counter() - warm_started:.6f}"
            )
            return 0

        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.unlink(missing_ok=True)
        call_started = time.perf_counter()
        embedding, extras = run_once(X, graph, init, task, configs, diagnostics)
        optimization_wall_s = time.perf_counter() - call_started
        embedding = np.asarray(embedding, dtype=np.float32, order="C")
        sanity = embedding_sanity(embedding, X.shape[0])
        if not sanity["row_count_matches"] or not sanity["finite"]:
            raise ValueError(f"Invalid embedding: {sanity}")
        atomic_save_npy(output_embedding, embedding)
        time_costs = extras.get("time_costs") if isinstance(extras.get("time_costs"), Mapping) else {}
        record = {
            "status": "ok",
            "stage": "optimization_only",
            "dataset": dataset_id,
            "dataset_family": task.dataset_entry["family"],
            "size_bin": task.dataset_entry["size_bin"],
            "expected_rows": int(task.dataset_entry["expected_rows"]),
            "run_id": task.run_id,
            "algorithm_id": task.algorithm_id,
            "device": task.device,
            "variant": task.variant,
            "repeat": task.repeat,
            "random_state": task.seed,
            "execution_index": task.execution_index,
            "configuration_hash": config_hash,
            "started_at": started_at,
            "finished_at": now_utc(),
            "optimization_wall_s": float(optimization_wall_s),
            "worker_wall_s": float(time.perf_counter() - process_started),
            "time_costs": dict(time_costs),
            "fft": fft_metadata(task.algorithm_entry),
            "effective_config": extras.get("effective_config"),
            "memory": parse_memory_diagnostics(diagnostics, task.device),
            "memory_diagnostics_path": repo_relative(diagnostics),
            "embedding_path": repo_relative(output_embedding),
            "embedding_sha256": file_sha256(output_embedding),
            "embedding_sanity": sanity,
            "fixed_input_identity_hash": fixed_metadata.get("identity_hash"),
            "fixed_graph_nnz": int(graph.nnz),
            "warmup": {"status": "separate_worker" if not args.skip_warmup else "skipped"},
            "process": {"pid": os.getpid(), "isolated": True},
            "environment": environment_snapshot(),
            "features_path": repo_relative(dataset.features_path),
        }
        write_json(output_record, record)
        output_error.unlink(missing_ok=True)
        print(
            f"OK dataset={dataset_id} algorithm={task.algorithm_id} repeat={task.repeat} "
            f"wall_s={optimization_wall_s:.6f} memory={record['memory']['primary_peak_bytes']}"
        )
        return 0
    except Exception as exc:
        payload = failure_payload(
            stage="warmup" if args.warmup_only else "optimization_only",
            task=task,
            error=exc,
            metadata={
                "configuration_hash": config_hash,
                "device": task.device,
                "variant": task.variant,
                "repeat": task.repeat,
                "random_state": task.seed,
                "worker_wall_s": time.perf_counter() - process_started,
                "memory": parse_memory_diagnostics(diagnostics, task.device),
                "memory_diagnostics_path": repo_relative(diagnostics),
            },
        )
        write_json(output_error, payload)
        print(f"FAILED dataset={task.dataset_id} algorithm={task.algorithm_id}: {type(exc).__name__}: {exc}")
        return 1
    finally:
        force_cleanup(include_gpu=task.device == "cuda")


def record_matches(record: Mapping[str, Any], task: BenchmarkTask, config_hash: str, output: Path) -> bool:
    return bool(
        record.get("status") == "ok"
        and record.get("configuration_hash") == config_hash
        and int(record.get("random_state", -1)) == task.seed
        and output.exists()
    )


def parent(args: argparse.Namespace, configs: Mapping[str, Any]) -> int:
    paths = experiment_paths(configs, args.config_dir.resolve())
    ensure_directories(paths)
    algorithms = algorithm_entries(
        configs, args.algorithms, devices=args.devices, variants=args.variants
    )
    datasets = dataset_entries(configs, args.datasets)
    repeats = args.repeats or list(range(1, len(repeat_seeds(configs)) + 1))
    tasks = build_tasks(configs, datasets, algorithms, repeats)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]
    print(
        f"tasks={len(tasks)} datasets={len(datasets)} algorithms={len(algorithms)} "
        f"repeats={repeats} mode={'run' if args.run else 'dry-run'}"
    )
    config_hash = configuration_hash(configs)
    if not args.run:
        for task in tasks:
            record = run_record_path(paths, task)
            output = embedding_path(paths, task)
            status = "complete" if record.exists() and output.exists() else "pending"
            print(
                f"- index={task.execution_index} dataset={task.dataset_id} "
                f"algorithm={task.algorithm_id} repeat={task.repeat} seed={task.seed} status={status}"
            )
        return 0

    failures = 0
    skipped = 0
    completed = 0
    for task in tasks:
        record_path = run_record_path(paths, task)
        output_path = embedding_path(paths, task)
        error_path = run_error_path(paths, task)
        diagnostics_path = memory_path(paths, task)
        if record_path.exists() and not args.overwrite:
            if record_matches(read_json(record_path), task, config_hash, output_path):
                skipped += 1
                print(f"SKIP dataset={task.dataset_id} run_id={task.run_id}")
                continue
            raise RuntimeError(f"Stale run output exists; pass --overwrite: {record_path}")
        if args.overwrite:
            for path in (record_path, output_path, error_path, diagnostics_path):
                path.unlink(missing_ok=True)
        base_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run",
            "--config-dir",
            str(args.config_dir.resolve()),
            "--dataset",
            task.dataset_id,
            "--algorithm",
            task.algorithm_id,
            "--repeat",
            str(task.repeat),
            "--execution-index",
            str(task.execution_index),
        ]
        warmup_config = (configs.get("experiment") or {}).get("warmup") or {}
        warmup_payload: dict[str, Any] = {"status": "skipped"}
        if bool(warmup_config.get("enabled", True)) and not args.skip_warmup:
            warmup_command = [*base_command, "--warmup-only"]
            warmup_started = time.perf_counter()
            warmup_result = subprocess.run(
                warmup_command, cwd=EXPERIMENT_ROOT, check=False
            )
            warmup_payload = {
                "status": "ok" if warmup_result.returncode == 0 else "failed",
                "subprocess_wall_s": time.perf_counter() - warmup_started,
                "returncode": warmup_result.returncode,
                "isolated_from_measurement": True,
                "max_rows": int(warmup_config.get("max_rows", 512)),
            }
            if warmup_result.returncode != 0:
                failures += 1
                if not error_path.exists():
                    write_json(
                        error_path,
                        {
                            "status": "failed",
                            "stage": "warmup_subprocess",
                            "dataset": task.dataset_id,
                            "run_id": task.run_id,
                            "error_type": "SubprocessExit",
                            "error_message": f"warmup worker exited with code {warmup_result.returncode}",
                            "metadata": warmup_payload,
                        },
                    )
                if not bool((configs.get("experiment") or {}).get("continue_on_error", True)):
                    return warmup_result.returncode or 1
                continue
        command = [*base_command, "--skip-warmup"]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=EXPERIMENT_ROOT, check=False)
        subprocess_wall = time.perf_counter() - started
        if result.returncode == 0 and record_path.exists():
            payload = read_json(record_path)
            payload["subprocess_wall_s"] = subprocess_wall
            payload["warmup"] = warmup_payload
            write_json(record_path, payload)
            completed += 1
            continue
        failures += 1
        if not error_path.exists():
            write_json(
                error_path,
                {
                    "status": "failed",
                    "stage": "optimization_only_subprocess",
                    "dataset": task.dataset_id,
                    "run_id": task.run_id,
                    "error_type": "SubprocessExit",
                    "error_message": f"worker exited with code {result.returncode}",
                    "metadata": {
                        "returncode": result.returncode,
                        "subprocess_wall_s": subprocess_wall,
                        "device": task.device,
                        "variant": task.variant,
                    },
                },
            )
        if not bool((configs.get("experiment") or {}).get("continue_on_error", True)):
            return result.returncode or 1
    print(f"completed={completed} skipped={skipped} failed={failures}")
    return 1 if failures else 0


def main() -> None:
    args = parse_args()
    args.config_dir = args.config_dir.resolve()
    configs = load_configs(args.config_dir)
    status = worker(args, configs) if args.worker else parent(args, configs)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
