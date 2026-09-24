#!/usr/bin/env python3
"""Score successful embeddings with trustworthiness, continuity and neighborhood preservation (k = 15).

All rows are scored (``max_samples: null``); source states are cached per dataset.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    algorithm_entries,
    atomic_save_npy,
    build_tasks,
    configuration_hash,
    dataset_entries,
    embedding_path,
    ensure_directories,
    evaluation_error_path,
    evaluation_path,
    experiment_paths,
    failure_payload,
    file_sha256,
    fixed_bundle_paths,
    load_configs,
    load_dataset,
    read_json,
    repeat_seeds,
    run_record_path,
)
from common.dataset_io import select_row_indices
from common.evaluation_cache import (
    cache_store_from_config,
    embedding_cache_key,
    evaluation_device_status,
    request_from_config,
    scalar_scores,
    source_cache_key,
    summarize_evaluation_scores,
)
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
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(configs, config_dir)
    ensure_directories(paths)
    datasets = dataset_entries(configs, args.datasets)
    algorithms = algorithm_entries(
        configs, args.algorithms, devices=args.devices, variants=args.variants
    )
    repeats = args.repeats or list(range(1, len(repeat_seeds(configs)) + 1))
    tasks = build_tasks(configs, datasets, algorithms, repeats)
    tasks_by_dataset: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        tasks_by_dataset[task.dataset_id].append(task)

    evaluation_config = dict(configs.get("evaluation") or {})
    if args.max_samples is not None:
        evaluation_config["max_samples"] = args.max_samples
    configured_max_samples = evaluation_config.get("max_samples")
    max_samples = (
        int(args.max_samples)
        if args.max_samples is not None
        else None
        if configured_max_samples is None
        else int(configured_max_samples)
    )
    sample_seed = int(evaluation_config.get("sample_seed", 42))
    request = request_from_config(evaluation_config)
    store = cache_store_from_config(
        evaluation_config,
        base_dir=config_dir,
        default_root=paths.evaluations_root / "metric_cache",
    )
    device_status = evaluation_device_status(request.device)
    config_hash = configuration_hash(configs)
    request_hash = configuration_hash({"evaluation": evaluation_config})
    print(
        f"tasks={len(tasks)} metrics={list(request.metrics)} max_samples={max_samples} "
        f"evaluation_device={device_status.get('resolved')} mode={'run' if args.run else 'dry-run'}"
    )
    if not args.run:
        for task in tasks:
            has_embedding = embedding_path(paths, task).exists() and run_record_path(paths, task).exists()
            has_evaluation = evaluation_path(paths, task).exists()
            print(
                f"- dataset={task.dataset_id} run_id={task.run_id} "
                f"embedding={'ok' if has_embedding else 'missing'} "
                f"evaluation={'complete' if has_evaluation else 'pending'}"
            )
        return

    entry_by_id = {str(entry["name"]): entry for entry in datasets}
    completed = skipped = missing = failures = 0
    for dataset_id, dataset_tasks in tasks_by_dataset.items():
        entry = entry_by_id[dataset_id]
        _, features, dataset = load_dataset(entry, paths, mmap=True)
        sample_indices = select_row_indices(
            int(features.shape[0]), max_samples, seed=sample_seed
        )
        source_points = np.asarray(
            features[sample_indices], dtype=request.dtype, order="C"
        )
        if source_points.ndim > 2:
            source_points = source_points.reshape(source_points.shape[0], -1)
        sample_path = paths.evaluations_root / "sample_indices" / f"{dataset_id}.npy"
        atomic_save_npy(sample_path, sample_indices)
        fixed_files = fixed_bundle_paths(configs, paths, dataset_id)
        fixed_metadata = read_json(fixed_files["metadata"]) if fixed_files["metadata"].exists() else {}
        source_key = source_cache_key(
            dataset_id=dataset_id,
            n_rows=source_points.shape[0],
            n_features=source_points.shape[1],
            request=request,
            sample_indices=sample_indices,
            extra={
                "sample_seed": sample_seed,
                "max_samples": max_samples,
                "feature_sha256": fixed_metadata.get("feature_sha256"),
            },
        )
        print(f"DATASET dataset={dataset_id} sampled_shape={source_points.shape}")
        for task in dataset_tasks:
            record_file = run_record_path(paths, task)
            embedding_file = embedding_path(paths, task)
            if not record_file.exists() or not embedding_file.exists():
                missing += 1
                continue
            run_record = read_json(record_file)
            if run_record.get("status") != "ok":
                missing += 1
                continue
            output = evaluation_path(paths, task)
            error_output = evaluation_error_path(paths, task)
            embedding_hash = str(
                run_record.get("embedding_sha256") or file_sha256(embedding_file)
            )
            if output.exists() and not args.overwrite:
                existing = read_json(output)
                if (
                    existing.get("configuration_hash") == config_hash
                    and existing.get("evaluation_request_hash") == request_hash
                    and existing.get("embedding_sha256") == embedding_hash
                ):
                    skipped += 1
                    continue
                raise RuntimeError(f"Stale evaluation exists; pass --overwrite: {output}")
            started = time.perf_counter()
            started_at = now_utc()
            try:
                raw_embedding = np.load(embedding_file, mmap_mode="r")
                if raw_embedding.shape[0] != features.shape[0]:
                    raise ValueError(
                        f"embedding rows {raw_embedding.shape[0]} != source rows {features.shape[0]}"
                    )
                embedding = np.asarray(
                    raw_embedding[sample_indices], dtype=request.dtype, order="C"
                )
                embed_key = embedding_cache_key(
                    embedding_id=task.run_id,
                    embedding=embedding,
                    request=request,
                    extra={"embedding_sha256": embedding_hash},
                )
                result = store.evaluate_embedding(
                    source_points,
                    embedding,
                    source_key=source_key,
                    embedding_key=embed_key,
                    request=request,
                    source_metadata={
                        "dataset_id": dataset_id,
                        "features_path": repo_relative(dataset.features_path),
                        "sample_indices_path": repo_relative(sample_path),
                        "sample_seed": sample_seed,
                    },
                    embedding_metadata={
                        "run_id": task.run_id,
                        "embedding_path": repo_relative(embedding_file),
                        "embedding_sha256": embedding_hash,
                    },
                )
                scores = summarize_evaluation_scores(result.scores)
                payload = {
                    "status": "ok",
                    "stage": "evaluation",
                    "dataset": dataset_id,
                    "dataset_family": task.dataset_entry["family"],
                    "size_bin": task.dataset_entry["size_bin"],
                    "run_id": task.run_id,
                    "algorithm_id": task.algorithm_id,
                    "device": task.device,
                    "variant": task.variant,
                    "repeat": task.repeat,
                    "random_state": task.seed,
                    "started_at": started_at,
                    "finished_at": now_utc(),
                    "configuration_hash": config_hash,
                    "evaluation_request_hash": request_hash,
                    "request": request.metadata(),
                    "sample_size": int(source_points.shape[0]),
                    "sample_seed": sample_seed,
                    "sample_indices_path": repo_relative(sample_path),
                    "source_key": source_key,
                    "embedding_key": embed_key,
                    "embedding_path": repo_relative(embedding_file),
                    "embedding_sha256": embedding_hash,
                    "scores": scores,
                    "score_scalars": scalar_scores(scores),
                    "cache": {
                        key: repo_relative(value) if key in {"root", "source_dir", "embedding_dir"} else value
                        for key, value in result.cache.items()
                    },
                    "evaluation_wall_s": time.perf_counter() - started,
                    "evaluation_device": device_status,
                }
                write_json(output, payload)
                error_output.unlink(missing_ok=True)
                completed += 1
                print(f"OK dataset={dataset_id} run_id={task.run_id}")
            except Exception as exc:
                failures += 1
                write_json(
                    error_output,
                    failure_payload(
                        stage="evaluation",
                        task=task,
                        error=exc,
                        metadata={
                            "configuration_hash": config_hash,
                            "evaluation_request_hash": request_hash,
                            "embedding_path": repo_relative(embedding_file),
                            "embedding_sha256": embedding_hash,
                            "evaluation_wall_s": time.perf_counter() - started,
                        },
                    ),
                )
                print(f"FAILED dataset={dataset_id} run_id={task.run_id}: {type(exc).__name__}: {exc}")
            finally:
                force_cleanup(include_gpu=device_status.get("resolved") == "gpu")
        store.cleanup_after_dataset(source_key)
        del source_points, features
    print(
        f"completed={completed} skipped={skipped} missing_runs={missing} failed={failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
