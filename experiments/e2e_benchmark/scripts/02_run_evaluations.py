#!/usr/bin/env python3
"""Score every successful embedding with the paper's five quality metrics.

Trustworthiness, continuity and neighborhood preservation (k = 15) and random
triplet accuracy / distance Spearman correlation (10^6 triplets / pairs, seed 42)
are computed on all rows, or on 10^6 rows sampled with seed 42 for larger
datasets. Source-side states are computed once per dataset and cached; an
embedding that is bitwise identical to an already scored repeat reuses its
scores through a separate, task-bound record.
"""

from __future__ import annotations

from _terminal_log import reexec_with_terminal_log, setup_terminal_logger

if __name__ == "__main__":
    reexec_with_terminal_log(__file__)

import argparse
import copy
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    PROTOCOL_VERSION,
    algorithm_entries,
    atomic_save_npy,
    benchmark_configuration_hash,
    benchmark_protocol_hash,
    build_tasks,
    dataset_entries,
    embedding_path,
    ensure_experiment_dirs,
    environment_snapshot,
    evaluation_error_path,
    evaluation_path,
    evaluation_request_hash,
    evaluation_runtime_signature,
    experiment_paths,
    failure_payload,
    file_sha256,
    hardware_signature,
    load_configs,
    load_dataset,
    n_neighbors_values,
    read_json,
    recorded_dataset_id,
    repeat_values,
    resource_skipped_task_count,
    runtime_signature,
    run_record_path,
    summary_configuration_hash,
    task_signature,
    validate_configs,
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
    parser.add_argument("--dataset", action="append", dest="dataset_filters")
    parser.add_argument("--algorithm", action="append", dest="algorithm_filters")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--repeat", type=int, action="append", dest="repeat_filters")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"))
    parser.add_argument("--gpu-batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    config_issues = validate_configs(configs)
    if config_issues:
        raise ValueError("Invalid experiment configuration: " + "; ".join(config_issues))
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    datasets = dataset_entries(configs, args.dataset_filters)
    algorithms = algorithm_entries(configs, args.algorithm_filters, profiles=args.profiles)
    repeats = repeat_values(configs, args.repeat_filters)
    neighbors = n_neighbors_values(configs)
    if len(neighbors) != 1:
        raise ValueError("Evaluation expects exactly one configured n_neighbors value")
    tasks = build_tasks(configs, datasets, algorithms, repeats, n_neighbors=neighbors[0])
    resource_skipped = resource_skipped_task_count(datasets, algorithms, repeats)
    tasks_by_dataset: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        tasks_by_dataset[task.dataset_id].append(task)

    evaluation_config = dict(configs.get("evaluation") or {})
    if args.device is not None:
        evaluation_config["device"] = args.device
    if args.gpu_batch_size is not None:
        evaluation_config["gpu_batch_size"] = args.gpu_batch_size
    request = request_from_config(evaluation_config)
    sample_seed = int(evaluation_config.get("sample_seed", 42))
    configured_max = evaluation_config.get("max_samples")
    max_samples = args.max_samples if args.max_samples is not None else (
        None if configured_max is None else int(configured_max)
    )
    store = cache_store_from_config(
        evaluation_config,
        base_dir=config_dir,
        default_root=paths.evaluation_root / "metric_cache",
    )
    benchmark_config_hash = benchmark_configuration_hash(configs)
    summary_config_hash = summary_configuration_hash(configs)
    request_hash = evaluation_request_hash(
        evaluation_config,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    device_status = evaluation_device_status(request.device)
    print(
        f"Evaluation tasks: {len(tasks)}; metrics: {list(request.metrics)}; "
        f"device: {device_status.get('resolved')}; max_samples: {max_samples}; "
        f"resource-skipped: {resource_skipped}"
    )
    dry_run = args.dry_run or (
        bool((configs.get("experiment") or {}).get("dry_run_default", False)) and not args.run
    )
    if dry_run:
        for task in tasks:
            run_ok = run_record_path(paths, task).exists() and embedding_path(paths, task).exists()
            eval_ok = evaluation_path(paths, task).exists()
            print(
                f"- dataset={task.dataset_id} run_id={task.run_id} "
                f"embedding={'ok' if run_ok else 'missing'} evaluation={'complete' if eval_ok else 'pending'}"
            )
        return

    logger = setup_terminal_logger("end_to_end_evaluation")
    environment = environment_snapshot()
    current_evaluation_runtime_signature = evaluation_runtime_signature(
        environment,
        device_status,
    )
    dataset_by_id = {str(entry["name"]): entry for entry in datasets}
    failures = 0
    completed = 0
    computed = 0
    reused = 0
    skipped = 0
    missing_runs = 0

    for dataset_id, dataset_tasks in tasks_by_dataset.items():
        evaluations_by_embedding_hash: dict[str, dict[str, Any]] = {}
        candidate_tasks: list[tuple[Any, dict[str, Any]]] = []
        for task in dataset_tasks:
            run_path = run_record_path(paths, task)
            emb_path = embedding_path(paths, task)
            if not run_path.is_file() or not emb_path.is_file():
                missing_runs += 1
                logger.warning(
                    "SKIP missing successful run dataset=%s run_id=%s",
                    dataset_id,
                    task.run_id,
                )
                continue
            run_record = read_json(run_path)
            if run_record.get("status") != "ok":
                missing_runs += 1
                continue
            candidate_tasks.append((task, run_record))
        if not candidate_tasks:
            logger.info("SKIP DATASET dataset=%s no successful runs", dataset_id)
            continue

        entry = dataset_by_id[dataset_id]
        _, features, dataset, _ = load_dataset(entry, paths, mmap=True)
        sample_indices = select_row_indices(
            int(features.shape[0]),
            max_samples,
            seed=sample_seed,
        )
        source_points = np.asarray(
            features[sample_indices],
            dtype=request.dtype,
            order="C",
        )
        if source_points.ndim > 2:
            source_points = source_points.reshape(source_points.shape[0], -1)
        feature_sha256 = file_sha256(dataset.features_path)
        source_key = source_cache_key(
            dataset_id=dataset_id,
            n_rows=source_points.shape[0],
            n_features=source_points.shape[1],
            request=request,
            sample_indices=sample_indices,
            extra={
                "sample_seed": sample_seed,
                "max_samples": max_samples,
                "feature_sha256": feature_sha256,
            },
        )
        sample_path = paths.evaluation_root / "sample_indices" / f"{dataset_id}.npy"
        atomic_save_npy(sample_path, sample_indices)
        logger.info(
            "DATASET dataset=%s sampled_shape=%s source_key=%s",
            dataset_id,
            source_points.shape,
            source_key,
        )

        for task, run_record in candidate_tasks:
            emb_path = embedding_path(paths, task)
            output = evaluation_path(paths, task)
            error_output = evaluation_error_path(paths, task)
            embedding_hash = ""
            started_at = now_utc()
            started = time.perf_counter()
            try:
                protocol_hash = benchmark_protocol_hash(configs, task.algorithm_entry)
                expected_task_signature = task_signature(
                    configs,
                    task,
                    feature_sha256=feature_sha256,
                    protocol_hash=protocol_hash,
                )
                run_environment = (
                    run_record.get("environment")
                    if isinstance(run_record.get("environment"), Mapping)
                    else {}
                )
                expected_runtime_signature = runtime_signature(
                    task.algorithm_entry,
                    run_environment,
                )
                expected_hardware_signature = hardware_signature(run_environment)
                expected_fields = {
                    "protocol_version": PROTOCOL_VERSION,
                    "benchmark_configuration_hash": benchmark_config_hash,
                    "benchmark_protocol_hash": protocol_hash,
                    "task_signature": expected_task_signature,
                    "runtime_signature": expected_runtime_signature,
                    "hardware_signature": expected_hardware_signature,
                    "feature_sha256": feature_sha256,
                    "run_id": task.run_id,
                    "algorithm_id": task.algorithm_id,
                    "repeat": task.repeat,
                    "n_neighbors": task.n_neighbors,
                }
                mismatches = [
                    f"{key}: {run_record.get(key)!r} != {value!r}"
                    for key, value in expected_fields.items()
                    if run_record.get(key) != value
                ]
                if recorded_dataset_id(run_record) != dataset_id:
                    mismatches.append(
                        "dataset_id: "
                        f"{recorded_dataset_id(run_record)!r} != {dataset_id!r}"
                    )
                if mismatches:
                    raise ValueError(
                        "Upstream run record is stale or inconsistent: "
                        + "; ".join(mismatches)
                    )
                recorded_embedding_hash = run_record.get("embedding_sha256")
                if not recorded_embedding_hash:
                    raise ValueError("Upstream run record has no embedding_sha256")
                embedding_hash = file_sha256(emb_path)
                if embedding_hash != recorded_embedding_hash:
                    raise ValueError(
                        "Embedding hash differs from the upstream run record"
                    )
                if output.exists() and not args.overwrite:
                    existing = read_json(output)
                    if (
                        existing.get("summary_configuration_hash")
                        == summary_config_hash
                        and existing.get("evaluation_request_hash") == request_hash
                        and existing.get("evaluation_runtime_signature")
                        == current_evaluation_runtime_signature
                        and existing.get("task_signature")
                        == expected_task_signature
                        and existing.get("embedding_sha256") == embedding_hash
                    ):
                        if (
                            existing.get("status") == "ok"
                            and existing.get("evaluation_computed") is True
                        ):
                            reusable_existing = copy.deepcopy(existing)
                            reusable_existing["evaluation_path"] = repo_relative(output)
                            evaluations_by_embedding_hash.setdefault(
                                embedding_hash,
                                reusable_existing,
                            )
                        skipped += 1
                        error_output.unlink(missing_ok=True)
                        continue
                    raise RuntimeError(
                        f"Stale evaluation exists; pass --overwrite: {output}"
                    )

                reusable = evaluations_by_embedding_hash.get(embedding_hash)
                if reusable is not None:
                    reused_from = {
                        "run_id": reusable["run_id"],
                        "repeat": reusable["repeat"],
                        "evaluation_path": str(
                            Path(dataset_id) / f"{reusable['run_id']}.json"
                        ),
                        "embedding_sha256": embedding_hash,
                    }
                    payload = {
                        "status": "ok",
                        "stage": "evaluation",
                        "protocol_version": PROTOCOL_VERSION,
                        "dataset": dataset_id,
                        "run_id": task.run_id,
                        "algorithm_id": task.algorithm_id,
                        "execution_profile": task.execution_profile,
                        "repeat": task.repeat,
                        "n_neighbors": task.n_neighbors,
                        "started_at": started_at,
                        "finished_at": now_utc(),
                        "benchmark_configuration_hash": benchmark_config_hash,
                        "benchmark_protocol_hash": protocol_hash,
                        "summary_configuration_hash": summary_config_hash,
                        "task_signature": expected_task_signature,
                        "runtime_signature": expected_runtime_signature,
                        "hardware_signature": expected_hardware_signature,
                        "feature_sha256": feature_sha256,
                        "evaluation_request_hash": request_hash,
                        "evaluation_runtime_signature": (
                            current_evaluation_runtime_signature
                        ),
                        "request": request.metadata(),
                        "sample_size": int(source_points.shape[0]),
                        "sample_seed": sample_seed,
                        "sample_indices_path": repo_relative(sample_path),
                        "source_key": source_key,
                        "embedding_key": reusable.get("embedding_key"),
                        "embedding_path": repo_relative(emb_path),
                        "embedding_sha256": embedding_hash,
                        "scores": copy.deepcopy(reusable["scores"]),
                        "score_scalars": copy.deepcopy(reusable["score_scalars"]),
                        "cache": {
                            "reused": True,
                            "source_run_id": reusable["run_id"],
                        },
                        "evaluation_computed": False,
                        "evaluation_reused_from": reused_from,
                        "evaluation_compute_wall_s": 0.0,
                        "evaluation_wall_s": float(time.perf_counter() - started),
                        "device": device_status,
                        "environment": environment,
                        "evaluation_path": repo_relative(output),
                    }
                    write_json(output, payload)
                    error_output.unlink(missing_ok=True)
                    completed += 1
                    reused += 1
                    logger.info(
                        "REUSE dataset=%s run_id=%s source_run_id=%s",
                        dataset_id,
                        task.run_id,
                        reusable["run_id"],
                    )
                    continue

                raw_embedding = np.load(emb_path, mmap_mode="r")
                if raw_embedding.shape[0] != features.shape[0]:
                    raise ValueError(
                        f"Embedding rows {raw_embedding.shape[0]} "
                        f"!= dataset rows {features.shape[0]}"
                    )
                embedding = np.asarray(
                    raw_embedding[sample_indices],
                    dtype=request.dtype,
                    order="C",
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
                        "feature_sha256": feature_sha256,
                    },
                    embedding_metadata={
                        "run_id": task.run_id,
                        "embedding_path": repo_relative(emb_path),
                        "embedding_sha256": embedding_hash,
                        "task_signature": expected_task_signature,
                    },
                )
                scores = summarize_evaluation_scores(result.scores)
                evaluation_wall_s = float(time.perf_counter() - started)
                payload = {
                    "status": "ok",
                    "stage": "evaluation",
                    "protocol_version": PROTOCOL_VERSION,
                    "dataset": dataset_id,
                    "run_id": task.run_id,
                    "algorithm_id": task.algorithm_id,
                    "execution_profile": task.execution_profile,
                    "repeat": task.repeat,
                    "n_neighbors": task.n_neighbors,
                    "started_at": started_at,
                    "finished_at": now_utc(),
                    "benchmark_configuration_hash": benchmark_config_hash,
                    "benchmark_protocol_hash": protocol_hash,
                    "summary_configuration_hash": summary_config_hash,
                    "task_signature": expected_task_signature,
                    "runtime_signature": expected_runtime_signature,
                    "hardware_signature": expected_hardware_signature,
                    "feature_sha256": feature_sha256,
                    "evaluation_request_hash": request_hash,
                    "evaluation_runtime_signature": (
                        current_evaluation_runtime_signature
                    ),
                    "request": request.metadata(),
                    "sample_size": int(source_points.shape[0]),
                    "sample_seed": sample_seed,
                    "sample_indices_path": repo_relative(sample_path),
                    "source_key": source_key,
                    "embedding_key": embed_key,
                    "embedding_path": repo_relative(emb_path),
                    "embedding_sha256": embedding_hash,
                    "scores": scores,
                    "score_scalars": scalar_scores(scores),
                    "cache": {
                        key: repo_relative(value) if key in {"root", "source_dir", "embedding_dir"} else value
                        for key, value in result.cache.items()
                    },
                    "evaluation_computed": True,
                    "evaluation_reused_from": None,
                    "evaluation_compute_wall_s": evaluation_wall_s,
                    "evaluation_wall_s": evaluation_wall_s,
                    "device": device_status,
                    "environment": environment,
                    "evaluation_path": repo_relative(output),
                }
                write_json(output, payload)
                evaluations_by_embedding_hash[embedding_hash] = payload
                error_output.unlink(missing_ok=True)
                completed += 1
                computed += 1
                logger.info("END dataset=%s run_id=%s", dataset_id, task.run_id)
            except Exception as exc:
                if output.exists() and not args.overwrite:
                    raise
                failures += 1
                failed = failure_payload(
                    stage="evaluation",
                    dataset_id=dataset_id,
                    run_id=task.run_id,
                    error=exc,
                    metadata={
                        "protocol_version": PROTOCOL_VERSION,
                        "benchmark_configuration_hash": benchmark_config_hash,
                        "summary_configuration_hash": summary_config_hash,
                        "evaluation_request_hash": request_hash,
                        "evaluation_runtime_signature": (
                            current_evaluation_runtime_signature
                        ),
                        "embedding_path": repo_relative(emb_path),
                        "embedding_sha256": embedding_hash or None,
                        "evaluation_wall_s": float(time.perf_counter() - started),
                    },
                )
                write_json(error_output, failed)
                logger.exception(
                    "FAILED dataset=%s run_id=%s error=%s",
                    dataset_id,
                    task.run_id,
                    exc,
                )
            finally:
                force_cleanup(include_gpu=device_status.get("resolved") == "gpu")

        source_deleted = store.cleanup_after_dataset(source_key)
        logger.info("CACHE dataset=%s source_deleted=%s", dataset_id, source_deleted)
        del source_points, features

    print(
        f"Completed: {completed} (computed: {computed}; reused: {reused}); "
        f"skipped: {skipped}; missing benchmark runs: {missing_runs}; "
        f"failed: {failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
