#!/usr/bin/env python3
"""Validate the dataset matrix (catalog and processed arrays), the FFT variants, paths and runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _common import (
    EXPERIMENT_ROOT,
    algorithm_entries,
    dataset_entries,
    ensure_directories,
    environment_snapshot,
    experiment_paths,
    fft_config_for,
    load_configs,
    load_dataset,
    validate_static_contract,
)
from common.paths import repo_relative
from common.run_metadata import now_utc, write_json
from ibumap.fft_schedule import resolve_fft_schedule, validate_fft_schedule_execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--write-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(configs, config_dir)
    ensure_directories(paths)
    issues = validate_static_contract(configs)

    catalog = json.loads(paths.catalog_path.read_text(encoding="utf-8"))
    catalog_by_id = {
        str(entry["dataset_id"]): entry
        for entry in catalog.get("datasets", [])
        if isinstance(entry, dict) and entry.get("dataset_id")
    }
    dataset_rows: list[dict[str, object]] = []
    maximum_rows = int(
        ((configs.get("experiment") or {}).get("dataset_contract") or {}).get(
            "maximum_rows", 200000
        )
    )
    for entry in dataset_entries(configs):
        name = str(entry["name"])
        try:
            catalog_entry = catalog_by_id[name]
            shape = catalog_entry.get("feature_shape")
            if not isinstance(shape, list) or not shape:
                raise ValueError("catalog feature_shape is missing")
            expected_rows = int(entry["expected_rows"])
            if int(shape[0]) != expected_rows:
                raise ValueError(f"catalog rows {shape[0]} != configured {expected_rows}")
            catalog_family = catalog_entry.get("family") or catalog_entry.get("kind")
            if str(catalog_family) != str(entry["family"]):
                raise ValueError(
                    f"catalog family/kind {catalog_family!r} != configured {entry['family']!r}"
                )
            dataset_id, features, dataset = load_dataset(entry, paths, mmap=True)
            if int(features.shape[0]) != expected_rows:
                raise ValueError(f"feature rows {features.shape[0]} != configured {expected_rows}")
            if expected_rows > maximum_rows:
                raise ValueError(f"row count exceeds {maximum_rows}")
            dataset_rows.append(
                {
                    "dataset": dataset_id,
                    "status": "ok",
                    "shape": list(features.shape),
                    "family": entry["family"],
                    "size_bin": entry["size_bin"],
                    "features_path": repo_relative(dataset.features_path),
                }
            )
        except Exception as exc:
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
            dataset_rows.append(
                {
                    "dataset": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    n_epochs = int(
        ((configs.get("algorithms") or {}).get("common_umap_params") or {}).get(
            "n_epochs", 200
        )
    )
    algorithm_rows: list[dict[str, object]] = []
    for entry in algorithm_entries(configs):
        try:
            fft = fft_config_for(entry)
            stages = resolve_fft_schedule(
                n_epochs=n_epochs,
                n_interpolation_points=fft.n_interpolation_points,
                combine_stages=fft.combine_stages,
                interpolation_schedule=fft.interpolation_schedule,
            )
            validate_fft_schedule_execution(
                stages, device=str(entry["device"]), p2m_mode=fft.p2m_mode
            )
            algorithm_rows.append(
                {
                    "algorithm": entry["name"],
                    "status": "ok",
                    "device": entry["device"],
                    "variant": entry["variant"],
                    "resolved_stages": [
                        {
                            "start_epoch": stage.start_epoch,
                            "end_epoch": stage.end_epoch,
                            "n_interpolation_points": stage.n_interpolation_points,
                        }
                        for stage in stages
                    ],
                }
            )
        except Exception as exc:
            issues.append(f"{entry.get('name')}: {type(exc).__name__}: {exc}")
            algorithm_rows.append(
                {
                    "algorithm": entry.get("name"),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "created_at": now_utc(),
        "status": "ok" if not issues else "failed",
        "issues": issues,
        "dataset_count": len(dataset_rows),
        "family_counts": dict(Counter(str(row.get("family")) for row in dataset_rows if row.get("status") == "ok")),
        "size_bin_counts": dict(Counter(str(row.get("size_bin")) for row in dataset_rows if row.get("status") == "ok")),
        "datasets": dataset_rows,
        "algorithms": algorithm_rows,
        "environment": environment_snapshot(),
    }
    print(
        f"datasets={len(dataset_rows)} algorithms={len(algorithm_rows)} "
        f"issues={len(issues)} cuda_available={payload['environment']['gpu'].get('available')}"
    )
    for issue in issues:
        print(f"- ERROR: {issue}")
    if args.write_json:
        write_json(paths.results_root / "00_validation.json", payload)
        print(paths.results_root / "00_validation.json")
    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
