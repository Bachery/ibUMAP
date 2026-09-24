#!/usr/bin/env python3
"""Convert the published summary bundle to the paper-data layout (``<output>/e2e_benchmark/``).

    python scripts/07_export_paper_data.py                 # -> paper/rerun/e2e_benchmark/
    python scripts/07_export_paper_data.py --output DIR    # -> DIR/e2e_benchmark/

Writes the files that ``scripts/paper`` reads from ``paper/data/e2e_benchmark/``:

  runs.csv.gz        run_summary.csv without the local path columns
  quality.csv.gz     one row per run and metric (source ``local``: TW, C, NP; ``global``: RTA, distance Spearman)
  stability.csv.gz   stability_summary.csv
  datasets.csv.gz    dataset_summary.csv
  bundle.json        scope, validation flags and evaluation protocol

Values are copied from the bundle CSVs as strings. Only a strict, full-scope,
published bundle is exported, because the paper builders require one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import EXPERIMENT_ROOT, experiment_paths, load_configs, published_summary_root, read_json
from common.paper_data import DEFAULT_RERUN_ROOT, read_csv, write_csv_gz, write_json

GROUP = "e2e_benchmark"
LOCAL_METRICS = ("trustworthiness", "continuity", "neighborhood_preservation")
GLOBAL_METRICS = ("rta", "distance_spearman")
RUN_DROP = ("embedding_path", "record_path")
QUALITY_FIELDS = ("dataset", "run_id", "algorithm_id", "family", "device", "execution_profile", "repeat",
                  "metric", "source", "higher_is_better", "value", "status", "sample_size", "sample_seed",
                  "embedding_sha256")
GLOBAL_CONFIG_KEYS = ("max_samples", "sample_seed", "rta_n_triplets", "rta_random_state",
                      "distance_spearman_n_pairs", "distance_spearman_random_state", "source_metric", "dtype")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--output", type=Path, default=DEFAULT_RERUN_ROOT,
                        help="paper-data root; the group is written to OUTPUT/e2e_benchmark (default: paper/rerun)")
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(config_dir, configs)
    bundle = published_summary_root(paths, configs)
    completeness = read_json(bundle / "completeness.json")
    out = args.output.resolve() / GROUP

    run_fields, runs = read_csv(bundle / "run_summary.csv")
    run_fields = [field for field in run_fields if field not in RUN_DROP]
    runs = [{field: row[field] for field in run_fields} for row in runs]

    _, quality_long = read_csv(bundle / "quality_long.csv")
    metrics = set(LOCAL_METRICS + GLOBAL_METRICS)
    unexpected = {row["metric"] for row in quality_long} - metrics
    if unexpected:
        raise ValueError(f"Unexpected metrics in quality_long.csv: {sorted(unexpected)}")
    quality = [{field: row.get(field, "") for field in QUALITY_FIELDS} for row in quality_long]

    stability_fields, stability = read_csv(bundle / "stability_summary.csv")
    dataset_fields, datasets = read_csv(bundle / "dataset_summary.csv")

    evaluation = dict(configs.get("evaluation") or {})
    record = {
        "description": "End-to-end benchmark (paper Section 5 and Appendix B), exported from a published bundle.",
        "protocol_version": completeness["protocol_version"],
        "summary_configuration_hash": completeness["summary_configuration_hash"],
        "benchmark_configuration_hash": completeness["benchmark_configuration_hash"],
        "strict_passed": completeness["strict_passed"],
        "full_scope": completeness["full_scope"],
        "published": completeness["published"],
        "successful_runs": completeness["successful_runs"],
        "resource_skipped_runs": completeness["resource_skipped_runs"],
        "scope": completeness["scope"],
        "quality_metrics": {
            "local": list(LOCAL_METRICS),
            "global": list(GLOBAL_METRICS),
            "global_coverage": "all runs; bitwise identical repeats reuse the scores of the first repeat",
            "global_configuration": {key: evaluation.get(key) for key in GLOBAL_CONFIG_KEYS},
        },
    }

    files = {
        "runs.csv.gz": write_csv_gz(out / "runs.csv.gz", run_fields, runs),
        "quality.csv.gz": write_csv_gz(out / "quality.csv.gz", QUALITY_FIELDS, quality),
        "stability.csv.gz": write_csv_gz(out / "stability.csv.gz", stability_fields, stability),
        "datasets.csv.gz": write_csv_gz(out / "datasets.csv.gz", dataset_fields, datasets),
        "bundle.json": write_json(out / "bundle.json", record),
    }
    print(f"Exported {bundle.name}: {len(runs)} runs, {len(quality)} quality rows, "
          f"{len(stability)} stability rows, {len(datasets)} datasets -> {out}")
    for name, digest in files.items():
        print(f"  {name}  {digest}")


if __name__ == "__main__":
    main()
