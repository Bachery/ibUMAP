#!/usr/bin/env python3
"""Reference: bounding-box statistics of the mechanism experiment's embeddings.

Reads the production ibUMAP and umap-learn embeddings that the mechanism
experiment wrote for its three size suites,
  <mechanism>/results/<suite>/02_runs/<dataset>/<algorithm>__seed_<seed>/embedding.npy
skips the datasets marked ``excluded`` in <mechanism>/configs/datasets.yaml, and
computes the same statistics as for the failure cases. This gives the range of a
normal layout (the paper quotes the median and maximum over its 59 datasets).
Nothing is optimized.

Writes results/reference_bbox.csv and results/reference_bbox_summary.json.

    python scripts/03_reference_bbox.py [--runs-root DIR]    # DIR/<suite>/02_runs (default: mechanism results)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from _common import Layout, bbox_metrics, load_config, repo_relative, resolve, sha256_array, write_csv, write_json

CFG = load_config()


def main() -> None:
    ref = CFG["reference"]
    mechanism = resolve(ref["mechanism_root"])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=mechanism / "results",
                        help="directory with <suite>/02_runs (default: the mechanism experiment's results)")
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    entries = yaml.safe_load((mechanism / "configs" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    exclude = {e["name"] for e in entries if e.get("excluded")}
    core_fraction = float(CFG["metrics"]["core_fraction"])
    far_multiple = float(CFG["metrics"]["far_radius_multiple"])

    rows = []
    for suite in ref["suites"]:
        runs = runs_root / suite / "02_runs"
        if not runs.is_dir():
            print(f"missing {repo_relative(runs)}; skipped")
            continue
        for dataset_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            if dataset_dir.name in exclude:
                continue
            for algorithm in ref["algorithms"]:
                for run_dir in sorted(dataset_dir.glob(f"{algorithm}__seed_*")):
                    path = run_dir / "embedding.npy"
                    if not path.exists():
                        continue
                    y = np.load(path)
                    rows.append({"suite": suite, "dataset": dataset_dir.name, "algorithm": algorithm,
                                 "seed": int(run_dir.name.rsplit("_", 1)[-1]), "embedding_sha256": sha256_array(y),
                                 **bbox_metrics(y, core_fraction, far_multiple)})
    results = Layout(CFG).results
    write_csv(results / "reference_bbox.csv", rows)

    summary = {"core_fraction": core_fraction, "far_radius_multiple": far_multiple,
               "excluded": sorted(exclude), "algorithms": {}}
    for algorithm in ref["algorithms"]:
        part = [r for r in rows if r["algorithm"] == algorithm]
        if not part:
            continue
        # One value per dataset: the maximum over seeds (the least favourable run).
        per_dataset: dict[str, float] = {}
        far: dict[str, int] = {}
        for r in part:
            per_dataset[r["dataset"]] = max(per_dataset.get(r["dataset"], 0.0), r["bbox_area_ratio"])
            far[r["dataset"]] = max(far.get(r["dataset"], 0), r["far_count"])
        values = np.asarray(list(per_dataset.values()))
        worst = max(per_dataset, key=per_dataset.get)
        summary["algorithms"][algorithm] = {
            "n_datasets": len(per_dataset), "n_runs": len(part),
            "bbox_area_ratio_median": float(np.median(values)),
            "bbox_area_ratio_p90": float(np.percentile(values, 90)),
            "bbox_area_ratio_max": float(values.max()), "bbox_area_ratio_max_dataset": worst,
            "datasets_with_far_points": int(sum(v > 0 for v in far.values())),
            "far_count_max": int(max(far.values())), "per_dataset_max_ratio": per_dataset,
        }
        print(f"{algorithm:18s} n={len(per_dataset)} median={np.median(values):.3f} "
              f"max={values.max():.3f} ({worst}) datasets_with_far={summary['algorithms'][algorithm]['datasets_with_far_points']}")
    write_json(results / "reference_bbox_summary.json", summary)


if __name__ == "__main__":
    main()
