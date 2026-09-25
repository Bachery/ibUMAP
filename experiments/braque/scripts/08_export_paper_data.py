#!/usr/bin/env python3
"""Export an evaluated batch to the paper-data layout (``<output>/braque/``).

    python scripts/08_export_paper_data.py --run-id l2_v1                # -> paper/rerun/braque/
    python scripts/08_export_paper_data.py --run-id l2_v1 --output DIR   # -> DIR/braque/

Writes the files read by ``scripts/paper/ch6_panels.py`` and ``ch6_reuse_table.py``:

  summary.csv.gz, pairwise_metrics.csv.gz, run_metrics.csv.gz, cost_ratios.csv.gz
      the tables of analysis/<analysis-id>/, unchanged
  plot_data.npz
      coordinates and HDBSCAN labels of the plotted pair (the first two scheduled
      unseeded UMAP repeats), run 2 rotated/reflected and translated onto run 1 for
      display (no scaling), run-2 labels matched to run 1 over all cells, change
      category of every cell and the paint order
  panels.json
      rendering parameters, alignment, plot limits, change counts and the pair's
      full-data label metrics

The pair metrics are recomputed here and must equal the evaluated ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _common import (CASE, label_metrics, load_experiment, match_labels, read_csv, read_json, rigid_align,
                     run_root, safe_name, sha, verify)

sys.path.insert(0, str(CASE.parents[1] / "scripts"))
from common.paper_data import DEFAULT_RERUN_ROOT, read_csv as read_csv_strings, write_csv_gz, write_json  # noqa: E402

GROUP = "braque"
TABLES = ("summary.csv", "pairwise_metrics.csv", "run_metrics.csv", "cost_ratios.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--analysis-id", default="main")
    parser.add_argument("--output", type=Path, default=DEFAULT_RERUN_ROOT,
                        help="paper-data root; the group is written to OUTPUT/braque (default: paper/rerun)")
    parser.add_argument("--left-repeat", type=int, default=0)
    parser.add_argument("--right-repeat", type=int, default=1)
    parser.add_argument("--panel-size", type=float, default=3.0, help="square plotting area in inches, all panels")
    parser.add_argument("--legend-width", type=float, default=1.65, help="extra width of panel 3, in inches")
    parser.add_argument("--point-size", type=float, default=.30, help="cluster-panel marker area (pt^2)")
    parser.add_argument("--change-size", type=float, default=.35, help="changed-cluster marker area (pt^2)")
    parser.add_argument("--noise-size", type=float, default=.65, help="noise-transition marker area (pt^2)")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--plot-seed", type=int, default=20260921)
    args = parser.parse_args()
    if not 0 <= args.left_repeat < args.right_repeat:
        raise ValueError("Require 0 <= left-repeat < right-repeat")
    root = run_root(args.run_id)
    exp = load_experiment(root)
    analysis = root / "analysis" / safe_name(args.analysis_id)
    verify(analysis)
    metadata = read_json(analysis / "metadata.json")
    if (metadata["contract_sha256"] != exp["contract_sha256"] or
            metadata["experiment_json_sha256"] != sha(root / "experiment.json")):
        raise ValueError("Analysis does not match the frozen experiment")
    sources = []
    for repeat in (args.left_repeat, args.right_repeat):
        record = next((r for r in metadata["sources"] if r["profile"] == "umap_unseeded" and r["repeat"] == repeat), None)
        if record is None:
            raise ValueError(f"Missing evaluated unseeded UMAP repeat {repeat}")
        directory = root / record["directory"]
        verify(directory)
        if (sha(directory / "run.json") != record["run_json_sha256"] or
                sha(directory / "SHA256SUMS") != record["ledger_sha256"]):
            raise ValueError("Source run changed after evaluation")
        sources.append(directory)

    first, second = [np.load(p / "embedding.npy", allow_pickle=False) for p in sources]
    labels_first, labels_second = [np.load(p / "labels.npy", allow_pickle=False) for p in sources]
    aligned, rotation, translation = rigid_align(first, second)
    matched, _ = match_labels(labels_first, labels_second)
    metrics = label_metrics(labels_first, labels_second)
    evaluated_pair = next(r for r in read_csv(analysis / "pairwise_metrics.csv")
                          if r["profile"] == "umap_unseeded" and int(r["repeat_left"]) == args.left_repeat
                          and int(r["repeat_right"]) == args.right_repeat)
    for key, value in metrics.items():
        recorded = evaluated_pair[key]
        if (value is None and recorded) or (value is not None and not np.isclose(value, float(recorded), rtol=0, atol=1e-12)):
            raise ValueError(f"Metric disagreement with the evaluation: {key}")
    count = len(first)
    order = np.random.default_rng(args.plot_seed).permutation(count)
    category = np.zeros(count, dtype=np.int8)
    category[(labels_first >= 0) & (labels_second >= 0) & (labels_first != matched)] = 1
    category[(labels_first >= 0) & (labels_second == -1)] = 2
    category[(labels_first == -1) & (labels_second >= 0)] = 3
    points = np.concatenate([first, aligned])
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    halfspan = max(float(np.max(upper - lower)), 1e-8) * .525
    limits = {"x": [center[0] - halfspan, center[0] + halfspan], "y": [center[1] - halfspan, center[1] + halfspan]}

    out = args.output.resolve() / GROUP
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in TABLES:
        fields, rows = read_csv_strings(analysis / name)
        files[name + ".gz"] = write_csv_gz(out / f"{name}.gz", fields, rows)
    np.savez_compressed(out / "plot_data.npz", reference_coordinates=first, candidate_coordinates_raw=second,
                        candidate_coordinates_aligned=aligned, reference_labels=labels_first,
                        candidate_labels=labels_second, candidate_labels_matched=matched,
                        change_category=category, paint_order=order)
    files["plot_data.npz"] = sha(out / "plot_data.npz")
    render = {k: getattr(args, k) for k in ("panel_size", "legend_width", "point_size", "change_size", "noise_size",
                                            "dpi", "plot_seed", "left_repeat", "right_repeat")}
    files["panels.json"] = write_json(out / "panels.json", {
        "description": "First two scheduled unseeded UMAP runs on the frozen BRAQUE input; run 2 rotated/reflected "
                       "and translated for display only (no scaling); colors matched using all cells.",
        "selection_rule": "First two scheduled unseeded UMAP repeats by default; no effect-size selection",
        "full_cell_count": count, "render_parameters": render,
        "rotation": rotation.tolist(), "translation": translation.tolist(), "limits": limits,
        "category_counts": {str(k): int(np.sum(category == k)) for k in range(4)},
        "metrics_full_data": metrics, "distance_spearman": float(evaluated_pair["distance_spearman"]),
        "smoke_subset": exp["contract"]["config"]["smoke_subset"]})
    print(f"Exported run pair {args.left_repeat + 1}/{args.right_repeat + 1} of {count:,} cells "
          f"(assignment disagreement {metrics['assignment_disagreement']:.2%}) -> {out}")
    for name, digest in files.items():
        print(f"  {name}  {digest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
