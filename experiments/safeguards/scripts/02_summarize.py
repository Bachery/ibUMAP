#!/usr/bin/env python3
"""Summarize bounding-box statistics, seed identity, traces and escaped points.

Writes, under results/,
  bbox_summary.csv     one row per (dataset, variant, seed)
  case_summary.csv     one row per (dataset, variant): statistics of the plotted seed, their range over
                       seeds, landmarks of the per-update trace, weighted degree of the far points
  escaped_points.csv   far points of each plotted run with their graph degree
  summary.json         the case rows and all trace landmarks
"""
from __future__ import annotations

import numpy as np

from _common import (Layout, bbox_metrics, core_mask, load_config, load_fixed_inputs, read_json, read_rows,
                     sha256_array, write_csv, write_json)

CFG = load_config()


def trace_landmarks(path) -> dict:
    if not path.exists():
        return {}
    rows = read_rows(path)

    def col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) if row.get(name) not in (None, "") else np.nan for row in rows])

    epoch = col("epoch").astype(int)
    ratio, far = col("after_bbox_area_ratio"), col("after_far_count")
    step, attr, repl = col("step_norm_max"), col("attr_norm_max"), col("repl_norm_max")

    def first(mask: np.ndarray):
        idx = np.flatnonzero(mask)
        return int(epoch[idx[0]]) if len(idx) else None

    return {
        "max_step_norm": float(np.nanmax(step)),
        "epoch_of_max_step_norm": int(epoch[np.nanargmax(step)]),
        "max_attr_norm": float(np.nanmax(attr)),
        "epoch_of_max_attr_norm": int(epoch[np.nanargmax(attr)]),
        "max_repl_norm": float(np.nanmax(repl)),
        "epoch_of_max_repl_norm": int(epoch[np.nanargmax(repl)]),
        "step_norm_max_epoch0": float(step[epoch == 0][0]) if (epoch == 0).any() else None,
        "attr_norm_max_epoch1": float(attr[epoch == 1][0]) if (epoch == 1).any() else None,
        "max_bbox_area_ratio_over_epochs": float(np.nanmax(ratio)),
        "epoch_of_max_bbox_area_ratio": int(epoch[np.nanargmax(ratio)]),
        "first_epoch_bbox_area_ratio_gt_10": first(ratio > 10),
        "first_epoch_far_count_gt_0": first(far > 0),
        "final_far_count": int(far[-1]),
        "updates_ratio_gt_10": int((ratio > 10).sum()),
        # Mesh of the state each field is evaluated on.
        "core_boxes_median": float(np.nanmedian(col("before_core_boxes"))),
        "grid_boxes_per_dim_median": float(np.nanmedian(col("before_grid_boxes_per_dim"))),
        "grid_boxes_per_dim_max": int(np.nanmax(col("before_grid_boxes_per_dim"))),
    }


def main() -> None:
    layout = Layout(CFG)
    core_fraction = float(CFG["metrics"]["core_fraction"])
    far_multiple = float(CFG["metrics"]["far_radius_multiple"])
    plot_seed = int(CFG["plot_seed"])
    per_run, per_case, escaped, landmarks = [], [], [], {}

    for dataset in CFG["datasets"]:
        graph, _ = load_fixed_inputs(CFG, dataset)
        weighted_degree = np.asarray(graph.sum(axis=1)).ravel()
        degree = np.diff(graph.indptr)
        d99 = float(np.percentile(weighted_degree, 99))
        percentile = (np.argsort(np.argsort(weighted_degree, kind="stable"), kind="stable")
                      / max(len(weighted_degree) - 1, 1))
        for variant in CFG["variants"]:
            hashes, rows = [], []
            for seed in CFG["seeds"]:
                path = layout.embedding(dataset, variant, seed)
                if not path.exists():
                    continue
                y = np.load(path)
                digest = sha256_array(y)
                if digest != read_json(layout.metadata(dataset, variant, seed))["embedding_sha256"]:
                    raise SystemExit(f"{path}: embedding does not match its metadata")
                hashes.append(digest)
                rows.append({"dataset": dataset, "variant": variant, "seed": seed, "embedding_sha256": digest,
                             **bbox_metrics(y, core_fraction, far_multiple)})
                landmarks[f"{dataset}/{variant}/{seed}"] = trace_landmarks(layout.trace(dataset, variant, seed))
            if not rows:
                continue
            per_run += rows
            ref = next((r for r in rows if r["seed"] == plot_seed), rows[0])
            case = {"dataset": dataset, "variant": variant, "n_seeds": len(rows),
                    "seeds_bitwise_identical": len(set(hashes)) == 1}
            for key in ("bbox_area", "core_bbox_area", "bbox_area_ratio", "bbox_extent_ratio",
                        "radius_max_over_core", "far_count", "far_pct"):
                values = np.asarray([r[key] for r in rows], dtype=float)
                case[key] = float(ref[key])
                case[f"{key}_min_over_seeds"] = float(values.min())
                case[f"{key}_max_over_seeds"] = float(values.max())
            case.update({f"trace_{k}": v for k, v in landmarks.get(f"{dataset}/{variant}/{ref['seed']}", {}).items()})

            y = np.load(layout.embedding(dataset, variant, int(ref["seed"])))
            _, radius, core_radius = core_mask(y, core_fraction)
            far_ids = np.flatnonzero(radius > far_multiple * core_radius)
            case.update({
                "weighted_degree_median_all": float(np.median(weighted_degree)),
                "weighted_degree_p99": d99,
                "weighted_degree_median_far": float(np.median(weighted_degree[far_ids])) if len(far_ids) else None,
                "far_share_above_d99": float(np.mean(weighted_degree[far_ids] > d99)) if len(far_ids) else None,
                "far_median_degree_percentile": float(np.median(percentile[far_ids])) if len(far_ids) else None,
            })
            per_case.append(case)
            escaped += [{"dataset": dataset, "variant": variant, "seed": int(ref["seed"]), "point_id": int(i),
                         "radius_over_core": float(radius[i] / core_radius),
                         "weighted_degree": float(weighted_degree[i]), "degree": int(degree[i]),
                         "weighted_degree_percentile": float(percentile[i])} for i in far_ids]

    results = layout.results
    write_csv(results / "bbox_summary.csv", per_run)
    write_csv(results / "case_summary.csv", per_case)
    write_csv(results / "escaped_points.csv", escaped)
    write_json(results / "summary.json", {"core_fraction": core_fraction, "far_radius_multiple": far_multiple,
                                          "cases": per_case, "trace_landmarks": landmarks})
    for case in per_case:
        print(f"{case['dataset']:12s} {case['variant']:22s} area_ratio={case['bbox_area_ratio']:10.2f} "
              f"far={int(case['far_count']):5d} ({case['far_pct']:.3f}%) "
              f"peak={case.get('trace_max_bbox_area_ratio_over_epochs', float('nan')):8.1f} "
              f"seeds_identical={case['seeds_bitwise_identical']}")


if __name__ == "__main__":
    main()
