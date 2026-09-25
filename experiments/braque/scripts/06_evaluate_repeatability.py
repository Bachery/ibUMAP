#!/usr/bin/env python3
"""Validate a completed batch and compute label agreement on all cells and sampled geometry / quality metrics.

Writes results/repeatability/<run-id>/analysis/<analysis-id>/: run_metrics.csv,
pairwise_metrics.csv (all 10 unordered run pairs per profile), summary.csv,
cost_ratios.csv, validation.json, metadata.json, SUMMARY.md, table_rows.tex.
Returns exit code 2 if the seeded profiles are not bitwise repeatable.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from _common import (PROFILES, DISPLAY, LIMITATIONS, completed, environment, label_metrics,
                         load_experiment, now, run_root, safe_name, seal, sha, write_csv, write_json)


def neighbors(X, k):
    # X=None excludes each query observation itself, including tied/duplicate coordinates.
    return NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X).kneighbors(return_distance=False)


def overlap(a, b):
    return float(np.mean([len(np.intersect1d(x, y)) / a.shape[1] for x, y in zip(a, b)]))


def correlation(a, b):
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def stats(rows, key):
    values = [x[key] for x in rows if x[key] is not None]
    if not values:
        return {key + suffix: None for suffix in ("_median", "_min", "_max")}
    return {key + "_median": float(np.median(values)), key + "_min": float(np.min(values)), key + "_max": float(np.max(values))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--analysis-id", default="main")
    p.add_argument("--quality-cells", type=int, default=3000)
    p.add_argument("--structure-cells", type=int, default=2048)
    p.add_argument("--distance-pairs", type=int, default=200000)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--evaluation-seed", type=int, default=20260920)
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if min(a.quality_cells, a.structure_cells) < 4 or min(a.distance_pairs, a.neighbors) < 1:
        raise ValueError("Invalid metric sample sizes")
    root = run_root(a.run_id)
    exp = load_experiment(root)
    cfg = exp["contract"]["config"]
    destination = root / "analysis" / safe_name(a.analysis_id)
    if destination.exists():
        raise FileExistsError("Analysis already exists; use a new --analysis-id")
    sources = {}
    for profile in PROFILES:
        for repeat in range(cfg["repeats"]):
            value = completed(root, profile, repeat, exp["contract_sha256"])
            if value is None:
                raise ValueError(f"Incomplete experiment: missing {profile} repeat {repeat}; resume stage 20 first")
            sources[profile, repeat] = value
    print(f"Validated {len(sources)} completed runs; output: {destination}")
    if not a.run:
        print("Dry run; add --run to evaluate.")
        return 0
    X = np.load(root / "inputs" / "umap_input.npy", allow_pickle=False)
    rng = np.random.default_rng(a.evaluation_seed)
    quality_idx = np.sort(rng.choice(len(X), min(len(X), a.quality_cells), replace=False))
    structure_idx = np.sort(rng.choice(len(X), min(len(X), a.structure_cells), replace=False))
    left = rng.integers(0, len(X), a.distance_pairs)
    right = (left + rng.integers(1, len(X), a.distance_pairs)) % len(X)
    quality_k = min(a.neighbors, (len(quality_idx) - 1) // 2)
    structure_k = min(a.neighbors, len(structure_idx) - 1)
    high_neighbors = neighbors(X[structure_idx], structure_k)
    run_rows, pair_rows, summary_rows, source_rows = [], [], [], []
    exact_checks = {}
    # Keep only one profile's arrays resident at a time.
    for profile in PROFILES:
        arrays, labels, local_neighbors, distances = {}, {}, {}, {}
        group_rows = []
        for repeat in range(cfg["repeats"]):
            directory, record = sources[profile, repeat]
            Y = np.load(directory / "embedding.npy", allow_pickle=False)
            L = np.load(directory / "labels.npy", allow_pickle=False)
            if Y.shape != (len(X), 2) or L.shape != (len(X),) or not np.isfinite(Y).all():
                raise ValueError(f"Shape/finite check failed: {directory}")
            if sha(directory / "embedding.npy") != record["embedding_sha256"] or sha(directory / "labels.npy") != record["labels_sha256"]:
                raise ValueError(f"Output hash differs from run metadata: {directory}")
            arrays[repeat], labels[repeat] = Y, L
            local_neighbors[repeat] = neighbors(Y[structure_idx], structure_k)
            distances[repeat] = np.linalg.norm(Y[left].astype(np.float64) - Y[right], axis=1)
            row = {"profile": profile, "repeat": repeat, "cell_count": len(X),
                   "cluster_count": int(np.sum(np.unique(L) >= 0)), "noise_fraction": float(np.mean(L == -1)),
                   "trustworthiness": float(trustworthiness(X[quality_idx], Y[quality_idx], n_neighbors=quality_k)),
                   "high_dim_neighbor_overlap": overlap(high_neighbors, local_neighbors[repeat]),
                   **record["timing_seconds"], "embedding_sha256": record["embedding_sha256"], "labels_sha256": record["labels_sha256"]}
            group_rows.append(row)
            source_rows.append({"profile": profile, "repeat": repeat, "directory": str(directory.relative_to(root)),
                                "run_json_sha256": sha(directory / "run.json"),
                                "ledger_sha256": sha(directory / "SHA256SUMS")})
        pairs = []
        for i, j in itertools.combinations(range(cfg["repeats"]), 2):
            equal = bool(np.array_equal(arrays[i], arrays[j]))
            row = {"profile": profile, "repeat_left": i, "repeat_right": j, "embedding_array_equal": equal,
                   "labels_array_equal": bool(np.array_equal(labels[i], labels[j])),
                   "embedding_max_abs_difference": float(np.max(np.abs(arrays[i].astype(np.float64) - arrays[j]))),
                   "run_neighbor_overlap": overlap(local_neighbors[i], local_neighbors[j]),
                   "distance_spearman": correlation(distances[i], distances[j]), **label_metrics(labels[i], labels[j])}
            pairs.append(row)
        exact_checks[profile] = {"all_embeddings_identical": all(x["embedding_array_equal"] for x in pairs),
                                 "all_labels_identical": all(x["labels_array_equal"] for x in pairs)}
        summary = {"profile": profile, "run_count": len(group_rows), "pair_count": len(pairs),
                   **exact_checks[profile], "unique_embedding_hashes": len({r["embedding_sha256"] for r in group_rows}),
                   "unique_label_hashes": len({r["labels_sha256"] for r in group_rows})}
        for key in ("embedding_fit", "hdbscan_fit", "embedding_plus_hdbscan", "cluster_count", "noise_fraction", "trustworthiness", "high_dim_neighbor_overlap"):
            summary.update(stats(group_rows, key))
        for key in ("distance_spearman", "run_neighbor_overlap", "ari_including_noise", "ari_both_nonnoise", "assignment_disagreement", "noise_status_disagreement"):
            summary.update(stats(pairs, key))
        run_rows.extend(group_rows)
        pair_rows.extend(pairs)
        summary_rows.append(summary)
        print(f"{profile}: median assignment disagreement {summary['assignment_disagreement_median']:.2%}; "
              f"embedding fit {summary['embedding_fit_median']:.3f}s", flush=True)
    by_profile = {r["profile"]: r for r in summary_rows}
    ratios = []
    for family in ("umap", "ibumap"):
        for metric in ("embedding_fit", "embedding_plus_hdbscan"):
            numerator = by_profile[family + "_seeded"][metric + "_median"]
            denominator = by_profile[family + "_unseeded"][metric + "_median"]
            ratios.append({"comparison": family + " seeded/unseeded", "metric": metric,
                           "numerator_seconds": numerator, "denominator_seconds": denominator, "ratio": numerator / denominator})
    for metric in ("embedding_fit", "embedding_plus_hdbscan"):
        numerator, denominator = [by_profile[p][metric + "_median"] for p in ("umap_seeded", "ibumap_seeded")]
        ratios.append({"comparison": "seeded UMAP/ibUMAP speedup", "metric": metric,
                       "numerator_seconds": numerator, "denominator_seconds": denominator, "ratio": numerator / denominator})
    seeded_pass = all(all(exact_checks[p].values()) for p in ("umap_seeded", "ibumap_seeded"))
    metadata = {"created_at": now(), "protocol": exp["protocol"], "contract_sha256": exp["contract_sha256"],
                "experiment_json_sha256": sha(root / "experiment.json"), "analysis_id": a.analysis_id,
                "parameters": {"quality_cells": len(quality_idx), "structure_cells": len(structure_idx),
                               "distance_pairs": len(left), "quality_k": quality_k, "structure_k": structure_k,
                               "evaluation_seed": a.evaluation_seed}, "sources": source_rows,
                "script_sha256": sha(Path(__file__)), "common_sha256": sha(Path(__file__).with_name("_common.py")),
                "evaluation_environment": environment(),
                "geometry_scope": "Trustworthiness and neighbor overlap use induced fixed point subsets; distance pairs sampled from full fitted embedding, with replacement, excluding self-pairs.",
                "noise_policy": "ARI includes noise as one label; complementary ARI on cells non-noise in BOTH runs; Hungarian matching reserves -1 for noise.",
                "summary_policy": "Medians/min/max over all measured repeats or all unordered run pairs; ranges are not confidence intervals.",
                "smoke_subset": cfg["smoke_subset"], "limitations": LIMITATIONS}
    validation = {"complete": True, "checksums_pass": True, "seeded_repeatability_pass": seeded_pass,
                  "exact_checks": exact_checks, "cost_evaluation": "Descriptive ratios only, no required direction or pass threshold.",
                  "quality_evaluation": "Report fidelity values and seeded ibUMAP-minus-UMAP differences; no post-hoc noninferiority threshold.",
                  "seeded_trustworthiness_difference": by_profile["ibumap_seeded"]["trustworthiness_median"] - by_profile["umap_seeded"]["trustworthiness_median"],
                  "seeded_neighbor_overlap_difference": by_profile["ibumap_seeded"]["high_dim_neighbor_overlap_median"] - by_profile["umap_seeded"]["high_dim_neighbor_overlap_median"]}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "evaluation_indices.npz", quality=quality_idx, structure=structure_idx, distance_left=left, distance_right=right)
    for name, rows in (("run_metrics.csv", run_rows), ("pairwise_metrics.csv", pair_rows), ("summary.csv", summary_rows), ("cost_ratios.csv", ratios)):
        write_csv(destination / name, rows)
    write_json(destination / "metadata.json", metadata)
    write_json(destination / "validation.json", validation)
    lines = ["# BRAQUE repeatability — " + ("SMOKE SUBSET" if cfg["smoke_subset"] else "full frozen input"), "",
             f"Cells: {len(X):,}; measured repeats/profile: {cfg['repeats']}. All runs retained after within-process warmup.", "",
             "| Profile | Embedding fit (s) | + HDBSCAN (s) | Distance rho | ARI | Assignment disagreement | Exact coordinates / labels |",
             "|---|---:|---:|---:|---:|---:|---|"]
    tex = []
    for r in summary_rows:
        rho = r['distance_spearman_median']
        rho_text = "NA" if rho is None else f"{rho:.3f}"
        lines.append(f"| {DISPLAY[r['profile']]} | {r['embedding_fit_median']:.3f} | {r['embedding_plus_hdbscan_median']:.3f} | {rho_text} | {r['ari_including_noise_median']:.3f} | {r['assignment_disagreement_median']:.2%} | {r['all_embeddings_identical']} / {r['all_labels_identical']} |")
        tex.append(f"{DISPLAY[r['profile']]} & {r['embedding_fit_median']:.2f} & {rho_text} & {r['ari_including_noise_median']:.3f} & {100*r['assignment_disagreement_median']:.1f}\\% " + r"\\")
    lines += ["", "Time columns exclude preprocessing/LNS, I/O, imports, warmup, estimator construction and diagnostics. The second adds the two timed calls; it is not full BRAQUE runtime.",
              "Geometry/label columns summarize all unordered pairs; labels use all cells. Quality metrics and all run/pair records are in CSV files.",
              "", f"Seeded coordinate-and-label repeatability passed: {seeded_pass}.", "", "## Descriptive cost ratios", ""]
    lines += [f"- {r['comparison']} ({r['metric']}): {r['ratio']:.4f}." for r in ratios]
    lines += ["", "## Quality and scope", "", f"Seeded trustworthiness difference (ibUMAP − UMAP): {validation['seeded_trustworthiness_difference']:+.6f}.",
              f"Seeded input-neighbor overlap difference: {validation['seeded_neighbor_overlap_difference']:+.6f}.", ""]
    lines += ["- " + s for s in LIMITATIONS]
    (destination / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (destination / "table_rows.tex").write_text("% Profile & embedding seconds & distance rho & ARI & assignment disagreement\n" + "\n".join(tex) + "\n")
    seal(destination, sorted(p.name for p in destination.iterdir() if p.is_file()))
    print(f"Saved {destination}")
    if not seeded_pass:
        print("WARNING: seeded repeatability failed; records preserved for investigation.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
