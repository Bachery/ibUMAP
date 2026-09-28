#!/usr/bin/env python3
"""Diagnostic figures (not in the paper): a combined 1x3 run-pair figure and per-run cost / quality.

The paper's three separate panels are drawn by scripts/paper/ch6_panels.py from
the data written by 08_export_paper_data.py.
"""
from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path

import numpy as np

from _common import (PROFILES, DISPLAY, label_metrics, load_experiment, match_labels, now,
                         read_csv, read_json, rigid_align, run_root, safe_name, seal, sha, verify, write_csv, write_json)


def palette(labels):
    """Explicit stable categorical colors; matched labels share a color, noise is grey."""
    mapping = {-1: (.66, .68, .70)}
    for i, label in enumerate(sorted(set(labels) - {-1})):
        mapping[int(label)] = colorsys.hls_to_rgb((.08 + i * .61803398875) % 1, .43 + .12 * (i % 2), .62)
    return mapping


def save(fig, directory, stem, dpi):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(directory / f"{stem}.{ext}", dpi=dpi, facecolor="white")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--analysis-id", default="main")
    p.add_argument("--figure-id", default="main")
    p.add_argument("--left-repeat", type=int, default=0)
    p.add_argument("--right-repeat", type=int, default=1)
    p.add_argument("--max-points", type=int, default=0, help="0 plots every cell; otherwise a fixed uniform subset")
    p.add_argument("--plot-seed", type=int, default=20260921)
    p.add_argument("--width", type=float, default=7.0, help="Figure width in inches")
    p.add_argument("--height", type=float, default=2.65)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.left_repeat < 0 or a.left_repeat >= a.right_repeat or a.max_points < 0 or min(a.width, a.height, a.dpi) <= 0:
        raise ValueError("Require 0 <= left-repeat < right-repeat, nonnegative max-points and positive figure dimensions")
    root = run_root(a.run_id)
    exp = load_experiment(root)
    analysis = root / "analysis" / safe_name(a.analysis_id)
    verify(analysis)
    meta = read_json(analysis / "metadata.json")
    if meta["contract_sha256"] != exp["contract_sha256"] or meta["experiment_json_sha256"] != sha(root / "experiment.json"):
        raise ValueError("Analysis belongs to a different experiment")
    destination = root / "figures" / safe_name(a.figure_id)
    if destination.exists():
        raise FileExistsError("Figure directory exists; choose a new --figure-id")
    selected = []
    for repeat in (a.left_repeat, a.right_repeat):
        source = next((s for s in meta["sources"] if s["profile"] == "umap_unseeded" and s["repeat"] == repeat), None)
        if source is None:
            raise ValueError(f"No evaluated unseeded UMAP repeat {repeat}")
        path = root / source["directory"]
        verify(path)
        if sha(path / "run.json") != source["run_json_sha256"] or sha(path / "SHA256SUMS") != source["ledger_sha256"]:
            raise ValueError("Source changed after evaluation")
        selected.append(path)
    print(f"Main figure: unseeded UMAP repeats {a.left_repeat}, {a.right_repeat}, and their changes; output: {destination}")
    if not a.run:
        print("Dry run; add --run to render.")
        return 0
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    Y1, Y2 = [np.load(p / "embedding.npy", allow_pickle=False) for p in selected]
    L1, L2 = [np.load(p / "labels.npy", allow_pickle=False) for p in selected]
    aligned, rotation, translation = rigid_align(Y1, Y2)
    mapped, mapping = match_labels(L1, L2)
    metrics = label_metrics(L1, L2)
    pair = next(r for r in read_csv(analysis / "pairwise_metrics.csv") if r["profile"] == "umap_unseeded"
                and int(r["repeat_left"]) == a.left_repeat and int(r["repeat_right"]) == a.right_repeat)
    if not np.isclose(metrics["assignment_disagreement"], float(pair["assignment_disagreement"]), atol=1e-12):
        raise ValueError("Plot and evaluation matching disagree")
    n = len(Y1)
    rng = np.random.default_rng(a.plot_seed)
    indices = (np.sort(rng.choice(n, min(a.max_points, n), replace=False)) if a.max_points else np.arange(n))
    order = rng.permutation(indices)  # Same paint order in both cluster panels.
    categories = np.zeros(n, dtype=np.int8)
    categories[(L1 != -1) & (L2 != -1) & (L1 != mapped)] = 1
    categories[(L1 != -1) & (L2 == -1)] = 2
    categories[(L1 == -1) & (L2 != -1)] = 3
    colors = palette(np.concatenate([L1, mapped]))
    fig, axes = plt.subplots(1, 3, figsize=(a.width, a.height))
    fig.subplots_adjust(left=.015, right=.995, top=.80, bottom=.18, wspace=.10)
    for ax, Y, L, repeat, original in zip(axes[:2], (Y1, aligned), (L1, mapped), (a.left_repeat, a.right_repeat), (L1, L2)):
        ax.scatter(Y[order, 0], Y[order, 1], c=[colors[int(v)] for v in L[order]], s=1.3,
                   alpha=.75, linewidths=0, rasterized=True)
        clusters = np.sum(np.unique(original) >= 0)
        ax.set_title(f"({'a' if ax is axes[0] else 'b'}) Run {repeat + 1}\n{clusters} clusters; {np.mean(original == -1):.1%} noise", pad=5)
    change_colors = ["#CFD3D6", "#C66A25", "#70499B", "#176B87"]
    markers = ["o", "o", "^", "o"]
    for category in range(4):
        ii = indices[categories[indices] == category]
        kwargs = ({"color": change_colors[category], "linewidths": 0} if category < 2
                  else {"facecolors": "none", "edgecolors": change_colors[category], "linewidths": .35})
        axes[2].scatter(Y1[ii, 0], Y1[ii, 1], s=1.4 if category < 2 else 5,
                        alpha=.38 if category == 0 else .8, marker=markers[category], rasterized=True, **kwargs)
    axes[2].set_title(f"(c) Run {a.left_repeat + 1} → {a.right_repeat + 1}\n{metrics['assignment_disagreement']:.1%} changed; ARI {metrics['ari_including_noise']:.3f}", pad=5)
    full = np.concatenate([Y1, aligned])
    lo, hi = full.min(0), full.max(0)
    pad = np.maximum(hi - lo, 1e-8) * .045
    for ax in axes:
        ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]), xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    legend = [Line2D([], [], linestyle="", marker=markers[i], markersize=4,
                     markerfacecolor=change_colors[i] if i < 2 else "none", markeredgecolor=change_colors[i],
                     markeredgewidth=.7, label=label)
              for i, label in enumerate(("Unchanged", "Cluster → cluster", "Cluster → noise", "Noise → cluster"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False,
               handletextpad=.3, columnspacing=1.25, fontsize=7)
    if exp["contract"]["config"]["smoke_subset"]:
        fig.text(.5, .99, "SMOKE TEST — SUBSET", ha="center", va="top", color="#9A3412", fontsize=8)
    destination.mkdir(parents=True)
    save(fig, destination, "ch6_braque_reruns_1x3", a.dpi)
    plt.close(fig)
    # Small supplementary figure retains all measured repetitions and quality tradeoffs.
    runs = read_csv(analysis / "run_metrics.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8))
    fig.subplots_adjust(left=.19, right=.975, top=.83, bottom=.16, wspace=.35)
    profile_colors = ["#BC7930", "#89502A", "#6BA1BC", "#26617D"]
    for ax, key, title in zip(axs, ("embedding_fit", "trustworthiness", "high_dim_neighbor_overlap"),
                              ("Embedding fit (s)", "Trustworthiness", "Input-neighbor overlap")):
        for i, profile in enumerate(PROFILES):
            values = [float(r[key]) for r in runs if r["profile"] == profile]
            ax.scatter(values, i + np.linspace(-.13, .13, len(values)), color=profile_colors[i], s=11, alpha=.65)
            ax.scatter(np.median(values), i, color=profile_colors[i], marker="D", s=27, zorder=3)
        ax.set_title(title)
        ax.set_yticks(range(4), [DISPLAY[x] if ax is axs[0] else "" for x in PROFILES])
        ax.set_ylim(3.5, -.5)
        ax.grid(axis="x", color="#E4E6E8", linewidth=.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if key == "embedding_fit":
            ax.set_xlim(left=0)
        else:
            ax.set_xlim(0, 1)
    fig.text(.99, .015, "Dots: measured runs; diamonds: medians. Quality: fixed induced subsets.", ha="right", fontsize=7)
    if exp["contract"]["config"]["smoke_subset"]:
        fig.text(.5, .99, "SMOKE TEST — SUBSET", ha="center", va="top", color="#9A3412", fontsize=8)
    save(fig, destination, "ch6_cost_and_quality", a.dpi)
    plt.close(fig)
    write_csv(destination / "cluster_color_mapping.csv", [{"candidate_label": c, "matched_label": r,
               "red": colors[r][0], "green": colors[r][1], "blue": colors[r][2]} for c, r in sorted(mapping.items())])
    np.savez_compressed(destination / "plot_data.npz", selected_indices=indices, paint_order=order,
                        reference_coordinates=Y1[indices], candidate_coordinates_raw=Y2[indices],
                        candidate_coordinates_aligned=aligned[indices], reference_labels=L1[indices],
                        candidate_labels=L2[indices], candidate_labels_matched=mapped[indices], change_category=categories[indices])
    caption = (f"Repeated unseeded UMAP embeddings in the BRAQUE-derived {exp['contract']['config']['sample']} workflow "
               f"({n:,} cells). Panels (a,b) show runs {a.left_repeat + 1} and {a.right_repeat + 1}, colored by HDBSCAN "
               "with maximum-overlap one-to-one label matching; noise is grey. Panel (b) is rigidly aligned to (a), "
               "allowing rotation/reflection and translation but no rescaling. HDBSCAN operates on the original coordinates. "
               "Panel (c) marks matched assignment changes on the coordinates of (a); noise transitions are separately marked. "
               f"All metrics use all {n:,} cells; {len(indices):,} cells are plotted. "
               f"Assignment disagreement is {100*metrics['assignment_disagreement']:.1f}%, noise-status disagreement "
               f"is {100*metrics['noise_status_disagreement']:.2f}%, and ARI is {metrics['ari_including_noise']:.3f}. "
               "Disagreement includes cluster splits and merges and does not measure biological classification error.")
    (destination / "caption.txt").write_text(caption + "\n")
    tex_escape = {"%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#"}
    (destination / "caption.tex").write_text("".join(tex_escape.get(c, c) for c in caption) + "\n")
    write_json(destination / "manifest.json", {"created_at": now(), "contract_sha256": exp["contract_sha256"],
               "analysis_metadata_sha256": sha(analysis / "metadata.json"), "analysis_ledger_sha256": sha(analysis / "SHA256SUMS"),
               "source_directories": [str(p.relative_to(root)) for p in selected], "full_cell_count": n,
               "plot_cell_count": len(indices), "selection_rule": "First two scheduled UMAP unseeded repeats by default; explicit overrides recorded.",
               "left_repeat": a.left_repeat, "right_repeat": a.right_repeat, "plot_seed": a.plot_seed,
               "rotation": rotation.tolist(), "translation": translation.tolist(), "scale": 1.0,
               "metrics_full_data": metrics, "distance_spearman": pair["distance_spearman"],
               "layout": "1x3", "width_inches": a.width, "height_inches": a.height, "dpi": a.dpi,
               "rendering_versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "script_sha256": sha(Path(__file__)), "common_sha256": sha(Path(__file__).with_name("_common.py")),
               "smoke_subset": exp["contract"]["config"]["smoke_subset"]})
    seal(destination, sorted(p.name for p in destination.iterdir() if p.is_file()))
    print(f"Saved main 1×3 figure and supplementary cost/quality figure as PDF, PNG and SVG in {destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
