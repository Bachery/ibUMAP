#!/usr/bin/env python3
"""Section 6 figure fig:braque-repeatability: two unseeded UMAP reruns on the same frozen BRAQUE input.

Three separate, untitled panels (equal height, identical plotting area):
  ch6_run1.pdf     run 1 colored by its HDBSCAN clusters
  ch6_run2.pdf     run 2, rotated/reflected and translated onto run 1 for display (no scaling),
                   clusters matched to run 1 by maximum overlap over all cells
  ch6_changes.pdf  run-1 coordinates colored by how each cell's assignment changed (legend on the right)

Input: paper/data/braque/{plot_data.npz, panels.json}. The script re-derives the matched
labels, change categories and display alignment from the frozen coordinates and labels,
checks them against the frozen arrays, and writes the pair metrics quoted in the Section 6
caption and appendix to paper/build/figures/data/ch6_pair_metrics.json.
"""
from __future__ import annotations

import colorsys

import numpy as np

import _style as style
from _common import check_group, configure_matplotlib_environment, parse_args, read_json, write_json


def match_labels(reference, candidate):
    """Maximum-overlap one-to-one matching; noise stays noise, unmatched clusters get new IDs."""
    from scipy.optimize import linear_sum_assignment

    refs = np.unique(reference[reference >= 0])
    cands = np.unique(candidate[candidate >= 0])
    mapping = {-1: -1}
    if len(refs) and len(cands):
        mask = (reference >= 0) & (candidate >= 0)
        counts = np.zeros((len(refs), len(cands)), dtype=np.int64)
        np.add.at(counts, (np.searchsorted(refs, reference[mask]), np.searchsorted(cands, candidate[mask])), 1)
        rows, cols = linear_sum_assignment(-counts)
        mapping.update({int(cands[c]): int(refs[r]) for r, c in zip(rows, cols) if counts[r, c] > 0})
    next_id = int(refs.max()) + 1 if len(refs) else 0
    for c in cands:
        if int(c) not in mapping:
            mapping[int(c)] = next_id
            next_id += 1
    return np.array([mapping[int(c)] for c in candidate], dtype=np.int64), mapping


def rigid_align(reference, candidate):
    """Display only: translation + orthogonal rotation/reflection, no scaling."""
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    u, _, vt = np.linalg.svd((b - b.mean(0)).T @ (a - a.mean(0)))
    rotation = u @ vt
    translation = a.mean(0) - b.mean(0) @ rotation
    return b @ rotation + translation, rotation, translation


def label_metrics(a, b):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    aligned, _ = match_labels(a, b)
    both = (a != -1) & (b != -1)
    noise = (a == -1) != (b == -1)
    return {"ari_including_noise": float(adjusted_rand_score(a, b)),
            "ari_both_nonnoise": float(adjusted_rand_score(a[both], b[both])) if both.sum() >= 2 else None,
            "both_nonnoise_count": int(both.sum()), "ami_including_noise": float(adjusted_mutual_info_score(a, b)),
            "assignment_disagreement": float(np.mean(a != aligned)),
            "cluster_to_cluster_disagreement": float(np.mean(both & (a != aligned))),
            "noise_status_disagreement": float(noise.mean()),
            "cluster_to_noise_count": int(np.sum((a != -1) & (b == -1))),
            "noise_to_cluster_count": int(np.sum((a == -1) & (b != -1)))}


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    check_group(paths, "braque")
    root = paths.data / "braque"
    meta = read_json(root / "panels.json")
    p = meta["render_parameters"]
    with np.load(root / "plot_data.npz", allow_pickle=False) as data:
        first = data["reference_coordinates"]
        second = data["candidate_coordinates_raw"]
        labels_first = data["reference_labels"]
        labels_second = data["candidate_labels"]
        frozen = {k: data[k] for k in ("candidate_coordinates_aligned", "candidate_labels_matched",
                                        "change_category", "paint_order")}

    aligned, rotation, translation = rigid_align(first, second)
    matched, _ = match_labels(labels_first, labels_second)
    metrics = label_metrics(labels_first, labels_second)
    order = np.random.default_rng(p["plot_seed"]).permutation(len(first))
    category = np.zeros(len(first), dtype=np.int8)
    category[(labels_first >= 0) & (labels_second >= 0) & (labels_first != matched)] = 1
    category[(labels_first >= 0) & (labels_second == -1)] = 2
    category[(labels_first == -1) & (labels_second >= 0)] = 3
    np.testing.assert_allclose(aligned, frozen["candidate_coordinates_aligned"], rtol=0, atol=1e-9)
    assert np.array_equal(matched, frozen["candidate_labels_matched"])
    assert np.array_equal(category, frozen["change_category"]) and np.array_equal(order, frozen["paint_order"])
    for key, value in meta["metrics_full_data"].items():
        assert (value is None and metrics[key] is None) or np.isclose(metrics[key], value, rtol=0, atol=1e-12), key

    palette = {-1: (.66, .68, .70)}
    for index, label in enumerate(sorted(set(np.concatenate([labels_first, matched])) - {-1})):
        palette[int(label)] = colorsys.hls_to_rgb((.08 + index * .61803398875) % 1, .43 + .12 * (index % 2), .62)
    points = np.concatenate([first, aligned])
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    halfspan = max(float(np.max(upper - lower)), 1e-8) * .525
    xlim, ylim = (center[0] - halfspan, center[0] + halfspan), (center[1] - halfspan, center[1] + halfspan)

    configure_matplotlib_environment()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    size = p["panel_size"]

    def canvas(with_legend=False):
        width = size + (p["legend_width"] if with_legend else 0)
        fig = plt.figure(figsize=(width, size))
        margin = .035  # inches; the same physical plotting area in all three files
        ax = fig.add_axes([margin / width, margin / size, (size - 2 * margin) / width, (size - 2 * margin) / size])
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        return fig, ax

    def export(fig, stem, legend=None):
        fig.canvas.draw()
        if legend is not None:
            bounds = legend.get_window_extent(fig.canvas.get_renderer())
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.x1 or bounds.y1 > fig.bbox.y1:
                raise ValueError("Legend exceeds page bounds; increase legend_width")
        # dpi also sets the resolution of the rasterized point layers inside the PDF.
        style.save_figure(fig, paths.figures / stem, args.formats, dpi=p["dpi"], facecolor="white")

    for coords, labels, stem in ((first, labels_first, "ch6_run1"), (aligned, matched, "ch6_run2")):
        fig, ax = canvas()
        ax.scatter(coords[order, 0], coords[order, 1], s=p["point_size"],
                   c=[palette[int(v)] for v in labels[order]], alpha=.8, linewidths=0, rasterized=True)
        export(fig, stem)

    fig, ax = canvas(with_legend=True)
    change_colors = ["#CCD1D5", "#CB6D26", "#79509A", "#19728B"]
    markers = ["o", "o", "^", "o"]
    for kind in range(4):
        ii = order[category[order] == kind]
        options = ({"color": change_colors[kind], "linewidths": 0} if kind < 2
                   else {"facecolors": "none", "edgecolors": change_colors[kind], "linewidths": .16})
        ax.scatter(first[ii, 0], first[ii, 1], marker=markers[kind],
                   s=p["point_size"] * .65 if kind == 0 else p["change_size"] if kind == 1 else p["noise_size"],
                   alpha=.45 if kind == 0 else .8, rasterized=True, **options)
    handles = [Line2D([], [], linestyle="", marker=markers[k], markersize=4.2,
                      markerfacecolor=change_colors[k] if k < 2 else "none", markeredgecolor=change_colors[k],
                      markeredgewidth=.7, label=label)
               for k, label in enumerate(("Unchanged", "Cluster → cluster", "Cluster → noise", "Noise → cluster"))]
    legend = fig.legend(handles=handles, loc="center left", bbox_to_anchor=((size + .025) / fig.get_figwidth(), .5),
                        frameon=False, borderaxespad=0, handletextpad=.5, labelspacing=1.1, fontsize=8)
    export(fig, "ch6_changes", legend)

    (paths.figures / "data").mkdir(exist_ok=True)
    write_json(paths.figures / "data" / "ch6_pair_metrics.json", {
        "cells": len(first), "runs": [p["left_repeat"] + 1, p["right_repeat"] + 1],
        "metrics": metrics, "category_counts": {str(k): int(np.sum(category == k)) for k in range(4)},
        "caption_percentages": {
            "cluster_to_cluster": round(100 * metrics["cluster_to_cluster_disagreement"], 2),
            "noise_status": round(100 * metrics["noise_status_disagreement"], 2),
            "total_changed": round(100 * metrics["assignment_disagreement"], 2),
            "ari_both_nonnoise": round(metrics["ari_both_nonnoise"], 3)},
        "rotation": rotation.tolist(), "translation": translation.tolist()})
    print(f"ch6_panels.py: wrote ch6_run1, ch6_run2, ch6_changes ({len(first):,} cells)")


if __name__ == "__main__":
    main()
