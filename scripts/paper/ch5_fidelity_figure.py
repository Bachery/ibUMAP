#!/usr/bin/env python3
"""Appendix figure fig:e2e-fidelity: unseeded fidelity differences on the full paired sets.

Five metrics (TW, C, NP, RTA, distance Spearman) for CPU / umap-learn (66 datasets),
GPU / cuML (71) and GPU / TorchDR (66). Each point is one dataset's difference of
five-run means; diamonds are means of the paired differences. Writes
paper/build/figures/figure_D_unseeded_fidelity.pdf. The plotted values equal the
unseeded rows of paper/build/tables/ch5_fidelity_plot_data.csv (checked when present).
"""
from __future__ import annotations

import csv
from statistics import mean

import _e2e
import _style as style
from _common import expect_paper, parse_args

METRICS = (
    ("trustworthiness", "Trustworthiness"),
    ("continuity", "Continuity"),
    ("neighborhood_preservation", "Neighborhood preservation"),
    ("rta", "Random triplet accuracy"),
    ("distance_spearman", "Distance Spearman correlation"),
)
EXPECTED = {"cpu_umap": 66, "gpu_cuml": 71, "gpu_torchdr": 66}


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    data = _e2e.load(paths)
    rows = _e2e.quality_delta_rows(data, METRICS)
    for key, expected in EXPECTED.items():
        for metric, _ in METRICS:
            found = sum(r["comparison"] == key and r["metric"] == metric for r in rows)
            expect_paper(paths, found == expected, f"{key} {metric}: {found} datasets, paper {expected}")
    table_export = paths.tables / "ch5_fidelity_plot_data.csv"
    if table_export.exists():
        labels = {"CPU/UMAP": "cpu_umap", "GPU/cuML": "gpu_cuml", "GPU/TorchDR": "gpu_torchdr"}
        with table_export.open() as stream:
            exported = {(labels[r["comparison"]], r["dataset"], r["metric"]): float(r["paired_delta"])
                        for r in csv.DictReader(stream) if r["seed"] == "none"}
        assert len(exported) == len(rows)
        for row in rows:
            assert abs(row["paired_delta"] - exported[row["comparison"], row["dataset"], row["metric"]]) < 1e-12

    style.configure_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(style.PAPER_TEXT_WIDTH_IN, 3.6))
    grid = fig.add_gridspec(2, 6, left=0.16, right=0.99, top=0.94, bottom=0.13, hspace=0.70, wspace=0.95)
    axes = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    axes += [fig.add_subplot(grid[1, 3 * i:3 * i + 3]) for i in range(2)]
    positions = list(reversed(range(len(style.COMPARISONS))))
    for panel, (ax, (metric, title)) in enumerate(zip(axes, METRICS)):
        part = [r for r in rows if r["metric"] == metric]
        limit = max(abs(r["paired_delta"]) for r in part)
        padding = max(limit * 0.12, 1e-4)
        ax.set_xlim(-limit - padding, limit + padding)
        ax.set_ylim(-0.4, 2.4)
        ax.axvline(0, color=style.REFERENCE_COLOR, linestyle="--", linewidth=0.85, zorder=2)
        for position, comparison in zip(positions, style.COMPARISONS):
            values = [r for r in part if r["comparison"] == comparison["key"]]
            ax.scatter([r["paired_delta"] for r in values],
                       [position + style.deterministic_jitter(r["dataset"], 0.12) for r in values],
                       s=13, color=comparison["color"], edgecolors="none", alpha=0.70, zorder=3)
            ax.scatter([mean(r["paired_delta"] for r in values)], [position], marker="D", s=32,
                       color=comparison["color"], edgecolors="white", linewidths=0.55, zorder=4)
        ax.set_title(title, fontsize=7.6, pad=4)
        ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.65, linewidth=0.4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.5, width=0.6, pad=2)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_xlabel("ibUMAP minus baseline", fontsize=7)
        ax.set_yticks(positions, [c["label"] for c in style.COMPARISONS] if panel in (0, 3) else [""] * 3,
                      fontsize=6.6)
    style.save_figure(fig, paths.figures / "figure_D_unseeded_fidelity", args.formats)
    print(f"ch5_fidelity_figure.py: {len(rows)} paired points; wrote figure_D_unseeded_fidelity")


if __name__ == "__main__":
    main()
