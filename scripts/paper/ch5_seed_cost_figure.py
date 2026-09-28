#!/usr/bin/env python3
"""Appendix figure fig:appendix-cpu-seed-cost: CPU runtime cost of seeded execution.

Each point is one dataset's seeded / unseeded ratio of five-run median end-to-end
times; diamonds are medians across datasets. Writes
paper/build/figures/figure_C_cpu_seeded_time_cost.pdf and
paper/build/figures/data/paper_figure_seed_cost_data.csv.
"""
from __future__ import annotations

from statistics import median

import _e2e
import _style as style
from _common import parse_args, write_csv


def plot_seed_cost(rows, stem, formats):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    ratios = [row["seeded_over_unseeded"] for row in rows]
    limits = (min(0.45, min(ratios) * 0.85), max(2.2, max(ratios) * 1.12))
    fig, ax = plt.subplots(figsize=(style.PAPER_TEXT_WIDTH_IN, 1.85))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.95, bottom=0.27)
    labels = ["umap-learn", "ibUMAP CPU"]
    colors = {"umap_learn": style.ORANGE, "ibumap_cpu": style.BLUE}
    for position, algorithm in enumerate(("umap_learn", "ibumap_cpu")):
        part = [row for row in rows if row["algorithm"] == algorithm]
        ax.scatter([row["seeded_over_unseeded"] for row in part],
                   [position + style.deterministic_jitter(row["dataset"]) for row in part],
                   s=18, color=colors[algorithm], edgecolors=colors[algorithm], linewidths=0.5, alpha=0.83, zorder=3)
        ax.scatter([median(row["seeded_over_unseeded"] for row in part)], [position],
                   s=42, marker="D", color=colors[algorithm], edgecolors="white", linewidths=0.7, zorder=4)
    ax.axvline(1, color=style.REFERENCE_COLOR, linestyle="--", linewidth=0.85, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.xaxis.set_major_locator(FixedLocator((0.5, 1, 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Seeded / unseeded end-to-end time")
    ax.grid(axis="x", which="major", color=style.GRID_COLOR, alpha=0.65, linewidth=0.4)
    ax.tick_params(which="both", width=0.6, pad=2.0, direction="out")
    ax.tick_params(which="major", length=3.0)
    ax.tick_params(which="minor", length=1.7)
    ax.spines[["top", "right"]].set_visible(False)
    style.save_figure(fig, stem, formats)


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    data = _e2e.load(paths)
    _, rows = _e2e.runtime_rows(data)
    (paths.figures / "data").mkdir(exist_ok=True)
    write_csv(paths.figures / "data" / "paper_figure_seed_cost_data.csv", rows)
    style.configure_matplotlib()
    plot_seed_cost(rows, paths.figures / "figure_C_cpu_seeded_time_cost", args.formats)
    for algorithm in ("umap_learn", "ibumap_cpu"):
        values = [r["seeded_over_unseeded"] for r in rows if r["algorithm"] == algorithm]
        print(f"{algorithm}: n={len(values)}, median seeded/unseeded time={median(values):.6f}x")


if __name__ == "__main__":
    main()
