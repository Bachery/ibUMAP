#!/usr/bin/env python3
"""Section 4 figure fig:mechanism-effects: per-dataset paired quality changes A->B ... G->H and A->H.

Five metrics (TW, C, NP, RTA, rho_D) on 59 datasets. Each point is one dataset's change
(later minus earlier variant, seed means); diamonds are medians; the right-hand numbers
count improving datasets. Axes are clipped at the 97.5th percentile of |change| per metric,
and clipped points are drawn as arrows at the edge. Writes
paper/build/figures/figure_ch4_mechanism_effects.pdf and
paper/build/tables/ch4_effect_plot_data.csv. Medians and counts are checked against
ch4_effect_statistics.csv when ch4_tables.py has been run.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

import _mechanism as mech
import _style as style
from _common import parse_args, write_json

STEM = "figure_ch4_mechanism_effects"
CLIP_PERCENTILE = 97.5
METRICS = [("trustworthiness", "TW"), ("continuity", "C"), ("neighborhood_preservation", "NP"),
           ("rta", "RTA"), ("distance_spearman", r"$\rho_D$")]
V = dict(A="umap_learn", B="sync_sampled", C="sync_expected", D="sync_direct", E="sync_direct_capped",
         F="sync_fft", G="sync_fft_guarded", H="ibumap_production")
STEPS = [("B--A", "A→B  synchrony", "A", "B"),
         ("C--B", "B→C  expectation", "B", "C"),
         ("D--C", "C→D  degree field", "C", "D"),
         ("E--D", "D→E  kernel cap", "D", "E"),
         ("F--E", "E→F  FFT", "E", "F"),
         ("G--F", "F→G  safeguards", "F", "G"),
         ("H--G", "G→H  production$^{*}$", "G", "H"),
         ("H--A", "A→H  net", "A", "H")]
ROWS = [0, 1, 2, 3, 4, 5, 6, 7.55]


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    raw, _, _ = mech.load_scores(paths)
    # Same values as paper/build/tables/ch4_dataset_means.csv: the manuscript figure was drawn from that
    # CSV, so the seed means are round-tripped through the identical CSV text (last-digit float agreement).
    per = pd.read_csv(io.StringIO(mech.per_dataset_means(raw).to_csv()))
    n_datasets = per.dataset.nunique()
    wide = per.pivot(index="dataset", columns="algorithm")
    stats_path = paths.tables / "ch4_effect_statistics.csv"
    stats = pd.read_csv(stats_path).set_index(["contrast", "metric"]) if stats_path.exists() else None

    plot_rows, summary = [], []
    for label, _, a, b in STEPS:
        for metric, _ in METRICS:
            d = wide[metric][V[b]] - wide[metric][V[a]]
            assert d.notna().all() and len(d) == n_datasets
            improved, worse = int((d > mech.TIE).sum()), int((d < -mech.TIE).sum())
            if stats is not None:
                ref = stats.loc[(label, metric)]
                np.testing.assert_allclose([d.mean(), d.median()], [ref["mean"], ref["median"]], atol=1e-12, rtol=0)
                assert (improved, worse) == (int(ref.improved), int(ref.worse)), (label, metric)
            summary.append(dict(contrast=label, metric=metric, median=d.median(), improved=improved, worse=worse))
            plot_rows += [dict(contrast=label, metric=metric, dataset=k, paired_delta=v) for k, v in d.items()]
    plot = pd.DataFrame(plot_rows)
    summary = pd.DataFrame(summary).set_index(["contrast", "metric"])

    style.configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(style.PAPER_TEXT_WIDTH_IN, 1.8), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.958, top=0.895, bottom=0.085, wspace=0.50)
    clipped = {}
    for ax, (metric, title) in zip(axes, METRICS):
        part = plot[plot.metric == metric]
        limit = float(np.percentile(part.paired_delta.abs(), CLIP_PERCENTILE)) * 1.08
        clipped[metric] = int((part.paired_delta.abs() > limit).sum())
        ax.set_xlim(-limit * 1.06, limit * 1.06)
        ax.axvline(0, color=style.REFERENCE_COLOR, linewidth=0.6, zorder=1)
        ax.axhline(6.78, color=style.GRID_COLOR, linewidth=0.5, zorder=0)
        for (label, _, _, _), y in zip(STEPS, ROWS):
            rows = part[part.contrast == label]
            x = rows.paired_delta.to_numpy()
            jitter = np.array([style.deterministic_jitter(k, 0.26) for k in rows.dataset])
            colour = np.where(x > mech.TIE, style.BLUE, np.where(x < -mech.TIE, style.ORANGE, "#999999"))
            inside = np.abs(x) <= limit
            ax.scatter(x[inside], y + jitter[inside], s=2.8, c=colour[inside], edgecolors="none", alpha=0.55,
                       zorder=2)
            for sign, marker in [(1, ">"), (-1, "<")]:
                out = (~inside) & (np.sign(x) == sign)
                ax.scatter(np.full(out.sum(), sign * limit * 1.03), y + jitter[out], s=5, marker=marker,
                           c=colour[out], edgecolors="none", alpha=0.8, zorder=2)
            s = summary.loc[(label, metric)]
            ax.scatter([s["median"]], [y], marker="D", s=13, color="#111111", edgecolors="white", linewidths=0.5,
                       zorder=4)
            ax.text(1.10, y, f"{int(s.improved)}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=5.8, color="#444444")
        ax.set_title(title, fontsize=7.6, pad=3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=5.6, width=0.5, length=2, pad=1.5)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+.2g}".replace("+0.", "+.").replace("-0.", "−.")))
    axes[0].set_yticks(ROWS, [row for _, row, _, _ in STEPS], fontsize=6.4)
    axes[0].set_ylim(ROWS[-1] + 0.55, -0.55)
    fig.text(0.999, 0.895, "n↑", ha="right", va="bottom", fontsize=5.8, color="#444444")
    # bbox_inches="tight" keeps the float height the manuscript was laid out with.
    style.save_figure(fig, paths.figures / STEM, args.formats, bbox_inches="tight", pad_inches=0.01)
    plot.to_csv(paths.tables / "ch4_effect_plot_data.csv", index=False)
    (paths.figures / "data").mkdir(exist_ok=True)
    write_json(paths.figures / "data" / "figure_ch4_summary.json",
               {"clip_percentile": CLIP_PERCENTILE, "clipped_points_per_metric": clipped, "points": len(plot)})
    print(f"ch4_effect_figure.py: {len(plot)} paired points; clipped per metric {clipped}")


if __name__ == "__main__":
    main()
