#!/usr/bin/env python3
"""Appendix C speed figures for both execution profiles (seeded and unseeded).

  figure_A_side_by_side.pdf                        fig:appendix-speedups
      dataset-level speedups, CPU / umap-learn (left) and GPU / cuML (right)
  figure_A_runtime_seeded_unseeded_side_by_side.pdf fig:appendix-runtime-profiles
      absolute end-to-end times of baseline and ibUMAP

Each point is one dataset. Speedup = median(baseline E2E seconds) / median(ibUMAP E2E seconds).
Runtime lines connect datasets in (sample count, name) order as a visual guide only.
Plotted rows: paper/build/figures/data/figure_A_data.csv.
"""
from __future__ import annotations

import _e2e
import _style as style
from _common import parse_args, write_csv

PANEL_GAP_IN = 0.2
PANEL_WIDTH_IN = (style.PAPER_TEXT_WIDTH_IN - PANEL_GAP_IN) / 2
PANEL_HEIGHT_IN = 2.2
LEGEND_SIZE_PT = 7.0
MARKER_SIZE_PT2 = 18.0
MARKER_EDGE_WIDTH_PT = 0.65
MARKER_ALPHA = 0.85
PROFILE_STYLES = {
    "seeded": {"label": "ibUMAP (seeded)", "color": "#0072B2", "marker": "o", "filled": True},
    "unseeded": {"label": "ibUMAP (unseeded)", "color": "#D55E00", "marker": "^", "filled": False},
}
LINE_WIDTH_PT = 1.0
Y_LIMITS = (0.04, 15)
Y_MAJOR_TICKS = (0.1, 1, 10)
LEGEND_KW = dict(frameon=False, fontsize=LEGEND_SIZE_PT, markerscale=1.0, labelspacing=0.35,
                 handletextpad=0.4, borderpad=0.2)
AXES_RECT = (0.19, 0.18, 0.775, 0.775)
RUNTIME_Y_LIMITS = (0.02, 5_000)
RUNTIME_Y_MAJOR_TICKS = (0.1, 1, 10, 100, 1_000)
RUNTIME_ALGORITHM_STYLES = {"baseline": {"color": "#D55E00", "marker": "s"},
                            "ibumap": {"color": "#0072B2", "marker": "o"}}
RUNTIME_PROFILE_STYLES = {"seeded": {"linestyle": "-", "filled": True},
                          "unseeded": {"linestyle": "--", "filled": False}}
COMPARISONS = {
    "cpu": {"baseline_id": "umap_learn", "candidate_id": "ibumap_cpu", "baseline_name": "umap-learn",
            "candidate_name": "ibUMAP CPU", "y_label": "Speedup over umap-learn"},
    "gpu": {"baseline_id": "cuml_umap", "candidate_id": "ibumap_cuda", "baseline_name": "cuML",
            "candidate_name": "ibUMAP CUDA", "y_label": "Speedup over cuML"},
}
RC = {"lines.linewidth": LINE_WIDTH_PT, "savefig.bbox": None}


def fit_limits(rows, paths):
    """Keep the paper's axis limits; widen them only for new results that fall outside."""
    global Y_LIMITS, RUNTIME_Y_LIMITS
    style.X_LIMITS = style.fit_log_limits(paths, rows, "n_samples", style.X_LIMITS, "X_LIMITS")
    Y_LIMITS = style.fit_log_limits(paths, rows, "speedup", Y_LIMITS, "Y_LIMITS")
    RUNTIME_Y_LIMITS = style.fit_log_limits(paths, rows, ("baseline_median_e2e_s", "ibumap_median_e2e_s"),
                                            RUNTIME_Y_LIMITS, "RUNTIME_Y_LIMITS")


def log_axes(ax, y_limits, y_ticks, y_format):
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*style.X_LIMITS)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(FixedLocator(style.SAMPLE_TICKS))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(y_format))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        axis.set_minor_formatter(NullFormatter())
    ax.set_axisbelow(True)
    ax.grid(which="major", color=style.GRID_COLOR, alpha=0.65, linewidth=0.4)


def finish_axes(ax, ylabel):
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(ylabel)
    ax.tick_params(which="both", width=0.6, pad=2.0, direction="out")
    ax.tick_params(which="major", length=3.0)
    ax.tick_params(which="minor", length=1.7)
    ax.spines[["top", "right"]].set_visible(False)


def draw_speedup_panel(ax, rows, device):
    log_axes(ax, Y_LIMITS, Y_MAJOR_TICKS, lambda value, _: f"{value:g}×")
    ax.axhline(1, color=style.REFERENCE_COLOR, linestyle="--", linewidth=0.85, zorder=2)
    for profile, st in PROFILE_STYLES.items():
        part = sorted((r for r in rows if r["device"] == device and r["execution_profile"] == profile),
                      key=lambda r: (r["n_samples"], r["dataset"]))
        ax.scatter([r["n_samples"] for r in part], [r["speedup"] for r in part], s=MARKER_SIZE_PT2,
                   marker=st["marker"], facecolors=st["color"] if st["filled"] else "none",
                   edgecolors=st["color"], linewidths=MARKER_EDGE_WIDTH_PT, alpha=MARKER_ALPHA,
                   label=st["label"], zorder=3)
    finish_axes(ax, COMPARISONS[device]["y_label"])
    ax.legend(loc="lower right", handlelength=1.2, **LEGEND_KW)


def draw_runtime_panel(ax, rows, device):
    log_axes(ax, RUNTIME_Y_LIMITS, RUNTIME_Y_MAJOR_TICKS, lambda value, _: f"{value:g}")
    for role, algorithm_style in RUNTIME_ALGORITHM_STYLES.items():
        name = COMPARISONS[device]["baseline_name"] if role == "baseline" else "ibUMAP"
        column = "baseline_median_e2e_s" if role == "baseline" else "ibumap_median_e2e_s"
        for profile in ("seeded", "unseeded"):
            profile_style = RUNTIME_PROFILE_STYLES[profile]
            part = sorted((r for r in rows if r["device"] == device and r["execution_profile"] == profile),
                          key=lambda r: (r["n_samples"], r["dataset"]))
            ax.plot([r["n_samples"] for r in part], [r[column] for r in part],
                    color=algorithm_style["color"], marker=algorithm_style["marker"],
                    linestyle=profile_style["linestyle"], linewidth=LINE_WIDTH_PT, markersize=2.6,
                    markeredgewidth=0.5,
                    markerfacecolor=algorithm_style["color"] if profile_style["filled"] else style.BACKGROUND_COLOR,
                    markeredgecolor=algorithm_style["color"], alpha=0.9, label=f"{name} ({profile})", zorder=3)
    finish_axes(ax, "End-to-end time (s)")
    ax.legend(loc="upper left", handlelength=2.0, **LEGEND_KW)


def side_by_side(rows, draw_panel, stem, formats):
    import matplotlib.pyplot as plt

    full_width = 2 * PANEL_WIDTH_IN + PANEL_GAP_IN
    fig = plt.figure(figsize=(full_width, PANEL_HEIGHT_IN))
    left, bottom, width, height = AXES_RECT
    for index, device in enumerate(COMPARISONS):
        offset = index * (PANEL_WIDTH_IN + PANEL_GAP_IN)
        rect = ((offset + left * PANEL_WIDTH_IN) / full_width, bottom, width * PANEL_WIDTH_IN / full_width, height)
        draw_panel(fig.add_axes(rect), rows, device)
    # Physical panel dimensions stay fixed (no bbox_inches="tight").
    style.save_figure(fig, stem, formats)


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    data = _e2e.load(paths)
    rows = _e2e.profile_rows(data, COMPARISONS)
    fit_limits(rows, paths)
    (paths.figures / "data").mkdir(exist_ok=True)
    write_csv(paths.figures / "data" / "figure_A_data.csv", rows)
    style.configure_matplotlib(rc_overrides=RC, hashsalt="ibumap-paper-figure-a")
    side_by_side(rows, draw_speedup_panel, paths.figures / "figure_A_side_by_side", args.formats)
    side_by_side(rows, draw_runtime_panel, paths.figures / "figure_A_runtime_seeded_unseeded_side_by_side",
                 args.formats)
    print(f"ch5_profile_figures.py: wrote 2 figures to {paths.figures}")


if __name__ == "__main__":
    main()
