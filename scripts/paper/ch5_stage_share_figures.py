#!/usr/bin/env python3
"""Appendix figure fig:appendix-stage-share: shares of ibUMAP end-to-end time by stage (2x2).

Writes paper/build/figures/figure_F_ibumap_stage_share_{cpu,cuda}_{unseeded,seeded}.pdf
and paper/build/figures/data/{stage_share_data.csv, stage_share_summary.json}.

Stage decomposition (mutually exclusive, per run):
    graph construction  = min(build_graph_time, fit_preprocess_time)
    other preprocessing = fit_preprocess_time - graph construction
    initialization      = embd_init_time
    optimization        = embd_opt_time
    unattributed        = max(0, e2e_call_wall_s - fit_preprocess - init - opt)
Per dataset: five-run median of every component divided by the sum of the medians.
Datasets with identical n_samples (MNIST, Fashion-MNIST) are averaged for the curve.
Only ibUMAP records stage timings, so the figure is not a cross-implementation comparison.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import median

import _e2e
import _style as style
from _common import expect_paper, parse_args, write_csv, write_json

PANEL_WIDTH_IN, PANEL_HEIGHT_IN = 2.70, 2.05
AXES_LEFT_IN, AXES_RIGHT_IN = 0.40, 0.10
AXES_BOTTOM_IN, AXES_TOP_IN = 0.36, 0.58
LEGEND_FONT_PT = 6.8
VARIANTS = (
    ("cpu_unseeded", "ibumap_cpu_seed_none", "(a) ibUMAP CPU, unseeded", 66),
    ("cpu_seeded", "ibumap_cpu_seed_42", "(b) ibUMAP CPU, seeded", 66),
    ("cuda_unseeded", "ibumap_cuda_seed_none", "(c) ibUMAP CUDA, unseeded", 71),
    ("cuda_seeded", "ibumap_cuda_seed_42", "(d) ibUMAP CUDA, seeded", 71),
)
STEM_PREFIX = "figure_F_ibumap_stage_share"
STAGES = (  # Okabe-Ito hues in pipeline order; the residual is neutral gray
    ("graph", "Graph construction", "#0072B2"),
    ("preprocess_other", "Other preprocessing", "#56B4E9"),
    ("initialization", "Initialization", "#E69F00"),
    ("optimization", "Optimization", "#009E73"),
    ("unattributed", "Unattributed", "#BDBDBD"),
)
REQUIRED_FIELDS = ("time_build_graph_time", "time_fit_preprocess_time", "time_embd_init_time",
                   "time_embd_opt_time", "e2e_call_wall_s")


def finite_nonnegative(row, field):
    value = float(row[field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {field} for {row['dataset']}/{row['run_id']}: {value}")
    return value


def run_components(row):
    graph_raw = finite_nonnegative(row, "time_build_graph_time")
    preprocess = finite_nonnegative(row, "time_fit_preprocess_time")
    init = finite_nonnegative(row, "time_embd_init_time")
    opt = finite_nonnegative(row, "time_embd_opt_time")
    wall = finite_nonnegative(row, "e2e_call_wall_s")
    graph = min(graph_raw, preprocess)
    return {"graph": graph, "preprocess_other": preprocess - graph, "initialization": init,
            "optimization": opt, "unattributed": max(0.0, wall - preprocess - init - opt),
            "_wall": wall, "_overlap": max(0.0, preprocess + init + opt - wall)}


def build_dataset_rows(runs, paths):
    groups = defaultdict(list)
    for row in runs:
        if row["algorithm_id"].startswith("ibumap"):
            groups[row["algorithm_id"], row["dataset"]].append(row)
    output = []
    for key, algorithm, _, expected in VARIANTS:
        datasets = sorted(d for a, d in groups if a == algorithm)
        expect_paper(paths, len(datasets) == expected, f"{algorithm}: {len(datasets)} datasets, paper {expected}")
        for dataset in datasets:
            part = groups[algorithm, dataset]
            if len(part) != 5 or {int(r["repeat"]) for r in part} != set(range(1, 6)):
                raise ValueError(f"Incomplete repetitions: {dataset}/{algorithm}")
            if any(r["status"] != "ok" for r in part):
                raise ValueError(f"Failed run: {dataset}/{algorithm}")
            for field in REQUIRED_FIELDS:
                if any(r.get(field) in (None, "") for r in part):
                    raise ValueError(f"Missing {field}: {dataset}/{algorithm}")
            sizes = {int(r["n_samples"]) for r in part}
            if len(sizes) != 1:
                raise ValueError(f"Inconsistent n_samples: {dataset}/{algorithm}")
            comps = [run_components(r) for r in part]
            med = {name: median(c[name] for c in comps) for name, _, _ in STAGES}
            total = sum(med.values())
            if total <= 0:
                raise ValueError(f"Non-positive stage total: {dataset}/{algorithm}")
            row = {"variant": key, "algorithm_id": algorithm, "dataset": dataset, "n_samples": sizes.pop(),
                   "median_e2e_call_wall_s": median(c["_wall"] for c in comps), "stage_total_s": total,
                   "max_timer_overlap_fraction": max(c["_overlap"] / c["_wall"] for c in comps)}
            for name, _, _ in STAGES:
                row[f"{name}_median_s"] = med[name]
                row[f"{name}_share"] = med[name] / total
            output.append(row)
    return output


def curve(rows):
    by_n = defaultdict(list)
    for row in rows:
        by_n[row["n_samples"]].append(row)
    xs = sorted(by_n)
    shares = {name: [sum(r[f"{name}_share"] for r in by_n[n]) / len(by_n[n]) * 100.0 for n in xs]
              for name, _, _ in STAGES}
    return [float(x) for x in xs], shares


def draw_panel(rows, *, title, x_limits, stem, formats):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter

    fig = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    ax = fig.add_axes((AXES_LEFT_IN / PANEL_WIDTH_IN, AXES_BOTTOM_IN / PANEL_HEIGHT_IN,
                       (PANEL_WIDTH_IN - AXES_LEFT_IN - AXES_RIGHT_IN) / PANEL_WIDTH_IN,
                       (PANEL_HEIGHT_IN - AXES_BOTTOM_IN - AXES_TOP_IN) / PANEL_HEIGHT_IN))
    xs, shares = curve(rows)
    ax.stackplot(xs, *[shares[name] for name, _, _ in STAGES], colors=[c for _, _, c in STAGES],
                 edgecolor="white", linewidth=0.35, zorder=2)
    # Rug marks show where datasets exist; the area between them is interpolated.
    ax.plot(xs, [101.6] * len(xs), linestyle="none", marker="|", markersize=2.6, markeredgewidth=0.45,
            color=style.REFERENCE_COLOR, clip_on=False, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator(style.SAMPLE_TICKS))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator((0, 25, 50, 75, 100)))
    ax.set_xlabel("Number of samples", labelpad=1.5)
    ax.set_ylabel("Share of time (%)", labelpad=2.0)
    ax.tick_params(which="both", width=0.6, pad=2.0, direction="out")
    ax.tick_params(which="major", length=3.0)
    ax.tick_params(which="minor", length=1.7)
    ax.grid(axis="y", color="white", alpha=0.55, linewidth=0.4, zorder=2.5)
    ax.set_axisbelow(False)
    ax.spines[["top", "right"]].set_visible(False)
    top = 1.0 - 0.06 / PANEL_HEIGHT_IN
    fig.text(0.02, top, f"{title} ($n={len(rows)}$)", ha="left", va="top", fontsize=8.5)
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in STAGES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, top - 0.090), ncol=3, frameon=False,
               fontsize=LEGEND_FONT_PT, handlelength=1.0, handleheight=0.8, handletextpad=0.35,
               columnspacing=0.75, labelspacing=0.25, borderaxespad=0.0)
    style.save_figure(fig, stem, formats)


def summarize(rows):
    bins = (("all", 0, math.inf), ("lt_1e4", 0, 1e4), ("1e4_1e5", 1e4, 1e5), ("ge_1e5", 1e5, math.inf),
            ("ge_1e6", 1e6, math.inf))
    result = {}
    for key, *_ in VARIANTS:
        local = [r for r in rows if r["variant"] == key]
        result[key] = {}
        for name, lo, hi in bins:
            part = [r for r in local if lo <= r["n_samples"] < hi]
            if part:
                result[key][name] = {"n": len(part), **{f"median_{s}_share": median(r[f"{s}_share"] for r in part)
                                                        for s, _, _ in STAGES}}
        result[key]["max_timer_overlap_fraction"] = max(r["max_timer_overlap_fraction"] for r in local)
    return result


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    data = _e2e.load(paths)
    rows = build_dataset_rows(data.runs, paths)
    all_n = [r["n_samples"] for r in rows]
    x_limits = (min(all_n), max(all_n))
    (paths.figures / "data").mkdir(exist_ok=True)
    write_csv(paths.figures / "data" / "stage_share_data.csv", rows)
    write_json(paths.figures / "data" / "stage_share_summary.json", summarize(rows))
    style.configure_matplotlib()
    for key, _, title, _ in VARIANTS:
        draw_panel([r for r in rows if r["variant"] == key], title=title, x_limits=x_limits,
                   stem=paths.figures / f"{STEM_PREFIX}_{key}", formats=args.formats)
    print(f"ch5_stage_share_figures.py: wrote 4 figures to {paths.figures}")


if __name__ == "__main__":
    main()
