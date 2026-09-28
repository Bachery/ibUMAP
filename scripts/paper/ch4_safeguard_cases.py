#!/usr/bin/env python3
"""Appendix B, safeguard failure cases: tab:safeguard-cases and fig:safeguard-cases.

Production ibUMAP (all safeguards) and the same optimizer without the repulsion-norm
clip or without the attraction degree damping, on the fixed inputs of CIFAR-10 and
scDEED CART from the mechanism study. A layout is summarized by the ratio of its
bounding-box area to that of the 99% of points nearest the coordinate-wise median;
a point is far when its distance to the median exceeds twice the core radius.

Input: paper/data/safeguards/ (protocol.json, runs, cases, traces, escaped_points,
reference, embeddings.npz). The script recomputes the final statistics from the
plotted embeddings and the peaks from the per-update traces and checks them against
the frozen summaries.

Writes
  paper/build/tables/ch4_safeguard_rows.tex       table rows (\\input by the appendix)
  paper/build/tables/ch4_safeguard_summary.json   statistics quoted in the appendix text
  paper/build/figures/figure_ch4_safeguard_cases.pdf
  paper/build/figures/data/ch4_safeguard_traces.csv   plotted ratio after every update
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

import _style as style
from _common import check_group, expect_paper, parse_args, read_csv, read_json, write_csv, write_json, write_rows

DATASET_LABELS = {"cifar10": "CIFAR-10", "scdeed_cart": "scDEED CART"}
VARIANT_TEX = {"full": "All safeguards", "no_repulsion_clip": "No repulsion-norm clip",
               "no_attraction_damping": "No attraction damping"}
COLUMN_TITLES = {"full": "ibUMAP (all safeguards)", "no_repulsion_clip": "no repulsion-norm clip",
                 "no_attraction_damping": "no attraction damping"}
TRACE_STYLE = {"full": dict(color="#444444", lw=1.1),
               "no_repulsion_clip": dict(color=style.ORANGE, lw=1.1),
               "no_attraction_damping": dict(color=style.BLUE, lw=1.1)}
# Font sizes of this figure (smaller than the shared defaults: six panels share one row width).
RC = {"axes.labelsize": 7.0, "axes.labelpad": 4.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0}


def core_mask(y: np.ndarray, core_fraction: float):
    """The ceil(core_fraction * n) points nearest the coordinate-wise median (ties by index)."""
    radius = np.linalg.norm(y - np.median(y, axis=0), axis=1)
    n_core = int(np.ceil(core_fraction * len(y)))
    order = np.argsort(radius, kind="stable")
    mask = np.zeros(len(y), dtype=bool)
    mask[order[:n_core]] = True
    return mask, radius, float(radius[order[n_core - 1]])


def final_statistics(y: np.ndarray, core_fraction: float, far_multiple: float) -> dict:
    mask, radius, core_radius = core_mask(y, core_fraction)
    extent = y.max(axis=0) - y.min(axis=0)
    core = y[mask]
    core_extent = core.max(axis=0) - core.min(axis=0)
    far = radius > far_multiple * core_radius
    return {"mask": mask, "far": far, "core_lo": core.min(axis=0), "core_hi": core.max(axis=0),
            "bbox_area_ratio": float(extent[0] * extent[1] / (core_extent[0] * core_extent[1])),
            "far_count": int(far.sum()), "far_pct": 100.0 * float(far.mean())}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def ratio_text(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}".replace(",", "{,}")
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def load(paths):
    check_group(paths, "safeguards")
    root = paths.data / "safeguards"
    protocol = read_json(root / "protocol.json")
    datasets, variants = protocol["datasets"], list(protocol["variants"])
    seeds, seed = [str(s) for s in protocol["seeds"]], str(protocol["plot_seed"])
    core_fraction = float(protocol["metrics"]["core_fraction"])
    far_multiple = float(protocol["metrics"]["far_radius_multiple"])
    n_epochs = int(protocol["params"]["n_epochs"])
    runs = read_csv(root / "runs.csv.gz")
    cases = {(r["dataset"], r["variant"]): r for r in read_csv(root / "cases.csv.gz")}
    traces = read_csv(root / "traces.csv.gz")
    escaped = read_csv(root / "escaped_points.csv.gz")
    reference = read_csv(root / "reference.csv.gz")
    out = {}
    with np.load(root / "embeddings.npz", allow_pickle=False) as npz:
        for dataset in datasets:
            for variant in variants:
                key = (dataset, variant)
                if key not in cases:
                    raise SystemExit(f"cases.csv.gz lacks {dataset}/{variant}")
                by_seed = {r["seed"]: r for r in runs if (r["dataset"], r["variant"]) == key}
                if sorted(by_seed) != sorted(seeds):
                    raise SystemExit(f"{dataset}/{variant}: runs for seeds {sorted(by_seed)}, expected {seeds}")
                y32 = npz[f"{dataset}__{variant}"]
                if hashlib.sha256(np.ascontiguousarray(y32).tobytes()).hexdigest() != by_seed[seed]["embedding_sha256"]:
                    raise SystemExit(f"{dataset}/{variant}: embeddings.npz does not match runs.csv.gz")
                y = y32.astype(np.float64)
                stats = final_statistics(y, core_fraction, far_multiple)
                for name in ("bbox_area_ratio", "far_count"):
                    if not close(stats[name], float(by_seed[seed][name])):
                        raise SystemExit(f"{dataset}/{variant}: recomputed {name} {stats[name]} differs from runs.csv.gz")
                rows = sorted((r for r in traces if (r["dataset"], r["variant"]) == key and r["seed"] == seed),
                              key=lambda r: int(r["epoch"]))
                if [int(r["epoch"]) for r in rows] != list(range(n_epochs)):
                    raise SystemExit(f"{dataset}/{variant}: trace does not cover epochs 0..{n_epochs - 1}")
                trace = {c: np.array([float(r[c]) for r in rows]) for c in
                         ("after_bbox_area_ratio", "step_norm_max", "attr_norm_max", "before_core_boxes")}
                if not close(trace["after_bbox_area_ratio"][-1], stats["bbox_area_ratio"]):
                    raise SystemExit(f"{dataset}/{variant}: last trace ratio differs from the final embedding")
                far_rows = [r for r in escaped if (r["dataset"], r["variant"]) == key and r["seed"] == seed]
                if len(far_rows) != stats["far_count"]:
                    raise SystemExit(f"{dataset}/{variant}: {len(far_rows)} escaped points, expected {stats['far_count']}")
                distinct = {r["embedding_sha256"] for r in by_seed.values()}
                out[key] = {"y": y, "stats": stats, "trace": trace, "case": cases[key], "far_rows": far_rows,
                            "seeds_identical": len(distinct) == 1}
    return protocol, out, reference


def summarize(protocol, data, reference, paths) -> dict:
    cases = {}
    for (dataset, variant), d in data.items():
        t, case = d["trace"], d["case"]
        ratio = t["after_bbox_area_ratio"]
        peak = int(np.argmax(ratio))
        entry = {
            "final_ratio": d["stats"]["bbox_area_ratio"], "far_count": d["stats"]["far_count"],
            "far_pct": d["stats"]["far_pct"], "peak_ratio": float(ratio[peak]), "peak_update": peak + 1,
            "updates_ratio_gt_10": int((ratio > 10).sum()),
            "max_step_first_update": float(t["step_norm_max"][0]),
            "max_attraction_norm": float(t["attr_norm_max"].max()),
            "core_boxes_median": float(np.median(t["before_core_boxes"])),
            "weighted_degree_p99": float(case["weighted_degree_p99"]),
            "seeds_bitwise_identical": d["seeds_identical"],
        }
        for key, column in (("peak_ratio", "trace_max_bbox_area_ratio_over_epochs"),
                            ("updates_ratio_gt_10", "trace_updates_ratio_gt_10"),
                            ("max_attraction_norm", "trace_max_attr_norm"),
                            ("core_boxes_median", "trace_core_boxes_median")):
            if not close(float(entry[key]), float(case[column])):
                raise SystemExit(f"{dataset}/{variant}: {key} from the trace differs from cases.csv.gz")
        if d["far_rows"]:
            pct = np.array([float(r["weighted_degree_percentile"]) for r in d["far_rows"]])
            degree = np.array([float(r["weighted_degree"]) for r in d["far_rows"]])
            entry.update(far_degree_percentile_median=100.0 * float(np.median(pct)),
                         far_share_above_p99=float(np.mean(degree > entry["weighted_degree_p99"])),
                         far_max_weighted_degree=float(degree.max()))
        cases.setdefault(dataset, {})[variant] = entry

    ref = {}
    for algorithm in protocol["reference"]["algorithms"]:
        rows = [r for r in reference if r["algorithm"] == algorithm]
        per_dataset = {}
        for r in rows:
            per_dataset[r["dataset"]] = max(per_dataset.get(r["dataset"], 0.0), float(r["bbox_area_ratio"]))
        values = np.array(list(per_dataset.values()))
        worst = max(per_dataset, key=per_dataset.get)
        ref[algorithm] = {"n_datasets": len(per_dataset), "n_runs": len(rows), "median": float(np.median(values)),
                          "max": float(values.max()), "max_dataset": worst,
                          "datasets_with_far_points": len({r["dataset"] for r in rows if int(r["far_count"]) > 0})}
    summary = {"cases": cases, "reference": ref,
               "definitions": protocol["metrics"], "plot_seed": protocol["plot_seed"]}

    for dataset in protocol["datasets"]:
        c = cases[dataset]
        expect_paper(paths, all(v["seeds_bitwise_identical"] for v in c.values()),
                     f"{dataset}: the optimizer seeds give different embeddings")
        expect_paper(paths, c["full"]["far_count"] == 0, f"{dataset}: far points with all safeguards")
        expect_paper(paths, c["no_repulsion_clip"]["far_count"] > 0 and c["no_repulsion_clip"]["final_ratio"] > 10,
                     f"{dataset}: no persistent escape without the repulsion-norm clip")
        expect_paper(paths, c["no_attraction_damping"]["far_count"] <= 1
                     and c["no_attraction_damping"]["updates_ratio_gt_10"] > len(data[(dataset, "full")]["trace"]
                                                                                ["after_bbox_area_ratio"]) // 2,
                     f"{dataset}: without damping the escape is not transient (ratio > 10 after most updates, "
                     f"at most one far point at the end)")
    for algorithm, r in ref.items():
        expect_paper(paths, r["n_datasets"] == 59 and r["datasets_with_far_points"] == 0,
                     f"reference {algorithm}: {r['n_datasets']} datasets, {r['datasets_with_far_points']} with far points")
    return summary


def table_rows(protocol, summary) -> list[str]:
    lines = []
    for d_index, dataset in enumerate(protocol["datasets"]):
        if d_index:
            lines.append(r"\midrule")
        for v_index, variant in enumerate(protocol["variants"]):
            c = summary["cases"][dataset][variant]
            name = DATASET_LABELS.get(dataset, dataset) if v_index == 0 else ""
            far = c["far_count"]
            far_text = "0" if far == 0 else (f"{far} ({c['far_pct']:.1f}\\%)" if c["far_pct"] >= 0.05
                                             else f"{far} ($<$0.1\\%)")
            lines.append(f"{name} & {VARIANT_TEX.get(variant, variant)} & {ratio_text(c['final_ratio'])} & "
                         f"{far_text} & {ratio_text(c['peak_ratio'])} ({c['peak_update']}) & "
                         f"{c['updates_ratio_gt_10']} \\\\")
    return lines


def square_limits(y: np.ndarray, pad: float = 0.05):
    lo, hi = y.min(axis=0), y.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo)) * (1.0 + 2.0 * pad)
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def draw(protocol, data, paths, formats) -> None:
    style.configure_matplotlib(rc_overrides=RC)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

    datasets, variants = protocol["datasets"], list(protocol["variants"])
    far_multiple = float(protocol["metrics"]["far_radius_multiple"])
    core_percent = f"{100 * float(protocol['metrics']['core_fraction']):g}"
    dash = (0, (2.5, 1.5))
    fig = plt.figure(figsize=(style.PAPER_TEXT_WIDTH_IN, 4.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.0], hspace=0.38,
                             left=0.085, right=0.985, top=0.955, bottom=0.11)

    # (a) final embeddings at equal aspect, each panel spanning its full extent
    grid = outer[0].subgridspec(len(datasets), len(variants), wspace=0.42, hspace=0.30)
    scatter_axes = []
    for r, dataset in enumerate(datasets):
        for c, variant in enumerate(variants):
            ax = fig.add_subplot(grid[r, c])
            scatter_axes.append(ax)
            d = data[(dataset, variant)]
            y, far = d["y"], d["stats"]["far"]
            ax.scatter(y[~far, 0], y[~far, 1], s=0.08, c=style.BLUE, linewidths=0, alpha=0.5,
                       rasterized=True, zorder=2)
            if far.any():
                ax.scatter(y[far, 0], y[far, 1], s=1.8, c=style.ORANGE, linewidths=0, rasterized=True, zorder=3)
            lo, hi = d["stats"]["core_lo"], d["stats"]["core_hi"]
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, lw=0.7, ls=dash, ec=style.TEXT_COLOR, zorder=4))
            xlim, ylim = square_limits(y)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(width=0.5, length=2.0, pad=1.5)
            ax.text(0.03, 0.97, f"ratio {ratio_text(d['stats']['bbox_area_ratio']).replace('{,}', ',')}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=5)
            if r == 0:
                ax.set_title(COLUMN_TITLES.get(variant, variant), fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(DATASET_LABELS.get(dataset, dataset), fontsize=7.5, labelpad=2)

    # (b) bounding-box ratio after every update
    grid = outer[1].subgridspec(1, len(datasets), wspace=0.10)
    ymax = max(float(d["trace"]["after_bbox_area_ratio"].max()) for d in data.values())
    trace_axes, plotted = [], []
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(grid[0, i], sharey=trace_axes[0] if trace_axes else None)
        trace_axes.append(ax)
        for variant in variants:
            ratio = data[(dataset, variant)]["trace"]["after_bbox_area_ratio"]
            updates = np.arange(1, len(ratio) + 1)
            ax.plot(updates, ratio, solid_joinstyle="round", solid_capstyle="round", zorder=3,
                    label=COLUMN_TITLES.get(variant, variant), **TRACE_STYLE[variant])
            plotted += [{"dataset": dataset, "variant": variant, "update": int(u), "bbox_area_ratio": repr(float(v))}
                        for u, v in zip(updates, ratio)]
        ax.set_yscale("log")
        ax.set_ylim(0.8, ymax * 2.0)
        ax.set_xlim(0, 202)
        ax.xaxis.set_major_locator(FixedLocator([0, 50, 100, 150, 200]))
        ax.yaxis.set_major_locator(FixedLocator(style.extend_decade_ticks((1, 10, 100, 1000), (0.8, ymax * 2.0))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", axis="y", color=style.GRID_COLOR, lw=0.4, zorder=0)
        ax.tick_params(which="both", width=0.5, pad=1.5)
        ax.tick_params(which="major", length=2.0)
        ax.tick_params(which="minor", length=1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.02, 0.97, DATASET_LABELS.get(dataset, dataset), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5)
        ax.set_xlabel("completed updates", labelpad=1.5)
        if i:
            ax.tick_params(labelleft=False)
    trace_axes[0].set_ylabel("bbox area ratio", labelpad=2)

    scatter_handles = [
        Line2D([], [], ls="none", marker="o", ms=2.5, mfc=style.BLUE, mec="none", label="points"),
        Line2D([], [], ls="none", marker="o", ms=3.0, mfc=style.ORANGE, mec="none",
               label=rf"far points ($r > {far_multiple:g}\,r_{{99}}$)"),
        Line2D([], [], color=style.TEXT_COLOR, lw=0.7, ls=dash, label=f"bounding box of {core_percent}% core"),
    ]
    bottom = min(ax.get_position().y0 for ax in scatter_axes)
    fig.legend(handles=scatter_handles, loc="upper center", ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.535, bottom - 0.018), handletextpad=0.4, columnspacing=1.6)
    trace_handles = [Line2D([], [], label=COLUMN_TITLES.get(v, v), **TRACE_STYLE[v]) for v in variants]
    fig.legend(handles=trace_handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.535, -0.008), handlelength=1.8, columnspacing=1.6)
    for label, ax, offset in (("(a)", scatter_axes[0], 0.012), ("(b)", trace_axes[0], 0.004)):
        fig.text(0.008, ax.get_position().y1 + offset, label, fontsize=8, fontweight="bold", ha="left", va="bottom")

    style.save_figure(fig, paths.figures / "figure_ch4_safeguard_cases", formats, dpi=400)
    (paths.figures / "data").mkdir(parents=True, exist_ok=True)
    write_csv(paths.figures / "data" / "ch4_safeguard_traces.csv", plotted)


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    protocol, data, reference = load(paths)
    summary = summarize(protocol, data, reference, paths)
    write_rows(paths, "ch4_safeguard_rows.tex", table_rows(protocol, summary), "ch4_safeguard_cases.py")
    write_json(paths.tables / "ch4_safeguard_summary.json", summary)
    draw(protocol, data, paths, args.formats)
    print(f"ch4_safeguard_cases.py: {len(data)} cases, reference "
          + ", ".join(f"{a}={r['n_datasets']} datasets" for a, r in summary["reference"].items()))


if __name__ == "__main__":
    main()
