#!/usr/bin/env python3
"""Section 5 main figure (fig:e2e-results): speedups and run-to-run 15-NN overlap in one 1x3 row.

Panels: (a) CPU speedup over umap-learn, seeded and unseeded; (b) unseeded GPU
speedup over cuML and TorchDR; (c) paired 15-NN overlap differences across
unseeded reruns. Writes paper/build/figures/figure_BE_runtime_stability_merged_h166.pdf
and the plotted rows to paper/build/figures/data/{runtime_data,stability_data}.csv.
"""
from __future__ import annotations

from collections import Counter
from statistics import median

import _e2e
import _style as style
from _common import parse_args, write_csv, write_json

STEM = "figure_BE_runtime_stability_merged_h166"
FIG_H = 1.66
STABILITY_METRIC = "neighbor_overlap_at_15"
EXPECTED_COUNTS = {"cpu_umap": 66, "gpu_cuml": 71, "gpu_torchdr": 66}
RUNTIME_SERIES = (("cpu_umap", "unseeded"), ("cpu_umap", "seeded"),
                  ("gpu_cuml", "unseeded"), ("gpu_torchdr", "unseeded"))
SHORT = {"cpu_umap": "umap-learn", "gpu_cuml": "cuML", "gpu_torchdr": "TorchDR"}

FIG_W = 5.5
M_LEFT, M_RIGHT, M_BOT, M_TOP = 0.42, 0.04, 0.36, 0.17   # inches
XLABEL_Y = -0.235                                        # axes coords, shared
YTICK_ROTATION = 45                                      # panel (c) row labels
GAP_AB, GAP_BC = 0.05, 0.44                              # inches
W_SPEED, W_STAB = 1.45, 1.71                             # inches


def build_rows(data):
    runtime, _ = _e2e.runtime_rows(data)
    stable = _e2e.stability_delta_rows(data)
    if dict(Counter(r["comparison"] for r in runtime)) != EXPECTED_COUNTS:
        raise ValueError("Unexpected runtime coverage.")
    for metric, _ in style.STABILITY_METRICS:
        if dict(Counter(r["comparison"] for r in stable if r["metric"] == metric)) != EXPECTED_COUNTS:
            raise ValueError(f"Unexpected stability coverage for {metric}.")
    seeded = _e2e.cpu_seeded_rows(data, {r["dataset"] for r in runtime if r["comparison"] == "cpu_umap"})
    runtime = [dict(r, execution_profile="unseeded") for r in runtime] + seeded
    counts = Counter((r["comparison"], r["execution_profile"]) for r in runtime)
    if dict(counts) != {(k, p): EXPECTED_COUNTS[k] for k, p in RUNTIME_SERIES}:
        raise ValueError("Unexpected execution-profile coverage in runtime figure.")
    return runtime, stable


def axes_rects(fig_h):
    """Explicit rectangles: (a),(b) sit close together, (c) needs a label gutter."""
    x = M_LEFT
    rects = []
    for width in (W_SPEED, W_SPEED, W_STAB):
        rects.append([x / FIG_W, M_BOT / fig_h, width / FIG_W, (fig_h - M_BOT - M_TOP) / fig_h])
        x += width + (GAP_AB if len(rects) == 1 else GAP_BC)
    return rects


def draw(runtime, stable, stem, formats, fig_h=FIG_H):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    style.validate_in_bounds(runtime, "n_samples", style.X_LIMITS)
    style.validate_in_bounds(runtime, "speedup", style.SPEEDUP_LIMITS)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax_cpu, ax_gpu, ax_stab = (fig.add_axes(r) for r in axes_rects(fig_h))

    # (a) and (b): shared log-log speedup axes
    for ax in (ax_cpu, ax_gpu):
        style.style_log_axis(ax, ylabel="")
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(FixedLocator((100, 10_000, 1_000_000)))
    ax_cpu.set_ylabel("Speedup")
    ax_gpu.set_yticklabels([])
    ax_cpu.set_xlabel("Number of samples")
    ax_cpu.xaxis.set_label_coords((W_SPEED + GAP_AB / 2) / W_SPEED, XLABEL_Y)
    for key, profile in RUNTIME_SERIES:
        comparison = style.comparison_for(key)
        part = [r for r in runtime if r["comparison"] == key and r["execution_profile"] == profile]
        ax = ax_cpu if key == "cpu_umap" else ax_gpu
        triangle, seeded = key == "gpu_torchdr", profile == "seeded"
        value = median(r["speedup"] for r in part)
        name = profile.capitalize() if key == "cpu_umap" else SHORT[key]
        ax.scatter([r["n_samples"] for r in part], [r["speedup"] for r in part],
                   s=13 if triangle or seeded else 10, marker="^" if triangle else "o",
                   facecolors="none" if seeded else comparison["color"], edgecolors=comparison["color"],
                   linewidths=0.6 if seeded else 0.25, alpha=0.95 if seeded else 0.8,
                   zorder=4 if seeded or triangle else 3, label=f"{name} {value:.3g}$\\times$")
    for ax in (ax_cpu, ax_gpu):
        ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", fontsize=6.3,
                  handletextpad=0.3, handlelength=0.9, labelspacing=0.18, borderpad=0.22, borderaxespad=0.25)

    # (c) paired 15-NN overlap differences; asymmetric limits (nearly all differences are positive)
    part = [r for r in stable if r["metric"] == STABILITY_METRIC]
    lo = min(r["paired_delta"] for r in part)
    hi = max(r["paired_delta"] for r in part)
    pad = 0.07 * (hi - lo)
    ax_stab.set_xlim(lo - pad, hi + pad)
    ax_stab.set_ylim(-0.45, 2.45)
    ax_stab.axvline(0, color=style.REFERENCE_COLOR, linestyle="--", linewidth=0.85, zorder=2)
    positions = (2, 1, 0)
    for position, comparison in zip(positions, style.COMPARISONS):
        local = [r for r in part if r["comparison"] == comparison["key"]]
        ax_stab.scatter([r["paired_delta"] for r in local],
                        [position + style.deterministic_jitter(r["dataset"], 0.13) for r in local],
                        s=8, color=comparison["color"], edgecolors="none", alpha=0.65, zorder=3)
        ax_stab.scatter([median(r["paired_delta"] for r in local)], [position], marker="D", s=30,
                        color=comparison["color"], edgecolors="white", linewidths=0.7, zorder=4)
    ax_stab.set_yticks(positions, [SHORT[c["key"]] for c in style.COMPARISONS],
                       rotation=YTICK_ROTATION, rotation_mode="anchor", ha="right", va="center")
    ax_stab.tick_params(axis="y", length=0, pad=1.5, labelsize=7.0)
    ax_stab.xaxis.set_major_locator(FixedLocator((-0.1, 0, 0.1, 0.2, 0.3, 0.4)))
    ax_stab.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_stab.tick_params(axis="x", labelsize=7, width=0.6, pad=2)
    ax_stab.set_xlabel("ibUMAP minus baseline")
    ax_stab.xaxis.set_label_coords(0.5, XLABEL_Y)
    ax_stab.grid(axis="x", color=style.GRID_COLOR, alpha=0.65, linewidth=0.4)
    ax_stab.spines[["top", "right", "left"]].set_visible(False)

    titles = ("(a) CPU vs umap-learn", "(b) GPU, unseeded", "(c) 15-NN overlap across runs")
    for ax, text in zip((ax_cpu, ax_gpu, ax_stab), titles):
        fig.text(ax.get_position().x0, 1 - 0.028, text, ha="left", va="top", fontsize=7.6)

    # The rotated row labels must not reach back over panel (b).
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    right_of_b = ax_gpu.get_position().x1
    for label in ax_stab.get_yticklabels():
        x0 = inv.transform(label.get_window_extent().corners())[:, 0].min()
        if x0 < right_of_b:
            raise ValueError(f"Row label {label.get_text()!r} overlaps panel (b); increase GAP_BC.")
    style.save_figure(fig, stem, formats)


def main() -> None:
    args, paths = parse_args(__doc__.splitlines()[0], style.add_format_argument)
    data = _e2e.load(paths)
    runtime, stable = build_rows(data)
    out = paths.figures / "data"
    out.mkdir(exist_ok=True)
    write_csv(out / "runtime_data.csv", runtime)
    write_csv(out / "stability_data.csv", stable)
    style.configure_matplotlib()
    draw(runtime, stable, paths.figures / STEM, args.formats)
    summary = {
        "medians": {f"{k}/{p}": median(r["speedup"] for r in runtime
                                       if r["comparison"] == k and r["execution_profile"] == p)
                    for k, p in RUNTIME_SERIES},
        "runtime_wins": {f"{k}/{p}": sum(r["speedup"] > 1 for r in runtime
                                         if r["comparison"] == k and r["execution_profile"] == p)
                         for k, p in RUNTIME_SERIES},
        "overlap_median_delta": {k: median(r["paired_delta"] for r in stable
                                           if r["metric"] == STABILITY_METRIC and r["comparison"] == k)
                                 for k in EXPECTED_COUNTS},
        "overlap_wins": {k: sum(r["paired_delta"] > 0 for r in stable
                                if r["metric"] == STABILITY_METRIC and r["comparison"] == k)
                         for k in EXPECTED_COUNTS},
    }
    write_json(out / "figure_BE_summary.json", summary)
    print(f"ch5_main_figure.py: wrote {STEM} to {paths.figures}")


if __name__ == "__main__":
    main()
