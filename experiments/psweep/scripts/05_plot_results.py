#!/usr/bin/env python3
"""Plot runtime, peak memory, quality and embedding stability of all variants (diagnostics, not in the paper)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import EXPERIMENT_ROOT, ensure_directories, experiment_paths, load_configs


VARIANTS = ("p1", "p2", "p3", "schedule")
COLORS = {"p1": "#4c78a8", "p2": "#f58518", "p3": "#e45756", "schedule": "#54a24b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def save(fig: plt.Figure, root: Path, name: str, dpi: int) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root / f"{name}.{suffix}", dpi=dpi if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def scaling_plot(runs: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    ok = runs[runs["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("runs.csv has no successful rows")
    median = (
        ok.groupby(["dataset", "device", "variant", "expected_rows"], as_index=False)[
            ["optimization_wall_s", "memory_primary_peak_bytes"]
        ]
        .median()
        .sort_values("expected_rows")
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
    for column, device in enumerate(("cpu", "cuda")):
        device_rows = median[median["device"] == device]
        for variant in VARIANTS:
            rows = device_rows[device_rows["variant"] == variant]
            if rows.empty:
                continue
            axes[0, column].plot(
                rows["expected_rows"],
                rows["optimization_wall_s"],
                marker="o",
                markersize=3,
                linewidth=1,
                color=COLORS[variant],
                label=variant,
            )
            axes[1, column].plot(
                rows["expected_rows"],
                rows["memory_primary_peak_bytes"] / (1024**3),
                marker="o",
                markersize=3,
                linewidth=1,
                color=COLORS[variant],
                label=variant,
            )
        axes[0, column].set_title(device.upper())
        axes[0, column].set_ylabel("optimization wall time (s)")
        axes[1, column].set_ylabel("peak memory (GiB)")
        axes[1, column].set_xlabel("dataset rows")
        for row in axes[:, column]:
            row.set_xscale("log")
            row.set_yscale("log")
            row.grid(alpha=0.25)
        axes[0, column].legend(title="FFT policy", frameon=False)
    fig.suptitle("Fixed interpolation order versus persistent FFT schedule", y=1.01)
    save(fig, plots_root, "runtime_memory_scaling", dpi)


def distribution_plot(runs: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    ok = runs[runs["status"] == "ok"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(VARIANTS))
    width = 0.36
    for device_index, device in enumerate(("cpu", "cuda")):
        device_rows = ok[ok["device"] == device]
        time_values = [
            device_rows[device_rows["variant"] == variant]["optimization_wall_s"].dropna().to_numpy()
            for variant in VARIANTS
        ]
        memory_values = [
            device_rows[device_rows["variant"] == variant]["memory_primary_peak_bytes"].dropna().to_numpy() / (1024**3)
            for variant in VARIANTS
        ]
        for axis, values, ylabel in (
            (axes[0], time_values, "optimization wall time (s)"),
            (axes[1], memory_values, "peak memory (GiB)"),
        ):
            positions = x + (device_index - 0.5) * width
            medians = [float(np.median(value)) if value.size else np.nan for value in values]
            q25 = [float(np.percentile(value, 25)) if value.size else np.nan for value in values]
            q75 = [float(np.percentile(value, 75)) if value.size else np.nan for value in values]
            axis.bar(
                positions,
                medians,
                width=width,
                label=device.upper(),
                alpha=0.78,
                yerr=[np.asarray(medians) - np.asarray(q25), np.asarray(q75) - np.asarray(medians)],
                capsize=3,
            )
            axis.set_ylabel(ylabel)
            axis.set_yscale("log")
            axis.set_xticks(x, VARIANTS)
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Median and IQR across successful tasks")
    axes[1].set_title("Device-specific primary memory measure")
    save(fig, plots_root, "runtime_memory_distribution", dpi)


def quality_plot(scores: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    ok = scores[(scores["status"] == "ok") & scores["value"].notna()].copy()
    metrics = list(dict.fromkeys(ok["metric"].tolist()))
    if not metrics:
        raise ValueError("quality_scores.csv has no successful scores")
    ncols = min(3, len(metrics))
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4 * nrows), squeeze=False)
    x = np.arange(len(VARIANTS))
    width = 0.36
    for metric, axis in zip(metrics, axes.flat):
        rows = ok[ok["metric"] == metric]
        for device_index, device in enumerate(("cpu", "cuda")):
            values = [
                rows[(rows["device"] == device) & (rows["variant"] == variant)]["value"].to_numpy()
                for variant in VARIANTS
            ]
            medians = [float(np.median(value)) if value.size else np.nan for value in values]
            axis.bar(
                x + (device_index - 0.5) * width,
                medians,
                width=width,
                alpha=0.78,
                label=device.upper(),
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(x, VARIANTS)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Quality scores (median across datasets and repeats)")
    save(fig, plots_root, "evaluation_quality", dpi)


def stability_plot(stability: pd.DataFrame, plots_root: Path, dpi: int) -> None:
    ok = stability[stability["status"] == "ok"].copy()
    measures = (
        ("procrustes_rms", "Procrustes RMS ↓"),
        ("pairwise_distance_spearman", "pairwise-distance Spearman ↑"),
        ("neighbor_overlap_at_15", "neighbor overlap@15 ↑"),
    )
    if ok.empty:
        raise ValueError("stability.csv has no successful comparisons")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(VARIANTS))
    width = 0.36
    for axis, (measure, label) in zip(axes, measures):
        for device_index, device in enumerate(("cpu", "cuda")):
            rows = ok[ok["device"] == device]
            values = [
                rows[rows["variant"] == variant][measure].dropna().to_numpy()
                for variant in VARIANTS
            ]
            medians = [float(np.median(value)) if value.size else np.nan for value in values]
            axis.bar(
                x + (device_index - 0.5) * width,
                medians,
                width=width,
                alpha=0.78,
                label=device.upper(),
            )
        axis.set_title(label)
        axis.set_xticks(x, VARIANTS)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Final embedding stability across paired random seeds")
    save(fig, plots_root, "embedding_stability", dpi)


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(configs, config_dir)
    ensure_directories(paths)
    runs = pd.read_csv(paths.summaries_root / "runs.csv")
    scores = pd.read_csv(paths.summaries_root / "quality_scores.csv")
    stability = pd.read_csv(paths.summaries_root / "stability.csv")
    scaling_plot(runs, paths.plots_root, args.dpi)
    distribution_plot(runs, paths.plots_root, args.dpi)
    quality_plot(scores, paths.plots_root, args.dpi)
    stability_plot(stability, paths.plots_root, args.dpi)
    print(paths.plots_root)


if __name__ == "__main__":
    main()
