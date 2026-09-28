#!/usr/bin/env python3
"""Plot quality deltas, ranks, Pareto fronts, stability, and metric failures (diagnostics, not used in the paper)."""

from __future__ import annotations

from _terminal_log import reexec_with_terminal_log

if __name__ == "__main__":
    reexec_with_terminal_log(__file__)

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    ensure_experiment_dirs,
    experiment_paths,
    load_configs,
    published_summary_root,
)
from common.plotting import boxplot_with_labels, save_heatmap, use_headless_matplotlib, use_theme


PROFILE_SUFFIX = {"seeded": "seed_42", "unseeded": "seed_none"}


def runtime_column(frame: Any) -> str:
    return "e2e_call_wall_s" if "e2e_call_wall_s" in frame else "optimization_call_wall_s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def save_figure(fig: Any, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def paired_delta_plot(frame: Any, metric: str, profile: str, output: Path) -> None:
    if frame.empty:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["metric"] == metric) & (frame["execution_profile"] == profile)].copy()
    if part.empty:
        return
    grouped = part.groupby(
        ["dataset", "device", "baseline_algorithm"],
        as_index=False,
    )["signed_improvement"].median()
    grouped["comparison"] = (
        grouped["dataset"].astype(str)
        + " — "
        + grouped["baseline_algorithm"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.2 * len(grouped) + 2)))
    colors = grouped["device"].map({"cpu": "#5477C4", "cuda": "#CC6F47"})
    ax.scatter(grouped["signed_improvement"], np.arange(len(grouped)), c=colors, s=24)
    ax.axvline(0.0, color="#777777", linewidth=1)
    ax.set_yticks(np.arange(len(grouped)), grouped["comparison"], fontsize=7)
    ax.set_xlabel("Signed improvement of ibUMAP over comparison")
    ax.set_title(f"{metric}: paired delta — {profile}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def distribution_plot(frame: Any, metric: str, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["metric"] == metric) & (frame["status"] == "ok")].copy()
    if part.empty:
        return
    labels = sorted(part["algorithm_id"].unique())
    values = [part.loc[part["algorithm_id"] == label, "value"].astype(float).to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5.8))
    boxplot_with_labels(ax, values, labels, showfliers=False)
    direction = "higher" if bool(part["higher_is_better"].iloc[0]) else "lower"
    ax.set_ylabel(f"{metric} ({direction} is better)")
    ax.set_title(f"{metric}: distribution by variant")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def win_loss_heatmap(frame: Any, output: Path) -> None:
    if frame.empty:
        return
    counts = frame.groupby(
        ["metric", "baseline_algorithm", "execution_profile", "outcome"]
    ).size().unstack(fill_value=0)
    for column in ("win", "tie", "loss"):
        if column not in counts:
            counts[column] = 0
    score = (counts["win"] - counts["loss"]).unstack(
        ["baseline_algorithm", "execution_profile"]
    )
    save_heatmap(
        score,
        title="Quality win-minus-loss counts",
        subtitle="Positive values favor ibUMAP over the comparison algorithm; metric direction is normalized before counting.",
        output_path=output,
        diverging=True,
    )


def rank_heatmap(frame: Any, output: Path) -> None:
    if frame.empty:
        return
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    ranks = []
    for (dataset, profile, repeat, metric), part in ok.groupby(["dataset", "execution_profile", "repeat", "metric"]):
        ascending = not bool(part["higher_is_better"].iloc[0])
        local = part[["algorithm_id", "value"]].copy()
        local["rank"] = local["value"].astype(float).rank(method="average", ascending=ascending)
        local["metric"] = metric
        ranks.append(local)
    if not ranks:
        return
    import pandas as pd

    combined = pd.concat(ranks, ignore_index=True)
    pivot = combined.pivot_table(index="metric", columns="algorithm_id", values="rank", aggfunc="mean")
    save_heatmap(
        pivot,
        title="Mean quality rank",
        subtitle="Lower rank is better; ranks are computed within each dataset/profile/repeat.",
        output_path=output,
    )


def pareto_plots(quality: Any, runs: Any, output_root: Path) -> None:
    runtime = runtime_column(runs)
    if quality.empty or runs.empty or runtime not in runs:
        return
    okq = quality[quality["status"] == "ok"].copy()
    okr = runs[runs["status"] == "ok"][["dataset", "run_id", runtime]]
    merged = okq.merge(okr, on=["dataset", "run_id"], how="inner")
    for metric, part in merged.groupby("metric"):
        use_headless_matplotlib(output_root)
        import matplotlib.pyplot as plt

        use_theme()
        grouped = part.groupby(["algorithm_id", "execution_profile"], as_index=False).agg(
            quality=("value", "median"), runtime=(runtime, "median")
        )
        if grouped.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        for _, row in grouped.iterrows():
            marker = "o" if row["execution_profile"] == "seeded" else "s"
            ax.scatter(row["runtime"], row["quality"], marker=marker, s=55)
            ax.annotate(row["algorithm_id"], (row["runtime"], row["quality"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Median end-to-end call wall time (s)")
        ax.set_ylabel(str(metric))
        ax.set_title(f"{metric}: quality/runtime trade-off")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save_figure(fig, output_root / f"{metric}__quality_vs_runtime_pareto.png")


def stability_plot(frame: Any, value: str, ylabel: str, output: Path, *, log: bool = False) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = frame[(frame["status"] == "ok") & frame[value].notna()].copy()
    if ok.empty:
        return
    labels = sorted(ok["algorithm_id"].unique())
    values = [ok.loc[ok["algorithm_id"] == label, value].astype(float).to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5.8))
    boxplot_with_labels(ax, values, labels, showfliers=False)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title("Repeat stability by variant")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def profile_quality_delta(quality: Any, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = quality[quality["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["implementation"] = ok["algorithm_id"].str.replace(r"_(opt_)?seed_(42|none)$", "", regex=True)
    pivot = ok.pivot_table(
        index=["dataset", "implementation", "repeat", "metric"],
        columns="execution_profile",
        values="value",
        aggfunc="last",
    ).dropna()
    if pivot.empty or not {"seeded", "unseeded"}.issubset(pivot.columns):
        return
    pivot["delta"] = pivot["seeded"] - pivot["unseeded"]
    data = pivot.reset_index()
    metrics = sorted(data["metric"].unique())
    values = [data.loc[data["metric"] == metric, "delta"].to_numpy() for metric in metrics]
    fig, ax = plt.subplots(figsize=(9, 5.7))
    boxplot_with_labels(ax, values, metrics, showfliers=False)
    ax.axhline(0.0, color="#777777", linewidth=1)
    ax.set_ylabel("Seeded minus unseeded quality")
    ax.set_title("Seeded/unseeded quality delta (raw metric direction)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def failure_heatmap(frame: Any, output: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["failed"] = (work["status"] != "ok").astype(float)
    pivot = work.pivot_table(index="dataset", columns="algorithm_id", values="failed", aggfunc="mean")
    save_heatmap(
        pivot,
        title="Metric failure / missing fraction",
        subtitle="Fraction across repeats and configured metrics.",
        output_path=output,
    )


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    summary_root = published_summary_root(paths, configs)
    quality = read_csv(summary_root / "quality_long.csv")
    pairs = read_csv(summary_root / "quality_pairs.csv")
    stability = read_csv(summary_root / "stability_summary.csv")
    runs = read_csv(summary_root / "run_summary.csv")
    if quality.empty:
        print("No quality summary rows; run evaluation and 03_summarize_results.py first.")
        return
    if args.dry_run:
        print(f"Quality rows: {len(quality)}; paired rows: {len(pairs)}; stability rows: {len(stability)}")
        return
    metrics = sorted(quality["metric"].dropna().unique())
    for metric in metrics:
        distribution_plot(quality, str(metric), paths.metric_plots_root / f"{metric}__distribution_by_variant.png")
        for profile, suffix in PROFILE_SUFFIX.items():
            paired_delta_plot(
                pairs,
                str(metric),
                profile,
                paths.metric_plots_root / f"{metric}__paired_delta__{suffix}.png",
            )
    win_loss_heatmap(pairs, paths.metric_plots_root / "quality_win_tie_loss_heatmap.png")
    rank_heatmap(quality, paths.metric_plots_root / "quality_rank_heatmap.png")
    pareto_plots(quality, runs, paths.metric_plots_root)
    if not stability.empty:
        stability_plot(
            stability,
            "procrustes_rms",
            "Procrustes RMS (lower is better)",
            paths.metric_plots_root / "procrustes_repeat_stability.png",
            log=True,
        )
        stability_plot(
            stability,
            "neighbor_overlap_at_15",
            "Repeated-neighbor overlap@15 (higher is better)",
            paths.metric_plots_root / "neighbor_overlap_repeat_stability.png",
        )
    profile_quality_delta(quality, paths.metric_plots_root / "seeded_vs_unseeded_quality_delta.png")
    failure_heatmap(quality, paths.metric_plots_root / "metric_failure_heatmap.png")
    print(f"Wrote quality plots under {paths.metric_plots_root}")


if __name__ == "__main__":
    main()
