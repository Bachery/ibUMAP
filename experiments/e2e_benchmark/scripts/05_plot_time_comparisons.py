#!/usr/bin/env python3
"""Plot runtime, speedup, scaling, repeat variance, breakdown, and failures (diagnostics, not used in the paper)."""

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


def scaling_plot(frame: Any, *, device: str, profile: str, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["device"] == device) & (frame["execution_profile"] == profile) & (frame["status"] == "ok")].copy()
    if part.empty:
        return
    runtime = runtime_column(part)
    grouped = part.groupby(["dataset", "algorithm_id"], as_index=False).agg(
        n_samples=("n_samples", "first"), runtime=(runtime, "median")
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for algorithm, values in grouped.groupby("algorithm_id"):
        values = values.sort_values("n_samples")
        ax.scatter(values["n_samples"], values["runtime"], s=24, alpha=0.8, label=algorithm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Synchronized end-to-end call wall time (s)")
    ax.set_title(f"{device.upper()} runtime scaling — {profile}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output)


def speedup_plot(frame: Any, *, device: str, output: Path) -> None:
    if frame.empty:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[frame["device"] == device].copy()
    if part.empty:
        return
    grouped = part.groupby(
        ["dataset", "execution_profile", "baseline_algorithm"],
        as_index=False,
    ).agg(
        n_samples=("n_samples", "first"), speedup=("speedup", "median")
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for (comparison, profile), values in grouped.groupby(
        ["baseline_algorithm", "execution_profile"]
    ):
        ax.scatter(
            values["n_samples"],
            values["speedup"],
            label=f"{comparison} — {profile}",
            s=28,
            alpha=0.8,
        )
    ax.axhline(1.0, color="#777777", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Comparison time / ibUMAP time")
    ax.set_title(f"{device.upper()} paired speedup")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output)


def distribution_plot(frame: Any, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    runtime = runtime_column(ok)
    labels = sorted(ok["algorithm_id"].unique())
    values = [ok.loc[ok["algorithm_id"] == label, runtime].astype(float).to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    boxplot_with_labels(ax, values, labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("End-to-end call wall time (s)")
    ax.set_title("Runtime distribution across datasets and repeats")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def runtime_cv_plot(frame: Any, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    runtime = runtime_column(ok)
    per_dataset = ok.groupby(["dataset", "algorithm_id"])[runtime].agg(["mean", "std"]).reset_index()
    per_dataset["cv"] = per_dataset["std"] / per_dataset["mean"]
    medians = per_dataset.groupby("algorithm_id")["cv"].median().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(medians.index, medians.values, color="#5477C4")
    ax.set_xlabel("Median within-dataset runtime CV")
    ax.set_title("Repeat timing stability")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def profile_delta_plot(frame: Any, output: Path) -> None:
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    runtime = runtime_column(ok)
    ok["implementation"] = ok["algorithm_id"].str.replace(r"_(opt_)?seed_(42|none)$", "", regex=True)
    pivot = ok.pivot_table(
        index=["dataset", "implementation", "repeat"],
        columns="execution_profile",
        values=runtime,
        aggfunc="last",
    ).dropna()
    if pivot.empty or not {"seeded", "unseeded"}.issubset(pivot.columns):
        return
    pivot["seeded_over_unseeded"] = pivot["seeded"] / pivot["unseeded"]
    data = pivot.reset_index()
    labels = sorted(data["implementation"].unique())
    values = [data.loc[data["implementation"] == label, "seeded_over_unseeded"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    boxplot_with_labels(ax, values, labels, showfliers=False)
    ax.axhline(1.0, color="#777777", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Seeded / unseeded runtime")
    ax.set_title("Seeded versus unseeded runtime")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def call_vs_core_plot(frame: Any, output: Path) -> None:
    runtime = runtime_column(frame)
    if frame.empty or "optimizer_compute_s" not in frame or runtime not in frame:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["status"] == "ok") & frame["optimizer_compute_s"].notna()].copy()
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for algorithm, values in part.groupby("algorithm_id"):
        ax.scatter(values["optimizer_compute_s"], values[runtime], label=algorithm, s=22, alpha=0.7)
    finite = np.concatenate([part["optimizer_compute_s"].astype(float).to_numpy(), part[runtime].astype(float).to_numpy()])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size:
        bounds = (finite.min(), finite.max())
        ax.plot(bounds, bounds, linestyle="--", color="#777777", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Internal optimizer time (s)")
    ax.set_ylabel("Synchronized end-to-end wall time (s)")
    ax.set_title("Call wall time versus internal optimizer time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output)


def ibumap_stage_breakdown_plot(frame: Any, output: Path) -> None:
    stage_columns = {
        "attr": "time_attr_time",
        "repl": "time_repl_time",
        "apply": "time_appl_time",
        "graph_pre": "time_graph_preprocess_time",
    }
    available = [name for name in stage_columns.values() if name in frame]
    if not available:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["status"] == "ok") & frame["algorithm_id"].astype(str).str.startswith("ibumap")].copy()
    if part.empty:
        return
    labels = sorted(part["algorithm_id"].unique())
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5.8))
    colors = {
        "attr": "#5477C4",
        "repl": "#CC6F47",
        "apply": "#6AA56A",
        "graph_pre": "#9B74C7",
    }
    for label, column in stage_columns.items():
        if column not in part:
            continue
        values = (
            part.groupby("algorithm_id")[column]
            .median()
            .reindex(labels)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        ax.bar(x, values, bottom=bottoms, label=label, color=colors.get(label))
        bottoms += values
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median recorded time (s)")
    ax.set_title("ibUMAP optimization-stage breakdown")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output)


def ibumap_e2e_stage_breakdown_plot(frame: Any, output: Path) -> None:
    required = {
        "time_fit_preprocess_time",
        "time_build_graph_time",
        "time_embd_init_time",
        "time_embd_opt_time",
    }
    runtime = runtime_column(frame)
    if runtime not in frame or not required.issubset(frame.columns):
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt
    import pandas as pd

    use_theme()
    part = frame[
        (frame["status"] == "ok")
        & frame["algorithm_id"].astype(str).str.startswith("ibumap")
    ].copy()
    if part.empty:
        return

    numeric_columns = [runtime, *sorted(required)]
    for column in numeric_columns:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    part = part.dropna(subset=numeric_columns)
    if part.empty:
        return

    # build_graph_time is nested inside fit_preprocess_time. Split it out so
    # every stacked segment is mutually exclusive and the total is meaningful.
    part["_graph"] = (
        part[["time_build_graph_time", "time_fit_preprocess_time"]]
        .clip(lower=0.0)
        .min(axis=1)
    )
    part["_preprocess_other"] = (
        part["time_fit_preprocess_time"] - part["_graph"]
    ).clip(lower=0.0)
    part["_initialization"] = part["time_embd_init_time"].clip(lower=0.0)
    part["_optimization"] = part["time_embd_opt_time"].clip(lower=0.0)
    accounted = (
        part["time_fit_preprocess_time"]
        + part["_initialization"]
        + part["_optimization"]
    )
    part["_wrapper_other"] = (part[runtime] - accounted).clip(lower=0.0)

    stages = [
        ("graph", "_graph", "#4C78A8"),
        ("other preprocessing", "_preprocess_other", "#72B7B2"),
        ("initialization", "_initialization", "#F2CF5B"),
        ("optimization", "_optimization", "#E45756"),
        ("wrapper / unaccounted", "_wrapper_other", "#B8B8B8"),
    ]
    labels = sorted(part["algorithm_id"].unique())
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.35), 5.8))
    grouped = part.groupby("algorithm_id")
    for stage_label, column, color in stages:
        values = (
            grouped[column]
            .median()
            .reindex(labels)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        ax.bar(x, values, bottom=bottoms, label=stage_label, color=color)
        bottoms += values

    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median recorded time (s)")
    ax.set_title("ibUMAP end-to-end stage breakdown")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output)


def stacked_stage_trend_plots(
    frame: Any,
    *,
    stages: list[tuple[str, str, str]],
    output_root: Path,
    title_prefix: str,
) -> None:
    """Plot per-algorithm stage times and normalized shares by sample count."""
    if frame.empty or "n_samples" not in frame:
        return
    use_headless_matplotlib(output_root)
    import matplotlib.pyplot as plt
    import pandas as pd

    use_theme()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_columns = [column for _, column, _ in stages]
    for algorithm_id, algorithm_rows in frame.groupby("algorithm_id"):
        part = algorithm_rows[["n_samples", *stage_columns]].copy()
        for column in part.columns:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["n_samples"])
        part = part[part["n_samples"] > 0]
        if part.empty:
            continue
        part[stage_columns] = part[stage_columns].fillna(0.0).clip(lower=0.0)
        grouped = (
            part.groupby("n_samples", as_index=True)[stage_columns]
            .median()
            .sort_index()
        )
        grouped = grouped.loc[grouped.sum(axis=1) > 0]
        if grouped.empty:
            continue

        x = grouped.index.to_numpy(dtype=float)
        values = [grouped[column].to_numpy(dtype=float) for column in stage_columns]
        labels = [label for label, _, _ in stages]
        colors = [color for _, _, color in stages]

        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        ax.stackplot(x, *values, labels=labels, colors=colors, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim(left=float(x.min()), right=float(x.max()))
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Median recorded time (s)")
        ax.set_title(f"{title_prefix} time — {algorithm_id}")
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_root / f"{algorithm_id}__stacked_time.png")

        totals = np.sum(np.vstack(values), axis=0)
        percentages = [
            np.divide(
                value,
                totals,
                out=np.zeros_like(value, dtype=float),
                where=totals > 0,
            )
            * 100.0
            for value in values
        ]
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        ax.stackplot(x, *percentages, labels=labels, colors=colors, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim(left=float(x.min()), right=float(x.max()))
        ax.set_ylim(0.0, 100.0)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Share of total recorded time (%)")
        ax.set_title(f"{title_prefix} share — {algorithm_id}")
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        save_figure(fig, output_root / f"{algorithm_id}__stacked_percent.png")


def ibumap_optimization_stage_trend_plots(frame: Any, output_root: Path) -> None:
    stage_columns = [
        ("attraction", "time_attr_time", "#5477C4"),
        ("repulsion", "time_repl_time", "#CC6F47"),
        ("apply", "time_appl_time", "#6AA56A"),
        ("graph preprocessing", "time_graph_preprocess_time", "#9B74C7"),
    ]
    required = {column for _, column, _ in stage_columns}
    total_column = (
        "time_embd_opt_time"
        if "time_embd_opt_time" in frame
        else "optimizer_compute_s"
    )
    if total_column not in frame or not required.issubset(frame.columns):
        return
    import pandas as pd

    part = frame[
        (frame["status"] == "ok")
        & frame["algorithm_id"].astype(str).str.startswith("ibumap")
    ].copy()
    numeric_columns = [total_column, *sorted(required)]
    for column in numeric_columns:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    part = part.dropna(subset=[total_column])
    if part.empty:
        return
    part[list(required)] = part[list(required)].fillna(0.0).clip(lower=0.0)
    accounted = part[list(required)].sum(axis=1)
    part["_optimization_other"] = (part[total_column] - accounted).clip(lower=0.0)
    stages = [
        *stage_columns,
        ("other / unaccounted", "_optimization_other", "#B8B8B8"),
    ]
    stacked_stage_trend_plots(
        part,
        stages=stages,
        output_root=output_root,
        title_prefix="ibUMAP optimization-stage",
    )


def ibumap_e2e_stage_trend_plots(frame: Any, output_root: Path) -> None:
    required = {
        "time_fit_preprocess_time",
        "time_build_graph_time",
        "time_embd_init_time",
        "time_embd_opt_time",
    }
    runtime = runtime_column(frame)
    if runtime not in frame or not required.issubset(frame.columns):
        return
    import pandas as pd

    part = frame[
        (frame["status"] == "ok")
        & frame["algorithm_id"].astype(str).str.startswith("ibumap")
    ].copy()
    numeric_columns = [runtime, *sorted(required)]
    for column in numeric_columns:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    part = part.dropna(subset=numeric_columns)
    if part.empty:
        return

    part["_graph"] = (
        part[["time_build_graph_time", "time_fit_preprocess_time"]]
        .clip(lower=0.0)
        .min(axis=1)
    )
    part["_preprocess_other"] = (
        part["time_fit_preprocess_time"] - part["_graph"]
    ).clip(lower=0.0)
    part["_initialization"] = part["time_embd_init_time"].clip(lower=0.0)
    part["_optimization"] = part["time_embd_opt_time"].clip(lower=0.0)
    accounted = (
        part["time_fit_preprocess_time"]
        + part["_initialization"]
        + part["_optimization"]
    )
    part["_wrapper_other"] = (part[runtime] - accounted).clip(lower=0.0)
    stages = [
        ("graph", "_graph", "#4C78A8"),
        ("other preprocessing", "_preprocess_other", "#72B7B2"),
        ("initialization", "_initialization", "#F2CF5B"),
        ("optimization", "_optimization", "#E45756"),
        ("wrapper / unaccounted", "_wrapper_other", "#B8B8B8"),
    ]
    stacked_stage_trend_plots(
        part,
        stages=stages,
        output_root=output_root,
        title_prefix="ibUMAP end-to-end stage",
    )


def ibumap_cache_diagnostic_plots(frame: Any, output_root: Path) -> None:
    diagnostics = [
        (
            "time_repl_kernel_cache_hit_rate",
            "FFT kernel cache hit rate",
            "Median hit rate",
            "ibumap_fft_kernel_cache_hit_rate.png",
        ),
        (
            "time_repl_fft_plan_cache_hit_rate",
            "FFT plan cache hit rate",
            "Median hit rate",
            "ibumap_fft_plan_cache_hit_rate.png",
        ),
        (
            "time_fused_m2p_update_rate",
            "Fused M2P/update activation",
            "Median fused epoch rate",
            "ibumap_fused_m2p_update_rate.png",
        ),
    ]
    use_headless_matplotlib(output_root)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["status"] == "ok") & frame["algorithm_id"].astype(str).str.startswith("ibumap")].copy()
    if part.empty:
        return
    for column, title, ylabel, filename in diagnostics:
        if column not in part or part[column].dropna().empty:
            continue
        medians = part.groupby("algorithm_id")[column].median().sort_values()
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(medians.index, medians.astype(float).to_numpy(), color="#5477C4")
        ax.set_xlim(0.0, max(1.0, float(medians.max()) * 1.05))
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        save_figure(fig, output_root / filename)


def ibumap_attraction_mode_heatmap(frame: Any, output: Path) -> None:
    if "time_attraction_kernel_mode" not in frame:
        return
    ok = frame[(frame["status"] == "ok") & frame["algorithm_id"].astype(str).str.startswith("ibumap")].copy()
    if ok.empty:
        return
    modes = sorted(str(value) for value in ok["time_attraction_kernel_mode"].dropna().unique())
    if not modes:
        return
    for mode in modes:
        ok[mode] = (ok["time_attraction_kernel_mode"].astype(str) == mode).astype(float)
    pivot = ok.pivot_table(index="algorithm_id", columns="device", values=modes, aggfunc="mean")
    pivot.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in pivot.columns
    ]
    save_heatmap(
        pivot,
        title="Resolved ibUMAP attraction kernel modes",
        subtitle="Fraction of successful runs reporting each mode; read from run_summary diagnostics.",
        output_path=output,
    )


def ibumap_p2m_mode_heatmap(frame: Any, output: Path) -> None:
    """Visualize the P2M route already reported by the optimizer."""
    column = "time_p2m_resolved_mode"
    if column not in frame:
        return
    ok = frame[
        (frame["status"] == "ok")
        & frame["algorithm_id"].astype(str).str.startswith("ibumap")
    ].copy()
    if ok.empty:
        return
    modes = sorted(
        {
            mode.strip()
            for value in ok[column].dropna()
            for mode in str(value).split(",")
            if mode.strip()
        }
    )
    if not modes:
        return
    for mode in modes:
        ok[mode] = ok[column].fillna("").map(
            lambda value: float(
                mode in {part.strip() for part in str(value).split(",")}
            )
        )
    pivot = ok.pivot_table(
        index="algorithm_id",
        columns="device",
        values=modes,
        aggfunc="mean",
    )
    pivot.columns = [
        "__".join(str(part) for part in item if str(part))
        if isinstance(item, tuple)
        else str(item)
        for item in pivot.columns
    ]
    save_heatmap(
        pivot,
        title="Resolved ibUMAP P2M modes",
        subtitle="Fraction of successful runs reporting each native optimizer route.",
        output_path=output,
    )


def ibumap_cpu_graph_thread_plot(frame: Any, output: Path) -> None:
    """Plot the CPU graph thread count already recorded by the pipeline."""
    column = "time_graph_n_jobs_effective"
    if column not in frame:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt
    import pandas as pd

    use_theme()
    part = frame[
        (frame["status"] == "ok")
        & (frame["device"] == "cpu")
        & frame["algorithm_id"].astype(str).str.startswith("ibumap")
    ].copy()
    part[column] = pd.to_numeric(part[column], errors="coerce")
    part = part.dropna(subset=[column])
    if part.empty:
        return
    medians = part.groupby("algorithm_id")[column].median().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(medians.index, medians.to_numpy(dtype=float), color="#5477C4")
    ax.set_xlabel("Median effective graph n_jobs")
    ax.set_title("Resolved ibUMAP CPU graph thread count")
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output)


def ibumap_calendar_diagnostic_plots(frame: Any, output_root: Path) -> None:
    required = [
        "time_attraction_calendar_build_time",
        "time_attraction_calendar_execute_time",
        "time_attraction_active_event_ratio",
        "time_attraction_active_row_ratio",
    ]
    if not any(column in frame for column in required):
        return
    use_headless_matplotlib(output_root)
    import matplotlib.pyplot as plt

    use_theme()
    part = frame[(frame["status"] == "ok") & frame["algorithm_id"].astype(str).str.startswith("ibumap")].copy()
    if part.empty:
        return
    time_columns = [
        column
        for column in (
            "time_attraction_calendar_build_time",
            "time_attraction_calendar_execute_time",
            "time_attraction_calendar_execute_zero_time",
            "time_attraction_calendar_execute_kernel_time",
        )
        if column in part and part[column].dropna().astype(float).sum() > 0
    ]
    if time_columns:
        labels = sorted(part["algorithm_id"].unique())
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels), dtype=float)
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5.6))
        for column in time_columns:
            values = (
                part.groupby("algorithm_id")[column]
                .median()
                .reindex(labels)
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )
            ax.bar(x, values, bottom=bottoms, label=column.removeprefix("time_attraction_calendar_"))
            bottoms += values
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Median recorded time (s)")
        ax.set_title("Attraction calendar substage times")
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, output_root / "ibumap_attraction_calendar_substage_times.png")

    ratio_columns = [column for column in required[2:] if column in part and part[column].dropna().any()]
    if ratio_columns:
        labels = sorted(part["algorithm_id"].unique())
        x = np.arange(len(labels))
        width = 0.8 / max(len(ratio_columns), 1)
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5.2))
        for offset, column in enumerate(ratio_columns):
            values = (
                part.groupby("algorithm_id")[column]
                .median()
                .reindex(labels)
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )
            ax.bar(x + (offset - (len(ratio_columns) - 1) / 2) * width, values, width=width, label=column.removeprefix("time_"))
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Median ratio")
        ax.set_title("Attraction calendar activity ratios")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, output_root / "ibumap_attraction_calendar_activity_ratios.png")


def failure_heatmap(frame: Any, output: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["failed"] = (~work["status"].isin(["ok"])).astype(float)
    pivot = work.pivot_table(index="dataset", columns="algorithm_id", values="failed", aggfunc="mean")
    save_heatmap(
        pivot,
        title="Failure / OOM / missing fraction",
        subtitle="1 means every expected repeat failed or is missing.",
        output_path=output,
    )


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    summary_root = published_summary_root(paths, configs)
    runs = read_csv(summary_root / "run_summary.csv")
    speedups = read_csv(summary_root / "speedup_pairs.csv")
    if runs.empty:
        print("No run summary rows; run 03_summarize_results.py first.")
        return
    if args.dry_run:
        print(f"Run rows: {len(runs)}; speedup pairs: {len(speedups)}")
        return
    for device in ("cpu", "cuda"):
        for profile, suffix in PROFILE_SUFFIX.items():
            scaling_plot(
                runs,
                device=device,
                profile=profile,
                output=paths.time_plots_root / f"{device}_runtime_vs_n_samples__{suffix}.png",
            )
        speedup_plot(speedups, device=device, output=paths.time_plots_root / f"{device}_speedup_vs_n_samples.png")
    distribution_plot(runs, paths.time_plots_root / "runtime_distribution_by_variant.png")
    runtime_cv_plot(runs, paths.time_plots_root / "runtime_cv_by_variant.png")
    profile_delta_plot(runs, paths.time_plots_root / "seeded_vs_unseeded_runtime_delta.png")
    call_vs_core_plot(runs, paths.time_plots_root / "call_wall_vs_core_time.png")
    ibumap_stage_breakdown_plot(runs, paths.time_plots_root / "ibumap_stage_breakdown.png")
    ibumap_e2e_stage_breakdown_plot(
        runs, paths.time_plots_root / "ibumap_e2e_stage_breakdown.png"
    )
    ibumap_optimization_stage_trend_plots(
        runs, paths.time_plots_root / "ibumap_stage_breakdown_by_n_samples"
    )
    ibumap_e2e_stage_trend_plots(
        runs, paths.time_plots_root / "ibumap_e2e_stage_breakdown_by_n_samples"
    )
    ibumap_cache_diagnostic_plots(runs, paths.time_plots_root)
    ibumap_attraction_mode_heatmap(runs, paths.time_plots_root / "ibumap_attraction_kernel_modes.png")
    ibumap_p2m_mode_heatmap(runs, paths.time_plots_root / "ibumap_p2m_modes.png")
    ibumap_cpu_graph_thread_plot(
        runs,
        paths.time_plots_root / "ibumap_cpu_graph_threads.png",
    )
    ibumap_calendar_diagnostic_plots(runs, paths.time_plots_root)
    failure_heatmap(runs, paths.time_plots_root / "failure_oom_heatmap.png")
    print(f"Wrote runtime plots under {paths.time_plots_root}")


if __name__ == "__main__":
    main()
