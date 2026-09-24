#!/usr/bin/env python3
"""Plot per-run embeddings, per-dataset profile grids and repeat stability (not used in the paper).

Reads the published summary bundle and the saved embeddings; writes PNGs under
``results/04_embedding_plots/``. ``--skip-per-run`` draws only the grids.
"""

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
    dataset_entries,
    ensure_experiment_dirs,
    experiment_paths,
    file_sha256,
    load_configs,
    load_dataset,
    published_summary_root,
)
from common.paths import resolve_repo_path
from common.plotting import boxplot_with_labels, plot_embedding_overview, use_headless_matplotlib, use_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="dataset_filters")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--grid-repeat", type=int, default=1)
    parser.add_argument("--max-render-points", type=int, default=10000)
    parser.add_argument(
        "--robust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Limit axes to the 1st-99th coordinate percentiles (default: disabled).",
    )
    parser.add_argument("--skip-per-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> Any:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_embedding_path(paths: Any, row: dict[str, Any]) -> Path:
    local = paths.embeddings_root / "per_run" / str(row["dataset"]) / f"{row['run_id']}.npy"
    if local.is_file():
        return local
    recorded = resolve_repo_path(str(row.get("embedding_path") or "")) if row.get("embedding_path") else None
    if recorded is not None and recorded.is_file():
        return recorded
    return local


def load_labels(paths: Any, configs: dict[str, Any], dataset: str) -> np.ndarray | None:
    entries = {str(entry["name"]): entry for entry in dataset_entries(configs)}
    entry = entries.get(dataset)
    if entry is None:
        return None
    try:
        _, _, loaded_dataset, _ = load_dataset(entry, paths, mmap=True)
        labels = loaded_dataset.load_labels()
    except Exception:
        return None
    return None if labels is None else labels.values.astype(str)


def plot_stability(frame: Any, output: Path, dataset: str) -> None:
    if frame.empty:
        return
    use_headless_matplotlib(output.parent)
    import matplotlib.pyplot as plt

    use_theme()
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["procrustes_rms"] = ok["procrustes_rms"].astype(float)
    order = sorted(ok["algorithm_id"].unique())
    values = [ok.loc[ok["algorithm_id"] == algorithm, "procrustes_rms"].to_numpy() for algorithm in order]
    fig, ax = plt.subplots(figsize=(max(9, len(order) * 1.2), 5.5))
    boxplot_with_labels(ax, values, order, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Procrustes RMS (lower is better)")
    ax.set_title(f"{dataset}: repeat stability")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    summary_root = published_summary_root(paths, configs)
    runs = read_csv(summary_root / "run_summary.csv")
    stability = read_csv(summary_root / "stability_summary.csv")
    if runs.empty:
        print("No run_summary.csv rows; run 03_summarize_results.py first.")
        return
    runs = runs[runs["status"] == "ok"].copy()
    if args.dataset_filters:
        runs = runs[runs["dataset"].isin(args.dataset_filters)]
        if not stability.empty:
            stability = stability[stability["dataset"].isin(args.dataset_filters)]
    if args.profiles:
        runs = runs[runs["execution_profile"].isin(args.profiles)]
        if not stability.empty:
            stability = stability[stability["execution_profile"].isin(args.profiles)]
    runs["embedding_path"] = [
        str(resolve_embedding_path(paths, row))
        for row in runs.to_dict("records")
    ]
    if not args.dry_run:
        for row in runs.to_dict("records"):
            embedding_file = Path(str(row["embedding_path"]))
            expected_hash = str(row.get("embedding_sha256") or "")
            if not embedding_file.is_file() or not expected_hash:
                raise FileNotFoundError(
                    f"Published embedding is missing: {embedding_file}"
                )
            if file_sha256(embedding_file) != expected_hash:
                raise RuntimeError(
                    "Published embedding hash mismatch: "
                    f"{row['dataset']}/{row['run_id']}"
                )
    print(f"Successful embeddings selected: {len(runs)}")
    if args.dry_run:
        for dataset, part in runs.groupby("dataset"):
            print(f"- {dataset}: {len(part)} embeddings")
        return

    written = 0
    for dataset, part in runs.groupby("dataset", sort=True):
        labels = load_labels(paths, configs, str(dataset))
        if not args.skip_per_run:
            for row in part.to_dict("records"):
                embedding_file = Path(str(row["embedding_path"]))
                output = paths.embedding_plots_root / "per_run" / str(dataset) / f"{row['run_id']}.png"
                if plot_embedding_overview(
                    dataset=str(dataset),
                    variants=[str(row["run_id"])],
                    embedding_paths=[embedding_file],
                    output_path=output,
                    labels=labels,
                    robust=args.robust,
                    max_render_points=args.max_render_points,
                ):
                    written += 1
        for profile, profile_part in part.groupby("execution_profile", sort=True):
            grid = profile_part[profile_part["repeat"].astype(int) == args.grid_repeat].sort_values("algorithm_id")
            if grid.empty:
                continue
            suffix = "seed_42" if profile == "seeded" else "seed_none"
            output = paths.embedding_plots_root / "per_dataset" / f"{dataset}__{suffix}_grid.png"
            result = plot_embedding_overview(
                dataset=f"{dataset} ({profile})",
                variants=grid["algorithm_id"].astype(str).tolist(),
                embedding_paths=[Path(value) for value in grid["embedding_path"].astype(str)],
                output_path=output,
                labels=labels,
                robust=args.robust,
                max_render_points=args.max_render_points,
            )
            written += int(result is not None)
        if not stability.empty:
            plot_stability(
                stability[stability["dataset"] == dataset],
                paths.embedding_plots_root / "per_dataset" / f"{dataset}__repeat_stability.png",
                str(dataset),
            )
    print(f"Wrote {written} embedding plot(s) under {paths.embedding_plots_root}")


if __name__ == "__main__":
    main()
