from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path
from typing import Any

import numpy as np


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.metadata import read_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"
FIGURE_ROOT = REPO_ROOT / "datasets/raw/tabula_sapiens_v2/reference_umap_figures"
UMAP_PANEL_SIZE = 7.0
MIN_LEGEND_WIDTH = 2.2
NUMERIC_LEGEND_WIDTH = 0.8
MAX_LEGEND_ROWS = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Tabula Sapiens v2 reference UMAP embeddings from processed datasets."
    )
    parser.add_argument("--dataset", help="Processed dataset_id to plot.")
    parser.add_argument("--all", action="store_true", help="Plot all Tabula Sapiens v2 datasets.")
    parser.add_argument("--limit", type=int, help="Only plot the first N selected datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing figures.")
    parser.add_argument("--color-by", help="obs.csv.gz column to color by.")
    return parser.parse_args()


def iter_dataset_dirs() -> list[Path]:
    dirs: list[Path] = []
    for metadata_path in sorted(PROCESSED_ROOT.glob("*/metadata.json")):
        try:
            metadata = read_metadata(metadata_path)
        except Exception:
            continue
        if metadata.get("family") == "tabula_sapiens_v2" and metadata.get("kind") == "single_cell":
            dirs.append(metadata_path.parent)
    return dirs


def selected_dataset_dirs(args: argparse.Namespace) -> list[Path]:
    if args.all:
        selected = iter_dataset_dirs()
    elif args.dataset:
        selected = [PROCESSED_ROOT / args.dataset]
    else:
        raise ValueError("Provide --dataset or --all")
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def read_obs_column(path: Path, column: str) -> list[str]:
    values: list[str] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise KeyError(f"Column {column!r} not found in {path}")
        for row in reader:
            values.append(row.get(column, ""))
    return values


def point_size(n_points: int) -> float:
    if n_points >= 1_000_000:
        return 0.05
    if n_points >= 250_000:
        return 0.1
    if n_points >= 50_000:
        return 0.3
    if n_points >= 10_000:
        return 0.8
    return 2.0


def categorical_legend_layout(labels: list[str]) -> tuple[int, int, float, float, float]:
    n_labels = max(len(labels), 1)
    rows = min(MAX_LEGEND_ROWS, n_labels)
    columns = int(np.ceil(n_labels / rows))
    longest_label = max((len(label) for label in labels), default=0)
    column_width = max(1.4, min(7.5, 0.075 * longest_label + 0.55))
    legend_width = max(MIN_LEGEND_WIDTH, columns * column_width)
    font_size = 7.0
    if n_labels > 120:
        font_size = 5.0
    elif n_labels > 80:
        font_size = 5.5
    elif n_labels > 50:
        font_size = 6.0
    return rows, columns, column_width, legend_width, font_size


def make_figure(legend_width: float):
    import matplotlib.pyplot as plt

    fig_width = UMAP_PANEL_SIZE + legend_width + 0.5
    fig = plt.figure(figsize=(fig_width, UMAP_PANEL_SIZE))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[UMAP_PANEL_SIZE, legend_width],
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    ax.set_box_aspect(1)
    return fig, ax, legend_ax


def draw_categorical_legend(
    legend_ax,
    labels: list[str],
    colors: np.ndarray,
    title: str,
    rows: int,
    column_width: float,
    font_size: float,
) -> None:
    legend_ax.set_axis_off()
    columns = int(np.ceil(max(len(labels), 1) / rows))
    legend_ax.set_xlim(0, columns * column_width)
    legend_ax.set_ylim(0, rows + 1.4)
    legend_ax.text(
        0,
        rows + 0.9,
        title,
        fontsize=max(font_size + 1.0, 7.0),
        fontweight="bold",
        ha="left",
        va="top",
    )
    for index, label in enumerate(labels):
        column = index // rows
        row = index % rows
        x = column * column_width
        y = rows - row - 0.15
        legend_ax.scatter([x + 0.08], [y], s=18, color=[colors[index]], linewidths=0)
        legend_ax.text(
            x + 0.22,
            y,
            label,
            fontsize=font_size,
            ha="left",
            va="center",
        )


def plot_one(dataset_dir: Path, color_by: str | None, force: bool) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import BoundaryNorm, ListedColormap

    metadata_path = dataset_dir / "metadata.json"
    metadata = read_metadata(metadata_path)
    dataset_id = str(metadata["dataset_id"])
    reference_file = metadata.get("reference_embedding_file")
    if not reference_file:
        print(f"[{dataset_id}] skip: no reference_embedding_file")
        return

    reference_path = dataset_dir / str(reference_file)
    obs_path = dataset_dir / str(metadata.get("obs_file", "obs.csv.gz"))
    output_path = FIGURE_ROOT / f"{dataset_id}_umap.png"
    if output_path.exists() and not force:
        print(f"[{dataset_id}] skip existing: {output_path}")
        return

    selected_color = color_by or metadata.get("default_color_by")
    if not selected_color:
        print(f"[{dataset_id}] skip: no default_color_by and --color-by not provided")
        return

    embedding = np.load(reference_path, mmap_mode="r")
    values = read_obs_column(obs_path, selected_color)
    if len(values) != int(embedding.shape[0]):
        raise ValueError(
            f"{dataset_id} row mismatch: embedding={embedding.shape[0]}, obs={len(values)}"
        )

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    value_series = pd.Series(values)
    numeric_values = pd.to_numeric(value_series, errors="coerce")
    if numeric_values.notna().all():
        fig, ax, legend_ax = make_figure(NUMERIC_LEGEND_WIDTH)
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=numeric_values.to_numpy(),
            s=point_size(int(embedding.shape[0])),
            cmap="viridis",
            linewidths=0,
        )
        colorbar = fig.colorbar(scatter, cax=legend_ax)
        colorbar.set_label(selected_color)
    else:
        codes, uniques = pd.factorize(value_series.astype(str), sort=True)
        labels = [str(value) for value in uniques]
        rows, _columns, column_width, legend_width, font_size = categorical_legend_layout(labels)
        fig, ax, legend_ax = make_figure(legend_width)
        if len(labels) <= 20:
            color_values = plt.get_cmap("tab20")(np.arange(len(labels)) % 20)
        else:
            color_values = plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95, len(labels)))
        cmap = ListedColormap(color_values)
        norm = BoundaryNorm(np.arange(-0.5, len(labels) + 0.5, 1), cmap.N)
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=codes,
            s=point_size(int(embedding.shape[0])),
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )
        draw_categorical_legend(
            legend_ax,
            labels,
            color_values,
            selected_color,
            rows,
            column_width,
            font_size,
        )

    ax.set_title(f"{dataset_id} reference UMAP colored by {selected_color}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset_id}] wrote {output_path}")


def main() -> int:
    args = parse_args()
    try:
        selected = selected_dataset_dirs(args)
        if not selected:
            print("No Tabula Sapiens v2 processed datasets found.", file=sys.stderr)
            return 1
        for dataset_dir in selected:
            plot_one(dataset_dir, args.color_by, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
