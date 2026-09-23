from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "datasets/catalog.json"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/dataset_shapes_scatter.png"
LABEL_OFFSETS = {
    "bank": (30, 58),
    "breast_cancer": (22, -38),
    "cifar10": (-90, 44),
    "diabetes": (-66, 34),
    "digits": (18, -42),
    "foresttype": (-72, -10),
    "fashion_mnist": (18, 30),
    "gist_960_euclidean": (18, 36),
    "google_news_300d": (-158, -38),
    "iris": (18, 20),
    "mnist784": (18, -44),
    "seismic": (20, 42),
    "spambase": (18, -26),
    "wine": (-68, -34),
}


@dataclass(frozen=True)
class DatasetShape:
    dataset_id: str
    display_name: str
    rows: int
    columns: int

    @property
    def label(self) -> str:
        name = "\n".join(textwrap.wrap(self.display_name, width=24))
        return f"{name}\n{self.rows:,} x {self.columns:,}"

    @property
    def compact_label(self) -> str:
        name = self.display_name
        if name.startswith("Tabula Sapiens v2 "):
            name = "TSv2 " + name.removeprefix("Tabula Sapiens v2 ")
        name = "\n".join(textwrap.wrap(name, width=18))
        return f"{name}\n{self.rows:,} x {self.columns:,}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dataset row/column counts from datasets/catalog.json."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear axes instead of automatic log scaling.",
    )
    return parser.parse_args()


def _shape_from_entry(entry: dict[str, Any]) -> DatasetShape | None:
    shape = entry.get("feature_shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None

    rows, columns = shape[:2]
    if not isinstance(rows, int) or not isinstance(columns, int):
        return None
    if rows <= 0 or columns <= 0:
        return None

    dataset_id = str(entry.get("dataset_id") or "unknown_dataset")
    display_name = str(entry.get("display_name") or dataset_id)
    return DatasetShape(
        dataset_id=dataset_id,
        display_name=display_name,
        rows=rows,
        columns=columns,
    )


def load_dataset_shapes(catalog_path: Path) -> list[DatasetShape]:
    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)

    datasets = catalog.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError(f"{catalog_path} does not contain a 'datasets' list")

    shapes = [
        shape
        for entry in datasets
        if isinstance(entry, dict)
        for shape in [_shape_from_entry(entry)]
        if shape is not None
    ]
    if not shapes:
        raise ValueError(f"No valid feature_shape entries found in {catalog_path}")
    return sorted(shapes, key=lambda shape: (shape.rows, shape.columns, shape.dataset_id))


def _span(values: list[int]) -> float:
    return max(values) / min(values)


def _use_log_axis(values: list[int]) -> bool:
    return _span(values) >= 100


def _crowded_columns(shapes: list[DatasetShape]) -> set[int]:
    counts = Counter(shape.columns for shape in shapes)
    return {columns for columns, count in counts.items() if count >= 8}


def plot_dataset_shapes(
    shapes: list[DatasetShape], output_path: Path, *, force_linear: bool = False
) -> None:
    cache_root = Path(tempfile.gettempdir()) / "ibumap_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter

    rows = [shape.rows for shape in shapes]
    columns = [shape.columns for shape in shapes]
    crowded_columns = _crowded_columns(shapes)
    crowded_shapes = [
        shape for shape in shapes if shape.columns in crowded_columns
    ]
    main_label_shapes = [
        shape for shape in shapes if shape.columns not in crowded_columns
    ]
    color_norm = LogNorm(vmin=min(rows), vmax=max(rows))

    if crowded_shapes:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(18, 14),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 1.65]},
        )
        ax = axes[0]
        detail_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
        detail_ax = None

    scatter = ax.scatter(
        rows,
        columns,
        s=110,
        c=rows,
        cmap="viridis",
        norm=color_norm,
        alpha=0.86,
        edgecolors="#222222",
        linewidths=0.8,
    )

    if not force_linear and _use_log_axis(rows):
        ax.set_xscale("log")
    if not force_linear and _use_log_axis(columns):
        ax.set_yscale("log")

    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    for index, shape in enumerate(main_label_shapes):
        offset = LABEL_OFFSETS.get(shape.dataset_id, (8, 8 + 12 * (index % 3)))
        ax.annotate(
            shape.label,
            (shape.rows, shape.columns),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.6,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.82,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )

    ax.set_title("Dataset Shapes in Catalog")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Columns")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.margins(x=0.12, y=0.18)

    if detail_ax is not None:
        detail_rows = [shape.rows for shape in crowded_shapes]
        lane_count = min(18, max(12, (len(crowded_shapes) + 1) // 2))
        lane_spacing = 1.65
        lanes = [
            (index % lane_count) * lane_spacing
            for index, _ in enumerate(crowded_shapes)
        ]
        detail_ax.scatter(
            detail_rows,
            lanes,
            s=72,
            c=detail_rows,
            cmap="viridis",
            norm=color_norm,
            alpha=0.86,
            edgecolors="#222222",
            linewidths=0.6,
        )
        detail_ax.set_xscale("log")
        detail_ax.xaxis.set_major_formatter(formatter)
        detail_ax.set_yticks([])
        detail_ax.set_ylim(-1.4, (lane_count - 1) * lane_spacing + 2.8)
        detail_ax.set_title(
            "Expanded Label Lanes for Crowded 50-Column Datasets", fontsize=12
        )
        detail_ax.set_xlabel("Rows")
        detail_ax.grid(True, axis="x", which="both", linestyle="--", linewidth=0.5, alpha=0.35)
        detail_ax.margins(x=0.08)
        for index, shape in enumerate(crowded_shapes):
            lane = lanes[index]
            y_offset = 8 if index % 2 == 0 else -14
            detail_ax.annotate(
                shape.compact_label,
                (shape.rows, lane),
                xytext=(4, y_offset),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=5.6,
                linespacing=1.0,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#d0d0d0",
                    "alpha": 0.82,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777777",
                    "linewidth": 0.4,
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
            )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Rows (log scale)")
    colorbar.ax.yaxis.set_major_formatter(formatter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    shapes = load_dataset_shapes(args.catalog)
    plot_dataset_shapes(shapes, args.output, force_linear=args.linear)
    print(f"Wrote {args.output} with {len(shapes)} dataset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
