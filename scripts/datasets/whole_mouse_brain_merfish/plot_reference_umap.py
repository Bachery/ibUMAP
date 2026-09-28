from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
H5AD_ROOT = REPO_ROOT / "datasets/raw/whole_mouse_brain_merfish/h5ad"
FIGURE_ROOT = REPO_ROOT / "datasets/raw/whole_mouse_brain_merfish/reference_umap_figures"
RAW_FILE_NAMES = (
    "WB_MERFISH_animal1_coronal.h5ad",
    "WB_MERFISH_animal2_coronal.h5ad",
    "WB_MERFISH_animal3_sagittal.h5ad",
    "WB_MERFISH_animal4_sagittal.h5ad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot source H5AD obsm/X_umap embeddings for whole mouse brain MERFISH."
    )
    parser.add_argument("--all", action="store_true", help="Plot all downloaded H5AD files.")
    parser.add_argument("--list", action="store_true", help="List available raw H5AD files.")
    parser.add_argument("--dataset", help="Plot one H5AD file by name or stem.")
    parser.add_argument("--color-by", default="cell_type", help="H5AD obs column used for coloring.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help="Maximum points retained in each figure (default: 1000000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for plotting subsamples.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing figure.")
    return parser.parse_args()


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_text_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(decode_scalar(value)) for value in values], dtype=object)


def list_h5ad_files() -> list[Path]:
    return [H5AD_ROOT / name for name in RAW_FILE_NAMES if (H5AD_ROOT / name).exists()]


def select_h5ad_files(args: argparse.Namespace) -> list[Path]:
    files = list_h5ad_files()
    if args.list:
        return files
    if args.all:
        return files
    if args.dataset:
        query = args.dataset
        query_stem = Path(query).stem
        selected = [path for path in files if path.name == query or path.stem == query_stem]
        if not selected:
            raise FileNotFoundError(f"No downloaded H5AD matched {query!r}. Use --list to inspect names.")
        return selected
    raise ValueError("Provide --all, --list, or --dataset")


def sample_indices(n_rows: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("--max-points must be positive")
    if n_rows <= max_points:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False)).astype(np.int64)


def read_obs_values(obs_group: h5py.Group, column: str) -> np.ndarray:
    if column not in obs_group:
        raise KeyError(f"obs column {column!r} was not found")
    obj = obs_group[column]
    if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
        categories = decode_text_array(obj["categories"][:])
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        values = np.full(codes.shape[0], "", dtype=object)
        valid = (codes >= 0) & (codes < categories.shape[0])
        values[valid] = categories[codes[valid]]
        return values
    if isinstance(obj, h5py.Dataset):
        values = obj[:]
        if values.dtype.kind in {"O", "S", "U"}:
            return decode_text_array(values)
        return np.asarray(values)
    raise TypeError(f"Unsupported obs/{column} object: {type(obj).__name__}")


def point_size(n_points: int) -> float:
    if n_points >= 750_000:
        return 0.08
    if n_points >= 250_000:
        return 0.18
    if n_points >= 50_000:
        return 0.45
    return 1.5


def output_path_for(path: Path, color_by: str) -> Path:
    safe_color = re.sub(r"[^A-Za-z0-9._-]+", "_", color_by).strip("._-") or "values"
    return FIGURE_ROOT / f"{path.stem}_umap_{safe_color}.png"


def plot_one(path: Path, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    output_path = output_path_for(path, args.color_by)
    if output_path.exists() and not args.force:
        print(f"[{path.stem}] skip existing: {output_path}")
        return

    with h5py.File(path, "r") as handle:
        if "obsm/X_umap" not in handle:
            raise KeyError("H5AD does not contain obsm/X_umap")
        embedding = np.asarray(handle["obsm/X_umap"][:], dtype=np.float32)
        values = read_obs_values(handle["obs"], args.color_by)

    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"obsm/X_umap must have at least two columns, got shape={embedding.shape}")
    if values.shape[0] != embedding.shape[0]:
        raise ValueError(f"obs/{args.color_by} has {values.shape[0]} rows, expected {embedding.shape[0]}")

    selected = sample_indices(int(embedding.shape[0]), args.max_points, args.seed)
    points = embedding[selected, :2]
    plotted_values = values[selected]

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    numeric = np.issubdtype(np.asarray(plotted_values).dtype, np.number)
    if numeric:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=np.asarray(plotted_values, dtype=np.float32),
            s=point_size(points.shape[0]),
            linewidths=0,
            cmap="viridis",
            rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, label=args.color_by)
    else:
        labels, codes = np.unique(np.asarray(plotted_values, dtype=str), return_inverse=True)
        colors = plt.get_cmap("tab20")(np.arange(labels.shape[0]) % 20)
        if labels.shape[0] > 20:
            colors = plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95, labels.shape[0]))
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=colors[codes],
            s=point_size(points.shape[0]),
            linewidths=0,
            rasterized=True,
        )
        if labels.shape[0] <= 40:
            handles = [
                plt.Line2D([], [], color=colors[index], marker="o", linestyle="", markersize=4)
                for index in range(labels.shape[0])
            ]
            ax.legend(
                handles,
                labels.tolist(),
                title=args.color_by,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=6,
                title_fontsize=7,
            )
        else:
            ax.text(
                1.01,
                1.0,
                f"{args.color_by}\n{labels.shape[0]} categories\n(legend omitted)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

    ax.set_title(f"{path.stem}: source UMAP colored by {args.color_by}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{path.stem}] wrote {output_path} ({points.shape[0]:,} points)")


def main() -> int:
    args = parse_args()
    try:
        selected = select_h5ad_files(args)
        if args.list:
            for path in selected:
                print(path.name)
            return 0
        if not selected:
            raise FileNotFoundError(f"No expected H5AD files found under {H5AD_ROOT}")
        for path in selected:
            plot_one(path, args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
