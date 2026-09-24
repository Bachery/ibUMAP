from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "orange": "#CC6F47",
}


def boxplot_with_labels(ax: Any, values: Any, labels: Sequence[str], **kwargs: Any) -> Any:
    """Call Matplotlib boxplot across the labels/tick_labels API rename."""
    parameters = inspect.signature(ax.boxplot).parameters
    label_kwarg = "tick_labels" if "tick_labels" in parameters else "labels"
    return ax.boxplot(values, **{label_kwarg: labels}, **kwargs)


def use_headless_matplotlib(mplconfig_dir: str | Path | None = None) -> None:
    if mplconfig_dir is not None:
        os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    import matplotlib

    matplotlib.use("Agg")


def use_theme() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "text.color": TOKENS["ink"],
            "font.family": "sans-serif",
        }
    )


def render_indices(n_rows: int, limit: int, *, seed: int = 42) -> np.ndarray:
    if limit <= 0 or limit >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=int(limit), replace=False).astype(np.int64))


def plot_embedding_overview(
    *,
    dataset: str,
    variants: Sequence[str],
    embedding_paths: Sequence[Path],
    output_path: Path,
    labels: np.ndarray | None = None,
    robust: bool = False,
    max_render_points: int = 0,
) -> Path | None:
    available = [(variant, path) for variant, path in zip(variants, embedding_paths) if path.exists()]
    if not available:
        return None
    use_headless_matplotlib(output_path.parent)
    import matplotlib.pyplot as plt

    use_theme()
    columns = min(4, len(variants))
    rows = math.ceil(len(variants) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, variant, path in zip(axes_flat, variants, embedding_paths):
        ax.set_title(variant, fontsize=9)
        if not path.exists():
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", color=TOKENS["muted"])
            ax.set_axis_off()
            continue
        embedding = np.load(path, mmap_mode="r")
        indices = render_indices(int(embedding.shape[0]), max_render_points)
        points = np.asarray(embedding[indices])
        plot_labels = labels[indices] if labels is not None and len(labels) == embedding.shape[0] else None
        if plot_labels is None or np.unique(plot_labels).size <= 1:
            ax.scatter(points[:, 0], points[:, 1], s=1.0, color=TOKENS["blue"], alpha=0.5, linewidths=0)
        else:
            for label in np.unique(plot_labels):
                mask = plot_labels == label
                ax.scatter(points[mask, 0], points[mask, 1], s=1.0, alpha=0.5, linewidths=0)
        if robust:
            x0, x1 = np.percentile(embedding[:, 0], [1.0, 99.0])
            y0, y1 = np.percentile(embedding[:, 1], [1.0, 99.0])
            ax.set_xlim(float(x0), float(x1))
            ax.set_ylim(float(y0), float(y1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_flat[len(variants):]:
        ax.set_axis_off()
    fig.suptitle(f"{dataset} embedding comparison", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_heatmap(frame: Any, *, title: str, subtitle: str, output_path: Path, diverging: bool = False) -> Path | None:
    if getattr(frame, "empty", False):
        return None
    use_headless_matplotlib(output_path.parent)
    import matplotlib.pyplot as plt

    use_theme()
    values = frame.to_numpy(dtype=float)
    width = max(9.0, 0.8 * frame.shape[1] + 4.0)
    height = max(4.8, 0.45 * frame.shape[0] + 2.8)
    fig, ax = plt.subplots(figsize=(width, height))
    if diverging:
        finite = values[np.isfinite(values)]
        vmax = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-12)
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    else:
        image = ax.imshow(values, aspect="auto", cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.72)
    ax.set_xticks(np.arange(frame.shape[1]), frame.columns)
    ax.set_yticks(np.arange(frame.shape[0]), frame.index)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title)
    if subtitle:
        fig.text(0.02, 0.93, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
