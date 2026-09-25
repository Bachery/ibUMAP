"""Shared matplotlib style for the paper figures (5.5-inch ICLR text width).

The manuscript figures were rendered with Times New Roman. Where it is not
installed matplotlib falls back to STIXGeneral, which changes glyph shapes and
text widths slightly but not the plotted values.
"""
from __future__ import annotations

import sys

import hashlib
from pathlib import Path

from _common import configure_matplotlib_environment

PAPER_TEXT_WIDTH_IN = 5.5
FONT_FAMILY = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
FONT_SIZE_PT = 8.0
AXIS_LABEL_SIZE_PT = 8.0
TICK_LABEL_SIZE_PT = 7.0
TEXT_COLOR = "#222222"
GRID_COLOR = "#D8D8D8"
REFERENCE_COLOR = "#666666"
BACKGROUND_COLOR = "white"
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7A5195"
PNG_DPI = 400

X_LIMITS = (100, 5_000_000)
SPEEDUP_LIMITS = (0.04, 30)
SPEEDUP_TICKS = (0.1, 1, 10)
SAMPLE_TICKS = (100, 1_000, 10_000, 100_000, 1_000_000)

COMPARISONS = (
    {"key": "cpu_umap", "label": "CPU / umap-learn", "candidate": "ibumap_cpu", "baseline": "umap_learn",
     "speedup_label": "Speedup over umap-learn", "color": BLUE},
    {"key": "gpu_cuml", "label": "GPU / cuML", "candidate": "ibumap_cuda", "baseline": "cuml_umap",
     "speedup_label": "Speedup over cuML", "color": ORANGE},
    {"key": "gpu_torchdr", "label": "GPU / TorchDR", "candidate": "ibumap_cuda", "baseline": "torchdr_umap",
     "speedup_label": "Speedup over TorchDR", "color": PURPLE},
)
STABILITY_METRICS = (
    ("neighbor_overlap_at_15", "15-NN overlap"),
    ("pairwise_distance_spearman", "Distance Spearman correlation"),
)

# Output formats; the manuscript uses PDF. Builders accept --formats to add svg/png.
OUTPUT_FORMATS = ("pdf",)


def add_format_argument(parser) -> None:
    parser.add_argument("--formats", nargs="+", default=list(OUTPUT_FORMATS), choices=("pdf", "svg", "png"))


def configure_matplotlib(*, rc_overrides: dict | None = None, hashsalt: str = "ibumap-paper-figures-v2") -> None:
    configure_matplotlib_environment()
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": FONT_FAMILY, "font.size": FONT_SIZE_PT,
        "mathtext.fontset": "stix", "text.usetex": False,
        "axes.labelsize": AXIS_LABEL_SIZE_PT, "axes.labelpad": 3.0,
        "axes.linewidth": 0.6, "axes.edgecolor": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR, "xtick.labelsize": TICK_LABEL_SIZE_PT,
        "ytick.labelsize": TICK_LABEL_SIZE_PT, "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR, "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": BACKGROUND_COLOR, "savefig.facecolor": BACKGROUND_COLOR,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "svg.hashsalt": hashsalt,
    })
    if rc_overrides:
        mpl.rcParams.update(rc_overrides)


def save_figure(fig, stem: Path, formats=OUTPUT_FORMATS, *, dpi: int = PNG_DPI, **savefig_kwargs) -> list[Path]:
    """Save with fixed metadata so repeated runs give identical files."""
    import matplotlib.pyplot as plt

    written = []
    for extension in formats:
        path = Path(stem).with_suffix("." + extension)
        metadata = {"pdf": {"CreationDate": None, "ModDate": None},
                    "svg": {"Date": None}, "png": {}}[extension]
        fig.savefig(path, dpi=dpi, metadata=metadata, **savefig_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def style_log_axis(ax, *, ylabel: str) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*SPEEDUP_LIMITS)
    ax.xaxis.set_major_locator(FixedLocator(SAMPLE_TICKS))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(FixedLocator(SPEEDUP_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color=GRID_COLOR, alpha=0.65, linewidth=0.4)
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.85, zorder=2)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(ylabel)
    ax.tick_params(which="both", width=0.6, pad=2.0, direction="out")
    ax.tick_params(which="major", length=3.0)
    ax.tick_params(which="minor", length=1.7)
    ax.spines[["top", "right"]].set_visible(False)


def comparison_for(key: str) -> dict:
    return next(item for item in COMPARISONS if item["key"] == key)


def fit_log_limits(paths, rows: list[dict], columns, limits: tuple[float, float], name: str,
                   margin: float = 1.25) -> tuple[float, float]:
    """Return ``limits`` if every value lies inside; otherwise widen them for new results.

    For the frozen data the configured limits are part of the paper's layout, so a
    value outside them is an error.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    values = [(row[c], row.get("dataset", "?"), c) for row in rows for c in columns]
    outside = [v for v in values if not limits[0] < v[0] < limits[1]]
    if not outside:
        return limits
    value, dataset, column = outside[0]
    if paths.frozen:
        raise ValueError(f"Configured {name} hide {dataset} ({column}={value}).")
    low = min([limits[0]] + [v / margin for v, _, _ in values if v > 0])
    high = max([limits[1]] + [v * margin for v, _, _ in values])
    print(f"NOTE: {name} widened to ({low:.3g}, {high:.3g}) to include {len(outside)} values", file=sys.stderr)
    return (low, high)


def deterministic_jitter(key: str, width: float = 0.13) -> float:
    """Stable vertical jitter derived from the dataset name."""
    fraction = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (fraction - 0.5) * 2 * width
