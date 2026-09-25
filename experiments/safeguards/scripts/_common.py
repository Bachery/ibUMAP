"""Shared helpers of the safeguard failure-case experiment (paper Appendix B).

Paths, configuration, digests of the fixed inputs and the bounding-box statistics.
The statistics are the ones used by ``scripts/paper/ch4_safeguard_cases.py``:

* the core is the ``ceil(core_fraction * n)`` points nearest the coordinate-wise
  median (ties broken by index); its radius is the distance of the last core point;
* ``bbox_area_ratio`` is the axis-aligned bounding-box area of all points divided
  by that of the core;
* a point is far when its distance to the median exceeds
  ``far_radius_multiple`` core radii.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from scipy import sparse

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.paths import ensure_ibumap_importable, repo_relative  # noqa: E402

ensure_ibumap_importable()

CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "experiment.yaml"
PROTOCOL = "safeguard-failure-cases-v1"
PACKAGES = ("numpy", "scipy", "numba", "umap-learn", "pyfftw", "scikit-learn")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (EXPERIMENT_ROOT / value).resolve()


class Layout:
    """Output paths under ``results_root``."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.results = resolve(cfg["results_root"])

    @staticmethod
    def stem(variant: str, seed: int) -> str:
        return f"{variant}__seed_{int(seed)}"

    def embedding(self, dataset: str, variant: str, seed: int) -> Path:
        return self.results / "embeddings" / dataset / f"{self.stem(variant, seed)}.npy"

    def metadata(self, dataset: str, variant: str, seed: int) -> Path:
        return self.results / "metadata" / dataset / f"{self.stem(variant, seed)}.json"

    def trace(self, dataset: str, variant: str, seed: int) -> Path:
        return self.results / "traces" / dataset / f"{self.stem(variant, seed)}.csv"

    def snapshots(self, dataset: str, variant: str, seed: int) -> Path:
        return self.results / "snapshots" / dataset / self.stem(variant, seed)


def fixed_input_dir(cfg: Mapping[str, Any], dataset: str) -> Path:
    return resolve(cfg["fixed_inputs"]["root"]) / dataset / cfg["fixed_inputs"]["neighbors_dir"]


def load_fixed_inputs(cfg: Mapping[str, Any], dataset: str):
    """Optimizer graph (canonical CSR) and spectral initialization of one dataset."""
    root = fixed_input_dir(cfg, dataset)
    graph_path, init_path = root / cfg["fixed_inputs"]["graph_file"], root / cfg["fixed_inputs"]["init_file"]
    for path in (graph_path, init_path):
        if not path.exists():
            raise SystemExit(f"Missing fixed input {repo_relative(path)}; build it with "
                             f"experiments/mechanism/scripts/01_prepare_fixed_inputs.py (see README).")
    return sparse.load_npz(graph_path).tocsr(), np.load(init_path)


def csr_digest(graph) -> str:
    """Content digest of a CSR matrix (canonical order; int64 indices, float32 data)."""
    g = sparse.csr_matrix(graph, dtype=np.float32, copy=True)
    g.sum_duplicates()
    g.sort_indices()
    h = hashlib.sha256(json.dumps({"shape": list(g.shape), "nnz": int(g.nnz)}).encode())
    for array, dtype in ((g.indptr, np.int64), (g.indices, np.int64), (g.data, np.float32)):
        h.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return h.hexdigest()


def array_digest(array) -> str:
    """Content digest of a float32 array (shape and bytes; independent of the .npy header)."""
    a = np.ascontiguousarray(array, dtype=np.float32)
    h = hashlib.sha256(json.dumps({"shape": list(a.shape)}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def input_digests(graph, init) -> dict[str, Any]:
    return {"optimizer_graph": csr_digest(graph), "init_embedding": array_digest(init),
            "n": int(init.shape[0]), "optimizer_graph_nnz": int(graph.nnz)}


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def thread_count(cfg: Mapping[str, Any]) -> int:
    return max(1, min(int(cfg["threads"]), os.cpu_count() or 1))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def core_mask(embedding: np.ndarray, core_fraction: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Mask of the core points, distance of every point to the median, and the core radius."""
    y = np.asarray(embedding, dtype=np.float64)
    radius = np.linalg.norm(y - np.median(y, axis=0), axis=1)
    n_core = int(np.ceil(core_fraction * len(y)))
    order = np.argsort(radius, kind="stable")
    mask = np.zeros(len(y), dtype=bool)
    mask[order[:n_core]] = True
    return mask, radius, float(radius[order[n_core - 1]])


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def bbox_metrics(embedding: np.ndarray, core_fraction: float = 0.99,
                 far_radius_multiple: float = 2.0) -> dict[str, Any]:
    """Full versus core bounding-box statistics of one 2-D embedding."""
    y = np.asarray(embedding, dtype=np.float64)
    if not bool(np.isfinite(y).all()):
        return {"finite": False}
    mask, radius, core_radius = core_mask(y, core_fraction)
    x0, y0, x1, y1 = _bbox(y)
    cx0, cy0, cx1, cy1 = _bbox(y[mask])
    width, height = x1 - x0, y1 - y0
    core_width, core_height = cx1 - cx0, cy1 - cy0
    area, core_area = width * height, core_width * core_height
    far = radius > far_radius_multiple * core_radius
    return {
        "finite": True,
        "n": int(len(y)),
        "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
        "core_bbox_x0": cx0, "core_bbox_y0": cy0, "core_bbox_x1": cx1, "core_bbox_y1": cy1,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "core_bbox_width": core_width,
        "core_bbox_height": core_height,
        "core_bbox_area": core_area,
        "bbox_area_ratio": area / core_area if core_area > 0 else float("nan"),
        "bbox_extent_ratio": max(width / core_width, height / core_height)
        if core_width > 0 and core_height > 0 else float("nan"),
        "core_radius": core_radius,
        "radius_max": float(radius.max()),
        "radius_max_over_core": float(radius.max() / core_radius) if core_radius > 0 else float("nan"),
        "far_count": int(far.sum()),
        "far_pct": 100.0 * float(far.mean()),
    }
