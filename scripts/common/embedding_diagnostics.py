from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.run_metadata import write_json


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def bbox_summary(embedding: np.ndarray) -> dict[str, float]:
    points = np.asarray(embedding, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Embedding must be 2D with at least two columns: shape={points.shape}")
    x, y = points[:, 0], points[:, 1]
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    rx0, rx1 = (float(value) for value in np.percentile(x, [1.0, 99.0]))
    ry0, ry1 = (float(value) for value in np.percentile(y, [1.0, 99.0]))
    width, height = x1 - x0, y1 - y0
    robust_width, robust_height = rx1 - rx0, ry1 - ry0
    area, robust_area = width * height, robust_width * robust_height
    outside = (x < rx0) | (x > rx1) | (y < ry0) | (y > ry1)
    radial = np.sqrt(x * x + y * y)
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "robust_bbox_width": robust_width,
        "robust_bbox_height": robust_height,
        "robust_bbox_area": robust_area,
        "bbox_area_expansion_ratio": safe_ratio(area, robust_area),
        "outlier_pct_p01_p99_box": 100.0 * float(np.mean(outside)),
        "radial_p95": float(np.percentile(radial, 95.0)),
        "radial_p99": float(np.percentile(radial, 99.0)),
        "radial_p999": float(np.percentile(radial, 99.9)),
    }


def encode_labels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    text = np.asarray(values).astype(str)
    names = sorted(set(text.tolist()))
    mapping = {name: index for index, name in enumerate(names)}
    encoded = np.fromiter((mapping[value] for value in text), dtype=np.int32, count=len(text))
    return encoded, names


def save_label_sidecars(root: str | Path, labels: np.ndarray | None, names: Sequence[str], source: str | None) -> None:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    if labels is not None:
        np.save(output_root / "labels.npy", np.asarray(labels, dtype=np.int32))
    write_json(
        output_root / "label_metadata.json",
        {"source": source, "names": list(names), "n_labels": len(names)},
    )


def load_label_sidecars(root: str | Path) -> tuple[np.ndarray | None, list[str]]:
    input_root = Path(root)
    labels_path = input_root / "labels.npy"
    metadata_path = input_root / "label_metadata.json"
    labels = np.load(labels_path) if labels_path.exists() else None
    names: list[str] = []
    if metadata_path.exists():
        import json

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        names = [str(value) for value in payload.get("names", [])]
    return labels, names


def per_label_shape_rows(
    dataset: str,
    variant: str,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    *,
    min_points: int = 30,
) -> list[dict[str, Any]]:
    if labels is None:
        return []
    rows: list[dict[str, Any]] = []
    points_all = np.asarray(embedding, dtype=np.float64)
    label_values = np.asarray(labels)
    if label_values.shape[0] != points_all.shape[0]:
        return rows
    for label in np.unique(label_values):
        points = points_all[label_values == label]
        if points.shape[0] < min_points:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(points[:, :2], rowvar=False))
        lambda2, lambda1 = float(max(eigenvalues[0], 0.0)), float(max(eigenvalues[-1], 0.0))
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "label": int(label) if np.issubdtype(np.asarray(label).dtype, np.integer) else str(label),
                "n_points": int(points.shape[0]),
                "lambda1": lambda1,
                "lambda2": lambda2,
                "linearity": lambda1 / (lambda2 + 1e-12),
                "area_proxy": float(np.sqrt(lambda1 * lambda2)),
            }
        )
    return rows


def read_diagnostic_peak(path: str | Path, columns: Mapping[str, str] | None = None) -> dict[str, float]:
    diagnostic_path = Path(path)
    if not diagnostic_path.exists() or diagnostic_path.stat().st_size == 0:
        return {}
    opener = gzip.open if diagnostic_path.name.endswith(".gz") else open
    with opener(diagnostic_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))

    column_map = dict(
        columns
        or {
            "repl_force_pre_clip_p999": "peak_repulsion_p999",
            "total_update_p999": "peak_total_update_p999",
            "total_update_pre_clip_p999": "peak_total_update_pre_clip_p999",
        }
    )
    output: dict[str, float] = {}
    for source, target in column_map.items():
        values = []
        for row in rows:
            try:
                value = float(row[source])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            output[target] = max(values)
    return output

