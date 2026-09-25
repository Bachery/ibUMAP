#!/usr/bin/env python3
"""Run and freeze BRAQUE Lognormal Shrinkage with per-marker checkpoints.

The numerical steps follow the authors' legacy implementation.  BGM fitting is
seeded, each completed marker is atomically cached, and the final raw LNS matrix
and robust-standardized UMAP input are checksummed.  Dry run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = CASE_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="L2")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-n-gaussians", type=int, default=15)
    parser.add_argument("--contraction-factor", type=float, default=5.0)
    parser.add_argument("--subsampling", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--overwrite", action="store_true", help="Discard the whole stage, including checkpoints.")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def verify_checksums(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(root / name) != expected:
            raise ValueError(f"Input checksum mismatch: {root / name}")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean[:80] or "marker"


def remove_stage_directory(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/"), Path.home().resolve(), CASE_ROOT.resolve(), (CASE_ROOT / "data").resolve()}
    if resolved in forbidden or len(resolved.parts) < 4:
        raise ValueError(f"Refusing unsafe recursive overwrite target: {resolved}")
    shutil.rmtree(resolved)


def main() -> int:
    args = parse_args()
    if args.max_n_gaussians < 2 or args.contraction_factor <= 1 or args.subsampling < 1:
        raise ValueError("Require max-n-gaussians >=2, contraction-factor >1, and subsampling >=1")
    sample_dir = PROCESSED_ROOT / args.sample
    input_dir = args.input_dir or sample_dir / "features_selected"
    output_dir = args.output_dir or sample_dir / "lns"
    config = {
        "algorithm": "BRAQUE legacy Lognormal Shrinkage",
        "max_n_gaussians": args.max_n_gaussians,
        "contraction_factor": args.contraction_factor,
        "subsampling": args.subsampling,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "covariance_type": "full",
        "n_init": 1,
    }
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(json.dumps(config, indent=2))
    if not args.run:
        print("Dry run only; pass --run to fit LNS models.")
        return 0

    try:
        import sklearn
        from sklearn.mixture import BayesianGaussianMixture
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for LNS") from exc

    verify_checksums(input_dir)
    source = np.load(input_dir / "markers_selected.npy", allow_pickle=False)
    features = pd.read_csv(input_dir / "selected_features.csv.gz")
    if source.shape[1] != len(features) or not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("Selected marker input violates shape/finite/nonnegative contract")

    if output_dir.exists() and args.overwrite:
        remove_stage_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "marker_cache"
    cache_dir.mkdir(exist_ok=True)

    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_hash = sha256_file(input_dir / "markers_selected.npy")
    means = source.astype(np.float64).mean(axis=0)
    scales = np.abs(source.astype(np.float64) - means).mean(axis=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        bad = features.loc[np.where(scales <= 0)[0], "feature_name"].tolist()
        raise ValueError(f"Zero/nonfinite robust scales: {bad}")
    scaled = source.astype(np.float64) / scales
    positive = scaled[scaled > 0]
    if positive.size == 0:
        raise ValueError("No positive marker values available for log transform")
    epsilon = float(positive.min())
    log_matrix = np.log2(scaled + epsilon)

    transformed = np.empty(source.shape, dtype=np.float32)
    marker_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, feature in enumerate(features["feature_name"].astype(str)):
        cache_path = cache_dir / f"{index:03d}_{slug(feature)}.npz"
        column_hash = sha256_array(source[:, index])
        report: dict[str, Any]
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_config = str(cached["config_hash"].item())
                cached_input = str(cached["input_hash"].item())
                cached_column = np.asarray(cached["transformed"], dtype=np.float32)
                if cached_config != config_hash or cached_input != column_hash or cached_column.shape != (len(source),):
                    raise ValueError(f"Stale or incompatible marker checkpoint: {cache_path}")
                transformed[:, index] = cached_column
                report = json.loads(str(cached["report_json"].item()))
                report["cache_status"] = "reused"
        else:
            marker_started = time.perf_counter()
            use = log_matrix[:, index]
            model = BayesianGaussianMixture(
                n_components=args.max_n_gaussians,
                max_iter=args.max_iter,
                n_init=1,
                tol=args.tol,
                random_state=args.random_state,
                covariance_type="full",
            )
            model.fit(use[:: args.subsampling, None])
            predictions = model.predict(use[:, None])
            centers = model.means_.reshape(-1)[predictions]
            contracted = centers - (centers - use) / args.contraction_factor
            values = np.exp2(contracted) - epsilon
            values -= values.min()
            column = np.asarray(values, dtype=np.float32)
            if not np.isfinite(column).all() or np.any(column < 0):
                raise ValueError(f"Invalid LNS output for marker {feature}")
            transformed[:, index] = column
            report = {
                "feature_index": index,
                "feature_name": feature,
                "cache_status": "created",
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "lower_bound": float(model.lower_bound_),
                "used_components": int(np.unique(predictions).size),
                "fit_cell_count": int(use[:: args.subsampling].size),
                "elapsed_seconds": time.perf_counter() - marker_started,
            }
            atomic_npz(
                cache_path,
                transformed=column,
                weights=np.asarray(model.weights_),
                means=np.asarray(model.means_),
                covariances=np.asarray(model.covariances_),
                config_hash=np.asarray(config_hash),
                input_hash=np.asarray(column_hash),
                report_json=np.asarray(json.dumps(report, sort_keys=True)),
            )
        marker_reports.append(report)
        print(
            f"[{index + 1:02d}/{source.shape[1]:02d}] {feature}: "
            f"{report['cache_status']}, components={report['used_components']}, "
            f"converged={report['converged']}",
            flush=True,
        )

    transformed64 = transformed.astype(np.float64)
    lns_means = transformed64.mean(axis=0)
    lns_scales = np.abs(transformed64 - lns_means).mean(axis=0)
    if np.any(lns_scales <= 0):
        raise ValueError("LNS produced a constant feature")
    umap_input = np.asarray(
        (transformed64 - np.median(transformed64, axis=0)) / lns_scales,
        dtype=np.float32,
        order="C",
    )
    if not np.isfinite(umap_input).all():
        raise ValueError("Robust-standardized UMAP input contains NaN/Inf")

    atomic_npy(output_dir / "markers_lns.npy", transformed)
    atomic_npy(output_dir / "umap_input.npy", umap_input)
    shutil.copy2(input_dir / "selected_features.csv.gz", output_dir / "selected_features.csv.gz")
    metadata = {
        "created_at": now_utc(),
        "sample": args.sample,
        "fidelity": "legacy equations with published seed; public-workbook proxy feature set",
        "config": config,
        "config_hash": config_hash,
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "input": {"directory": str(input_dir), "markers_selected_sha256": source_hash},
        "shape": list(source.shape),
        "dtype": "float32",
        "global_log_epsilon": epsilon,
        "run_elapsed_seconds": time.perf_counter() - started,
        "model_fit_elapsed_seconds": float(sum(record["elapsed_seconds"] for record in marker_reports)),
        "checkpoint_counts": {
            "created": int(sum(record["cache_status"] == "created" for record in marker_reports)),
            "reused": int(sum(record["cache_status"] == "reused" for record in marker_reports)),
        },
        "all_models_converged": all(record["converged"] for record in marker_reports),
        "marker_reports": marker_reports,
    }
    atomic_json(output_dir / "metadata.json", metadata)
    files = ["markers_lns.npy", "umap_input.npy", "selected_features.csv.gz", "metadata.json"]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in files), encoding="utf-8"
    )
    print(f"Frozen LNS shape: {transformed.shape}; all converged: {metadata['all_models_converged']}")
    print(
        f"Run elapsed: {metadata['run_elapsed_seconds']:.1f} seconds; "
        f"recorded model-fit time: {metadata['model_fit_elapsed_seconds']:.1f} seconds"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
