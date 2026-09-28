#!/usr/bin/env python3
"""Serve a live BRAQUE demo backed by umap-learn / ibUMAP and HDBSCAN (not part of the paper's results).

The browser front end in ../demo_live/ requests a fresh embedding and clustering
of a fixed subset of the frozen BRAQUE input (01-04 must have run). A single
background job is allowed at a time so a local presentation cannot start several
CPU-heavy runs. Each run can be compared with an earlier run of the same server:
clusters are matched by maximum overlap and the share of cells whose matched
cluster differs is reported, as in 06_evaluate_repeatability.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from _common import label_metrics, match_labels


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
LIVE_DEMO_ROOT = CASE_ROOT / "demo_live"
PROCESSED_ROOT = CASE_ROOT / "data" / "processed"
IBUMAP_SRC = REPO_ROOT / "src"

ALGORITHMS = {
    "ibumap": {
        "label": "ibUMAP",
        "description": "ibUMAP's interpolation-based FFT optimizer on CPU.",
    },
    "umap_learn": {
        "label": "umap-learn",
        "description": "The public umap-learn estimator on CPU.",
    },
}
CELL_OPTIONS = (1000, 2500, 5000, 10000, 20000)
DEFAULT_CELL_COUNT = 2500
DEFAULT_SEED = 42
SUBSET_SEED = 20260820
N_EPOCHS = 200
HDBSCAN_EPSILON = 0.1
MAX_REQUEST_BYTES = 64 * 1024
JOB_ID_PATTERN = re.compile(r"[0-9a-f]{12}")


def case_study_min_cluster_size(cell_count: int) -> int:
    """HDBSCAN min_cluster_size (= min_samples) rule of the case study."""
    return max(int(cell_count * 0.00005), 10)


# Tunable parameters. Defaults are the case-study values; the HDBSCAN default is
# the case-study rule, which gives 10 for every offered subset size.
PARAMETERS = {
    "n_neighbors": {"default": 50, "min": 2, "max": 200, "step": 1, "type": "int"},
    "min_dist": {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01, "type": "float"},
    "min_cluster_size": {
        "default": case_study_min_cluster_size(DEFAULT_CELL_COUNT),
        "min": 2,
        "max": 500,
        "step": 1,
        "type": "int",
    },
}
COMPARISON_METHOD = (
    "Clusters of the two runs are matched one-to-one by maximum overlap on the shared cells "
    "(Hungarian assignment; noise stays noise). The share counts cells whose matched cluster "
    "differs, including changes to or from noise, as in 06_evaluate_repeatability.py. It "
    "includes cluster splits and merges and is not a misclassification rate."
)


class BusyError(RuntimeError):
    """Raised when the one-job execution slot is occupied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="L2")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--check", action="store_true", help="validate dependencies/data and exit")
    parser.add_argument("--smoke-test", action="store_true", help="run one small job and exit")
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="ibumap")
    parser.add_argument("--cells", type=int, choices=CELL_OPTIONS, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-neighbors", type=int, default=PARAMETERS["n_neighbors"]["default"])
    parser.add_argument("--min-dist", type=float, default=PARAMETERS["min_dist"]["default"])
    parser.add_argument(
        "--min-cluster-size", type=int, default=PARAMETERS["min_cluster_size"]["default"],
        help="HDBSCAN min_cluster_size; min_samples is set to the same value",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="smoke test only: run this many times, comparing each run with the previous one",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path) -> None:
    ledger = root / "SHA256SUMS"
    if not ledger.is_file():
        raise FileNotFoundError(f"Missing checksum ledger: {ledger}")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        if sha256_file(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def quantize_coordinates(embedding: Any) -> list[int]:
    """Quantize to uint16 with one scale for both axes, so the aspect ratio is kept."""
    import numpy as np

    array = np.asarray(embedding, dtype=np.float64)
    minimum = array.min(axis=0)
    span = float(np.max(array.max(axis=0) - minimum))
    if span <= np.finfo(np.float64).eps:
        span = 1.0
    normalized = np.clip((array - minimum) / span, 0.0, 1.0)
    return np.rint(normalized * 65535).astype(np.uint16).ravel().astype(int).tolist()


def parse_parameter(request: dict[str, Any], name: str) -> int | float:
    spec = PARAMETERS[name]
    raw = request.get(name, spec["default"])
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a number")
    try:
        value: int | float = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if spec["type"] == "int":
        if value != int(value):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{name} must be between {spec['min']} and {spec['max']}")
    return value


def compare_with_previous(
    indices: Any,
    labels: Any,
    previous: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Compare labels with an earlier run on the cells both runs contain.

    Returns the comparison record and display labels: this run's labels renamed to
    the matched cluster IDs of the earlier run's display labels, so matched
    clusters keep their colour. Clusters without a match get IDs that the earlier
    run does not use. Renaming does not change the partition.
    """
    import numpy as np

    _, current_pos, previous_pos = np.intersect1d(
        indices, previous["indices"], assume_unique=True, return_indices=True
    )
    base = {
        "available": True,
        "previous_job_id": previous["job_id"],
        "current_cell_count": int(len(indices)),
        "previous_cell_count": int(len(previous["indices"])),
        "shared_cell_count": int(len(current_pos)),
        "method": COMPARISON_METHOD,
    }
    if len(current_pos) == 0:
        return {**base, "available": False, "reason": "The two runs share no cells."}, labels.copy()

    reference = np.asarray(previous["display_labels"])[previous_pos].astype(np.int64)
    candidate = labels[current_pos].astype(np.int64)
    metrics = label_metrics(reference, candidate)
    _, mapping = match_labels(reference, candidate)

    reference_ids = reference[reference >= 0]
    reference_max = int(reference_ids.max()) if reference_ids.size else -1
    next_id = max(int(np.max(previous["display_labels"])), reference_max) + 1
    full_mapping = {-1: -1}
    for label in np.unique(labels[labels >= 0]):
        matched = mapping.get(int(label))
        if matched is None or matched > reference_max:  # no match on the shared cells
            full_mapping[int(label)] = next_id
            next_id += 1
        else:
            full_mapping[int(label)] = matched
    lookup = np.vectorize(full_mapping.__getitem__, otypes=[np.int64])
    display_labels = lookup(labels) if len(labels) else labels.astype(np.int64)

    changed = reference != display_labels[current_pos]
    changed_positions = np.sort(current_pos[changed])
    comparison = {
        **base,
        "assignment_disagreement": metrics["assignment_disagreement"],
        "changed_count": int(changed.sum()),
        "cluster_to_cluster_disagreement": metrics["cluster_to_cluster_disagreement"],
        "noise_status_disagreement": metrics["noise_status_disagreement"],
        "cluster_to_noise_count": metrics["cluster_to_noise_count"],
        "noise_to_cluster_count": metrics["noise_to_cluster_count"],
        "ari_including_noise": metrics["ari_including_noise"],
        "ami_including_noise": metrics["ami_including_noise"],
        "changed_indices": changed_positions.astype(int).tolist(),
    }
    return comparison, display_labels


class LiveComputeService:
    def __init__(self, sample: str, threads: int) -> None:
        import numpy as np
        import pandas as pd

        if threads < 1:
            raise ValueError("--threads must be positive")
        if not LIVE_DEMO_ROOT.is_dir():
            raise FileNotFoundError(f"Missing live frontend: {LIVE_DEMO_ROOT}")
        input_dir = PROCESSED_ROOT / sample / "lns"
        verify_checksums(input_dir)
        matrix_path = input_dir / "umap_input.npy"
        cells_path = PROCESSED_ROOT / sample / "cells.csv.gz"
        self.matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        cells = pd.read_csv(cells_path, usecols=["cell_id"])
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(cells):
            raise ValueError(
                f"Matrix/cell mismatch: matrix={self.matrix.shape}, cells={len(cells)}"
            )
        if not np.isfinite(self.matrix).all():
            raise ValueError("UMAP input contains non-finite values")
        if max(CELL_OPTIONS) > len(cells):
            raise ValueError("Configured live-demo cell count exceeds the prepared dataset")
        self.sample = sample
        self.threads = threads
        self.cell_ids = cells["cell_id"].astype(str).to_numpy()
        self.input_shape = tuple(int(value) for value in self.matrix.shape)
        self.input_sha256 = sha256_file(matrix_path)
        rng = np.random.default_rng(SUBSET_SEED)
        self.permutation = rng.permutation(len(cells))

        for module_name in ("umap", "hdbscan"):
            __import__(module_name)
        if str(IBUMAP_SRC) not in sys.path:
            sys.path.insert(0, str(IBUMAP_SRC))
        from ibumap import IBUMAP  # noqa: F401

    def public_config(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "algorithms": ALGORITHMS,
            "cell_count_options": CELL_OPTIONS,
            "default_cell_count": DEFAULT_CELL_COUNT,
            "default_seed": DEFAULT_SEED,
            "full_cell_count": self.input_shape[0],
            "feature_count": self.input_shape[1],
            "threads": self.threads,
            "parameters": PARAMETERS,
            "model": {
                "n_epochs": N_EPOCHS,
                "metric": "euclidean",
                "init": "spectral",
                "hdbscan_min_samples": "equal to min_cluster_size",
                "hdbscan_cluster_selection_epsilon": HDBSCAN_EPSILON,
                "hdbscan_cluster_selection_method": "eom",
            },
            "comparison_method": COMPARISON_METHOD,
            "software": {
                "umap_learn": package_version("umap-learn"),
                "hdbscan": package_version("hdbscan"),
                "ibumap": package_version("ibumap"),
            },
            "input": {
                "sha256": self.input_sha256,
                "subset_seed": SUBSET_SEED,
                "note": "The fixed cell subset is independent of the algorithm seed.",
            },
        }

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        algorithm = str(request.get("algorithm", ""))
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        deterministic = request.get("deterministic", False)
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be true or false")
        try:
            cell_count = int(request.get("cell_count", DEFAULT_CELL_COUNT))
        except (TypeError, ValueError) as exc:
            raise ValueError("cell_count must be an integer") from exc
        if cell_count not in CELL_OPTIONS:
            raise ValueError(f"cell_count must be one of {CELL_OPTIONS}")
        seed: int | None = None
        if deterministic:
            try:
                seed = int(request.get("seed", DEFAULT_SEED))
            except (TypeError, ValueError) as exc:
                raise ValueError("seed must be an integer") from exc
            if not 0 <= seed <= 2**31 - 1:
                raise ValueError("seed must be between 0 and 2147483647")
        compare_to = request.get("compare_to")
        if compare_to is not None and (
            not isinstance(compare_to, str) or not JOB_ID_PATTERN.fullmatch(compare_to)
        ):
            raise ValueError("compare_to must be a job id")
        return {
            "algorithm": algorithm,
            "deterministic": deterministic,
            "seed": seed,
            "cell_count": cell_count,
            "n_neighbors": parse_parameter(request, "n_neighbors"),
            "min_dist": parse_parameter(request, "min_dist"),
            "min_cluster_size": parse_parameter(request, "min_cluster_size"),
            "compare_to": compare_to,
        }

    def run(
        self,
        raw_request: dict[str, Any],
        progress: Callable[[str], None] | None = None,
        previous: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one job; return the public result and the arrays kept for later comparisons."""
        import numpy as np

        request = self.validate_request(raw_request)
        report = progress or (lambda _phase: None)
        total_started = time.perf_counter()
        report("preparing_input")
        indices = np.sort(self.permutation[: request["cell_count"]])
        matrix = np.asarray(self.matrix[indices], dtype=np.float32, order="C")
        cell_ids = self.cell_ids[indices].tolist()
        n_neighbors = min(request["n_neighbors"], len(matrix) - 1)
        min_dist = request["min_dist"]

        report("embedding")
        if request["algorithm"] == "umap_learn":
            import umap

            n_jobs = 1 if request["deterministic"] else self.threads
            estimator = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                n_epochs=N_EPOCHS,
                min_dist=min_dist,
                metric="euclidean",
                init="spectral",
                low_memory=True,
                random_state=request["seed"],
                n_jobs=n_jobs,
            )
            reduction_started = time.perf_counter()
            embedding = estimator.fit_transform(matrix)
            reduction_seconds = time.perf_counter() - reduction_started
            stage_timings: dict[str, Any] = {"end_to_end": reduction_seconds}
            engine = "umap-learn public estimator"
        else:
            from ibumap import IBUMAP

            n_jobs = self.threads
            estimator = IBUMAP(
                algorithm="ibumap",
                device="cpu",
                n_neighbors=n_neighbors,
                n_components=2,
                n_epochs=N_EPOCHS,
                min_dist=min_dist,
                metric="euclidean",
                init="spectral",
                low_memory=True,
                random_state=request["seed"],
                deterministic=request["deterministic"],
                n_jobs=n_jobs,
            )
            reduction_started = time.perf_counter()
            embedding = estimator.fit_transform(matrix)
            reduction_seconds = time.perf_counter() - reduction_started
            stage_timings = json_value(estimator.get_time_costs())
            engine = "ibUMAP end-to-end"

        embedding = np.asarray(embedding, dtype=np.float32, order="C")
        if embedding.shape != (len(matrix), 2) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding result: {embedding.shape}")

        report("hdbscan")
        import hdbscan

        min_cluster_size = request["min_cluster_size"]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            cluster_selection_epsilon=HDBSCAN_EPSILON,
            cluster_selection_method="eom",
            core_dist_n_jobs=1,
        )
        hdbscan_started = time.perf_counter()
        labels = np.asarray(clusterer.fit_predict(embedding), dtype=np.int32)
        hdbscan_seconds = time.perf_counter() - hdbscan_started
        result_fingerprint = hashlib.sha256(
            embedding.tobytes(order="C") + labels.tobytes(order="C")
        ).hexdigest()

        comparison: dict[str, Any] | None = None
        display_labels = labels.astype(np.int64)
        if request["compare_to"] is not None:
            report("comparing")
            if previous is None:
                comparison = {
                    "available": False,
                    "previous_job_id": request["compare_to"],
                    "reason": "The previous run is no longer held by the server (for example after a restart).",
                }
            else:
                comparison, display_labels = compare_with_previous(indices, labels, previous)

        unique, counts = np.unique(labels, return_counts=True)
        display_unique, display_counts = np.unique(display_labels, return_counts=True)
        cluster_sizes = sorted(
            (
                {"label": int(label), "count": int(count)}
                for label, count in zip(display_unique, display_counts)
            ),
            key=lambda item: item["count"],
            reverse=True,
        )
        total_seconds = time.perf_counter() - total_started
        kept = {"indices": indices, "display_labels": display_labels}
        return {
            "schema_version": 2,
            "created_at": now_utc(),
            "sample": self.sample,
            "request": request,
            "input": {
                "cell_count": len(matrix),
                "feature_count": matrix.shape[1],
                "subset_seed": SUBSET_SEED,
            },
            "effective": {
                "engine": engine,
                "random_state": request["seed"],
                "deterministic": request["deterministic"],
                "n_jobs": n_jobs,
                "n_neighbors": n_neighbors,
                "n_epochs": N_EPOCHS,
                "min_dist": min_dist,
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_cluster_size,
                "hdbscan_cluster_selection_epsilon": HDBSCAN_EPSILON,
            },
            "timing_seconds": {
                "reduction": reduction_seconds,
                "hdbscan": hdbscan_seconds,
                "total": total_seconds,
                "reduction_stages": stage_timings,
            },
            "summary": {
                "cluster_count_excluding_noise": int(np.sum(unique >= 0)),
                "noise_count": int(np.sum(labels == -1)),
                "noise_fraction": float(np.mean(labels == -1)),
                "result_fingerprint": result_fingerprint,
            },
            "comparison": comparison,
            "cell_ids": cell_ids,
            "coordinates_uint16": quantize_coordinates(embedding),
            "labels": labels.astype(int).tolist(),
            "display_labels": display_labels.astype(int).tolist(),
            "cluster_sizes": cluster_sizes,
        }, kept


class JobManager:
    def __init__(self, service: LiveComputeService) -> None:
        self.service = service
        self.jobs: dict[str, dict[str, Any]] = {}
        # Labels and cell indices of completed jobs, kept server-side for comparisons.
        self.kept: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.execution_lock = threading.Lock()

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        validated = self.service.validate_request(request)
        if not self.execution_lock.acquire(blocking=False):
            raise BusyError("A live computation is already running")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "created_at": now_utc(),
            "request": validated,
        }
        with self.lock:
            self._prune()
            self.jobs[job_id] = job
            previous = self.kept.get(validated["compare_to"]) if validated["compare_to"] else None
        thread = threading.Thread(
            target=self._execute,
            args=(job_id, validated, previous),
            name=f"braque-live-{job_id}",
            daemon=True,
        )
        thread.start()
        return self.get(job_id)

    def _prune(self) -> None:
        completed = [
            key for key, value in self.jobs.items()
            if value["status"] in {"completed", "failed"}
        ]
        for key in completed[:-7]:
            self.jobs.pop(key, None)
            self.kept.pop(key, None)

    def _update(self, job_id: str, **values: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(values)
            self.jobs[job_id]["updated_at"] = now_utc()

    def _execute(
        self,
        job_id: str,
        request: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> None:
        started = time.perf_counter()
        self._update(job_id, status="running", phase="preparing_input", started_at=now_utc())
        try:
            result, kept = self.service.run(
                request,
                progress=lambda phase: self._update(job_id, phase=phase),
                previous=previous,
            )
            with self.lock:
                self.kept[job_id] = {"job_id": job_id, **kept}
            self._update(
                job_id,
                status="completed",
                phase="completed",
                finished_at=now_utc(),
                server_elapsed_seconds=time.perf_counter() - started,
                result=result,
            )
        except Exception as exc:  # Keep server alive and expose a concise error.
            traceback.print_exc()
            self._update(
                job_id,
                status="failed",
                phase="failed",
                finished_at=now_utc(),
                server_elapsed_seconds=time.perf_counter() - started,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        finally:
            self.execution_lock.release()

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])


class LiveDemoHandler(SimpleHTTPRequestHandler):
    service: LiveComputeService
    jobs: JobManager

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(json_value(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(HTTPStatus.OK, {"status": "ok", "sample": self.service.sample})
            return
        if path == "/api/config":
            self.send_json(HTTPStatus.OK, self.service.public_config())
            return
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            try:
                self.send_json(HTTPStatus.OK, self.jobs.get(job_id))
            except KeyError:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown job"})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/run":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Invalid request body size")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object")
            job = self.jobs.submit(request)
            self.send_json(
                HTTPStatus.ACCEPTED,
                {
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "status_url": f"/api/jobs/{job['job_id']}",
                },
            )
        except BusyError as exc:
            self.send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    cache_dir = PROCESSED_ROOT / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.threads))

    service = LiveComputeService(args.sample, args.threads)
    if args.check:
        print(json.dumps({"verification": "passed", **service.public_config()}, indent=2))
        return 0
    if args.smoke_test:
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        previous = None
        for repeat in range(args.repeats):
            job_id = f"{repeat:012x}"
            result, kept = service.run(
                {
                    "algorithm": args.algorithm,
                    "deterministic": args.deterministic,
                    "seed": args.seed,
                    "cell_count": args.cells,
                    "n_neighbors": args.n_neighbors,
                    "min_dist": args.min_dist,
                    "min_cluster_size": args.min_cluster_size,
                    "compare_to": previous["job_id"] if previous else None,
                },
                previous=previous,
            )
            previous = {"job_id": job_id, **kept}
            comparison = result["comparison"]
            if comparison is not None:
                comparison = {
                    key: value for key, value in comparison.items()
                    if key not in {"changed_indices", "method"}
                }
            print(
                json.dumps(
                    {
                        "smoke_test": "passed",
                        "repeat": repeat,
                        "request": result["request"],
                        "timing_seconds": {
                            key: result["timing_seconds"][key]
                            for key in ("reduction", "hdbscan", "total")
                        },
                        "summary": result["summary"],
                        "comparison_with_previous_repeat": comparison,
                    },
                    indent=2,
                ),
                flush=True,
            )
        return 0

    manager = JobManager(service)

    class BoundHandler(LiveDemoHandler):
        pass

    BoundHandler.service = service
    BoundHandler.jobs = manager
    handler = partial(BoundHandler, directory=str(LIVE_DEMO_ROOT))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    server.daemon_threads = True
    print(f"Serving live BRAQUE demo at http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
