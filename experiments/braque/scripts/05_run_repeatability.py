#!/usr/bin/env python3
"""Run the four profiles x repeats in fresh, warmed CPU worker processes (paper Section 6). Dry run unless --run.

Each worker loads the frozen input, warms up the same embedding and HDBSCAN
configuration on a fixed 5,000-cell subset, then times one ``fit_transform`` of a
new estimator on all cells and one HDBSCAN ``fit_predict``. The profiles are
interleaved in a shuffled order within each repeat block (fixed schedule seed).
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import os
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from _common import (CASE, SRC, PROTOCOL, PAPER_INPUT_SHA256, PROFILES, LIMITATIONS, completed, digest,
                         environment, load_experiment, now, read_csv, read_json,
                         run_root, safe_name, seal, sha, source_record, write_csv, write_json)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="New batch name, e.g. l2_v1")
    p.add_argument("--sample", default="L2")
    p.add_argument("--input-dir", type=Path, help="Frozen LNS directory; default data/processed/L2/lns")
    p.add_argument("--cells-file", type=Path, help="Ordered cell table; default data/processed/L2/cells.csv.gz")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule-seed", type=int, default=20260918)
    p.add_argument("--warmup-cells", type=int, default=5000)
    p.add_argument("--max-cells", type=int, help="Smoke test only: fixed subset, prominently marked in outputs")
    p.add_argument("--resume", action="store_true", help="Skip verified successes; retry failures in NEW attempt folders")
    p.add_argument("--run", action="store_true")
    p.add_argument("--worker-profile", choices=PROFILES, help=argparse.SUPPRESS)
    p.add_argument("--worker-repeat", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker-attempt", type=Path, help=argparse.SUPPRESS)
    return p.parse_args()


def _relative(path):
    try:
        return str(Path(path).resolve().relative_to(CASE.parents[1]))
    except ValueError:
        return Path(path).name


def estimator(profile, config):
    common = dict(config["umap"])
    seeded = profile.endswith("_seeded")
    common.update(random_state=config["seed"] if seeded else None,
                  n_jobs=1 if seeded and profile.startswith("umap_") else config["threads"])
    if profile.startswith("umap_"):
        import umap
        return umap.UMAP(**common)
    from ibumap import IBUMAP
    return IBUMAP(algorithm="ibumap", device="cpu", deterministic=seeded, **common)


def worker(args):
    import numpy as np
    import hdbscan
    import numba
    from threadpoolctl import threadpool_info
    sys.path.insert(0, str(SRC))
    root = run_root(args.run_id)
    exp = load_experiment(root)
    cfg = exp["contract"]["config"]
    dest = args.worker_attempt.resolve()
    if not dest.is_relative_to((root / "runs").resolve()):
        raise ValueError("Invalid worker destination")
    record = {"status": "running", "created_at": now(), "profile": args.worker_profile,
              "repeat": args.worker_repeat, "contract_sha256": exp["contract_sha256"], "pid": os.getpid()}
    write_json(dest / "run.json", record)
    try:
        X = np.load(root / "inputs" / "umap_input.npy", allow_pickle=False)
        indices = np.load(root / "inputs" / "warmup_indices.npy", allow_pickle=False)
        # Import, initialization of libraries, input loading and checksum work precede all timers.
        warm_model = estimator(args.worker_profile, cfg)
        start = time.perf_counter()
        warm_y = warm_model.fit_transform(np.ascontiguousarray(X[indices]))
        warm_embedding_s = time.perf_counter() - start
        warm_cluster = hdbscan.HDBSCAN(**cfg["hdbscan"])
        start = time.perf_counter()
        warm_cluster.fit_predict(warm_y)
        warm_clustering_s = time.perf_counter() - start
        del warm_model, warm_cluster, warm_y
        gc.collect()
        # A new estimator on the complete selected input rebuilds graph, spectral init and layout.
        model = estimator(args.worker_profile, cfg)
        wall, cpu = time.perf_counter(), time.process_time()
        raw_y = model.fit_transform(X)
        embedding_s, embedding_cpu_s = time.perf_counter() - wall, time.process_time() - cpu
        Y = np.asarray(raw_y, dtype=np.float32, order="C")
        if Y.shape != (len(X), 2) or not np.isfinite(Y).all():
            raise ValueError("Nonfinite or malformed embedding")
        clusterer = hdbscan.HDBSCAN(**cfg["hdbscan"])
        gc.collect()
        wall, cpu = time.perf_counter(), time.process_time()
        raw_labels = clusterer.fit_predict(Y)
        hdbscan_s, hdbscan_cpu_s = time.perf_counter() - wall, time.process_time() - cpu
        labels = np.asarray(raw_labels, dtype=np.int32)
        if labels.shape != (len(X),) or (labels < -1).any():
            raise ValueError("Malformed HDBSCAN labels")
        # Saving, conversions, checksums, diagnostics and GC are outside measured fit calls.
        np.save(dest / "embedding.npy", Y, allow_pickle=False)
        np.save(dest / "labels.npy", labels, allow_pickle=False)
        effective = (model.explain_effective_config() if args.worker_profile.startswith("ibumap_")
                     else {"engine": "umap-learn public estimator", "parameters": model.get_params()})
        write_json(dest / "effective_config.json", effective)
        internal = model.get_time_costs() if args.worker_profile.startswith("ibumap_") else None
        write_json(dest / "internal_timings.json", internal)
        sizes, counts = np.unique(labels, return_counts=True)
        write_csv(dest / "cluster_sizes.csv", [{"label": int(k), "count": int(v), "fraction": float(v / len(X))}
                                               for k, v in zip(sizes, counts)])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record.update(status="completed", finished_at=now(),
                      timing_seconds={"embedding_fit": embedding_s, "embedding_fit_cpu": embedding_cpu_s,
                                      "hdbscan_fit": hdbscan_s, "hdbscan_fit_cpu": hdbscan_cpu_s,
                                      "embedding_plus_hdbscan": embedding_s + hdbscan_s},
                      warmup={"cell_count": len(indices), "embedding_seconds": warm_embedding_s,
                              "hdbscan_seconds": warm_clustering_s, "excluded_from_timing": True},
                      cluster_count=int(np.sum(sizes >= 0)), noise_count=int(np.sum(labels == -1)),
                      noise_fraction=float(np.mean(labels == -1)), cell_count=len(X),
                      embedding_sha256=sha(dest / "embedding.npy"), labels_sha256=sha(dest / "labels.npy"),
                      peak_process_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
                      peak_rss_scope="entire worker including imports and warmup, not isolated fitting peak",
                      numba_threads=numba.get_num_threads(), numba_threading_layer=numba.threading_layer(),
                      native_threadpools=threadpool_info())
        write_json(dest / "run.json", record)
        seal(dest, ["run.json", "embedding.npy", "labels.npy", "effective_config.json", "internal_timings.json", "cluster_sizes.csv"])
        return 0
    except Exception as exc:
        record.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(dest / "run.json", record)
        traceback.print_exc()
        return 1


def parent(args):
    import numpy as np
    safe_name(args.sample)
    if args.repeats < 2 or args.threads < 1 or args.warmup_cells < 64:
        raise ValueError("Need repeats >=2, threads >=1 and warmup-cells >=64")
    if not 0 <= args.seed < 2**31 or not 0 <= args.schedule_seed < 2**32:
        raise ValueError("Seed outside supported range")
    root = run_root(args.run_id)
    input_dir = (args.input_dir or CASE / "data" / "processed" / args.sample / "lns").resolve()
    cells_file = (args.cells_file or CASE / "data" / "processed" / args.sample / "cells.csv.gz").resolve()
    matrix_file = input_dir / "umap_input.npy"
    X = np.load(matrix_file, allow_pickle=False)
    if X.ndim != 2 or len(X) < 64 or not np.isfinite(X).all():
        raise ValueError("Need a finite matrix with at least 64 cells")
    # Verify the frozen input against its original ledger, without requiring unused LNS files.
    ledger = dict(line.split("  ", 1)[::-1] for line in (input_dir / "SHA256SUMS").read_text().splitlines())
    matrix_sha = sha(matrix_file)
    if ledger.get("umap_input.npy") != matrix_sha:
        raise ValueError("Frozen UMAP matrix does not match its source SHA256SUMS")
    cells = read_csv(cells_file)
    if len(cells) != len(X) or len({c["cell_id"] for c in cells}) != len(X):
        raise ValueError("Cell table length/unique cell IDs do not match matrix")
    if any(int(c["source_row"]) != i for i, c in enumerate(cells)):
        raise ValueError("Expected cells in canonical source_row order 0..N-1")
    if args.max_cells is not None and not 64 <= args.max_cells <= len(X):
        raise ValueError("max-cells must be in [64, full cell count]")
    subset = args.max_cells is not None and args.max_cells < len(X)
    indices = (np.sort(np.random.default_rng(20260918).choice(len(X), args.max_cells, replace=False))
               if subset else np.arange(len(X)))
    X = np.asarray(X[indices], dtype=np.float32, order="C")
    minimum = max(int(len(X) * .00005), 10)
    config = {"sample": args.sample, "repeats": args.repeats, "threads": args.threads,
              "seed": args.seed, "schedule_seed": args.schedule_seed, "warmup_cells": min(args.warmup_cells, len(X)),
              "max_cells": args.max_cells, "subset_seed": 20260918, "smoke_subset": subset,
              "umap": {"n_components": 2, "n_neighbors": 50, "min_dist": 0.0, "spread": 1.0,
                       "n_epochs": 200, "metric": "euclidean", "init": "spectral", "learning_rate": 1.0,
                       "repulsion_strength": 1.0, "negative_sample_rate": 5, "local_connectivity": 1.0,
                       "set_op_mix_ratio": 1.0, "low_memory": True, "verbose": False},
              "hdbscan": {"min_cluster_size": minimum, "min_samples": minimum, "metric": "euclidean",
                          "cluster_selection_epsilon": .1, "cluster_selection_method": "eom", "core_dist_n_jobs": 1,
                          "algorithm": "best", "approx_min_span_tree": True}}
    env, source = environment(), source_record()
    if any(v is None for v in env["packages"].values()):
        raise RuntimeError(f"Missing required packages: {[k for k,v in env['packages'].items() if v is None]}")
    contract = {"config": config, "source_matrix_sha256": matrix_sha, "source_cells_sha256": sha(cells_file),
                "input_shape": list(X.shape), "environment": env, "runner_and_ibumap_source_sha256": source["sha256"]}
    print(f"{len(X):,} cells × {X.shape[1]} features; {4 * args.repeats} fresh workers; {args.threads} configured CPU threads")
    print("Input matrix matches the paper's frozen input" if matrix_sha == PAPER_INPUT_SHA256 else
          f"NOTE: input matrix SHA-256 {matrix_sha} differs from the paper's {PAPER_INPUT_SHA256}")
    print(f"Output: {root}")
    if subset:
        print("SMOKE SUBSET — not a full L2 paper experiment")
    if root.exists():
        if not args.resume:
            raise FileExistsError("Batch exists. Use the same options with --resume, or a new --run-id")
        old = load_experiment(root)
        if old["contract"] != contract:
            raise ValueError("Resume rejected: configuration, inputs, software, host or runner/source changed; use a new run ID")
    elif args.resume:
        raise FileNotFoundError("Cannot resume a nonexistent batch")
    if not args.run:
        print("Dry run: prerequisites validated; nothing written. Add --run to execute.")
        return 0
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:
            inputs = root / "inputs"
            inputs.mkdir()
            np.save(inputs / "umap_input.npy", X, allow_pickle=False)
            np.save(inputs / "source_row_indices.npy", indices, allow_pickle=False)
            warm_indices = np.sort(np.random.default_rng(20260919).choice(len(X), config["warmup_cells"], replace=False))
            np.save(inputs / "warmup_indices.npy", warm_indices, allow_pickle=False)
            write_csv(inputs / "cells.csv", [{"cell_id": cells[i]["cell_id"], "source_row": int(i)} for i in indices])
            names = ["umap_input.npy", "source_row_indices.npy", "warmup_indices.npy", "cells.csv"]
            for name in ("metadata.json", "selected_features.csv.gz"):
                if (input_dir / name).exists():
                    if ledger.get(name) != sha(input_dir / name):
                        raise ValueError(f"Source metadata checksum mismatch: {name}")
                    shutil.copy2(input_dir / name, inputs / ("source_" + name))
                    names.append("source_" + name)
            seal(inputs, names)
            rng, schedule = random.Random(args.schedule_seed), []
            for repeat in range(args.repeats):
                block = list(PROFILES)
                rng.shuffle(block)
                schedule.extend({"profile": p, "repeat": repeat} for p in block)
            exp = {"protocol": PROTOCOL, "created_at": now(), "contract": contract, "contract_sha256": digest(contract),
                   "schedule": schedule, "source": source, "source_paths": {"matrix": _relative(matrix_file), "cells": _relative(cells_file)},
                   "limitations": LIMITATIONS,
                   "timing_scope": "warm fit_transform and fit_predict calls only; independent fresh estimators; all measured repeats retained",
                   "input_sha256s": {name: sha(inputs / name) for name in names}}
            write_json(root / "experiment.json", exp)
        else:
            exp = old
        env_child = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env_child[key] = str(args.threads)
        env_child.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(root / "cache" / "numba"))
        (root / "cache" / "numba").mkdir(parents=True, exist_ok=True)
        failed = 0
        for ordinal, item in enumerate(exp["schedule"], 1):
            profile, repeat = item["profile"], item["repeat"]
            if completed(root, profile, repeat, exp["contract_sha256"]):
                print(f"[{ordinal}/{len(exp['schedule'])}] verified; skip {profile} repeat {repeat}", flush=True)
                continue
            base = root / "runs" / profile / f"repeat_{repeat:03d}"
            base.mkdir(parents=True, exist_ok=True)
            attempt = max([int(p.name.split('_')[1]) for p in base.glob('attempt_*')] + [0]) + 1
            dest = base / f"attempt_{attempt:03d}"
            dest.mkdir()
            print(f"[{ordinal}/{len(exp['schedule'])}] {profile} repeat {repeat}, attempt {attempt}", flush=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-id", args.run_id,
                   "--worker-profile", profile, "--worker-repeat", str(repeat), "--worker-attempt", str(dest)]
            with (dest / "stdout.log").open("w") as out, (dest / "stderr.log").open("w") as err:
                result = subprocess.run(cmd, env=env_child, stdout=out, stderr=err)
            if result.returncode != 0:
                failed += 1
                print(f"  FAILED; see {dest / 'stderr.log'}", flush=True)
        successes = sum(completed(root, p, r, exp["contract_sha256"]) is not None
                        for p in PROFILES for r in range(args.repeats))
        write_json(root / "status.json", {"updated_at": now(), "completed": successes, "expected": len(exp["schedule"]),
                                        "failed_attempts_this_launch": failed, "status": "completed" if successes == len(exp["schedule"]) else "incomplete"})
        print(f"Completed {successes}/{len(exp['schedule'])}. Results: {root}")
        return 0 if successes == len(exp["schedule"]) else 1


if __name__ == "__main__":
    try:
        a = parse_args()
        sys.exit(worker(a) if a.worker_profile else parent(a))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
