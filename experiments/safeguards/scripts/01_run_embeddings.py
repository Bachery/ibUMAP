#!/usr/bin/env python3
"""Run production ibUMAP and the two single-safeguard ablations on the fixed inputs.

For every (dataset, variant, seed) this writes, under results/,
  embeddings/<dataset>/<variant>__seed_<seed>.npy   final embedding (float32)
  metadata/<dataset>/<variant>__seed_<seed>.json    parameters, input digests, embedding hash, environment
  traces/<dataset>/<variant>__seed_<seed>.csv       one row per update (see below)
  snapshots/<dataset>/<variant>__seed_<plot_seed>/  states after the configured numbers of updates

The trace is recorded by a read-only wrapper around the CPU ibFFT call
(``ibumap.optimizers.ibumap_optimizer.ibFFT_repulsive_sampling``, the same pattern
as the mechanism experiment's snapshot hook). Before each field evaluation it
copies the embedding and the attraction that the fused update is about to apply;
it never changes the computation (tests/test_safeguard_cases.py checks this bit for
bit). Per update t (0-based) a row holds

  before_grid_*        the ibFFT mesh built for the state before the update (same rule
                       as kernels/cpu/ibfft.py) and before_core_boxes, the number of
                       mesh boxes spanned by the longer side of the 99% core
  step_norm_*          norms of the applied update y(t+1) - y(t)
  attr_norm_*          norms of the applied (clipped, damped) attraction
  repl_norm_*          norms of the applied repulsion (update minus attraction)
  after_*              bounding-box statistics of y(t+1)

Library diagnostics are not used: they switch off the fused execution path, and
the result is then no longer bitwise equal to production.

Usage (from this directory; the fixed inputs must exist, see README):
  python scripts/01_run_embeddings.py                    # plan only
  python scripts/01_run_embeddings.py --run [--dataset cifar10] [--variant full] [--seed 42] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

from _common import (PACKAGES, Layout, bbox_metrics, core_mask, fixed_input_dir, input_digests, load_config,
                     load_fixed_inputs, repo_relative, sha256_array, thread_count, write_csv, write_json)

CFG = load_config()
THREADS = thread_count(CFG)
os.environ.setdefault("NUMBA_NUM_THREADS", str(THREADS))
os.environ.setdefault("IBUMAP_FFT_THREADS", str(THREADS))

import numba  # noqa: E402
import numpy as np  # noqa: E402

from common.gpu_runtime import machine_info  # noqa: E402


def _norms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", values, values))


def grid_stats(y: np.ndarray, core_fraction: float) -> dict:
    """Mesh the CPU ibFFT builds for state ``y`` (kernels/cpu/ibfft.py, production settings).

    The square mesh spans [min(Y), max(Y)] over both coordinates. With
    intervals_per_integer = 1, min_num_intervals = 100 and box scale 1 the box count
    per dimension is min(sqrt(16N), max(sqrt(4N/log N), max(100, span))), raised to the
    next supported FFT size; the box width is then rounded up onto the logarithmic grid.
    """
    from ibumap.kernels.cpu.ibfft import _ALLOWED_N_BOXES_PER_DIM as allowed
    from ibumap.kernels.fft_grid import quantize_fft_box_width

    n = len(y)
    span = float(y.max() - y.min())
    estimate = int(min(np.sqrt(16 * n), max(np.sqrt(4 * n / np.log(n)), max(100.0, span))))
    boxes = int(allowed[-1]) if estimate >= allowed[-1] else int(allowed[allowed > estimate][0])
    _, width, _ = quantize_fft_box_width(span, boxes, np.float32)
    mask, _, _ = core_mask(y, core_fraction)
    core = y[mask]
    core_extent = float(np.max(core.max(axis=0) - core.min(axis=0)))
    return {"grid_span": span, "grid_boxes_per_dim": boxes, "grid_box_width": width,
            "core_boxes": core_extent / width}


def trace_row(epoch: int, y_before: np.ndarray, y_after: np.ndarray, attr: np.ndarray | None,
              core_fraction: float, far_multiple: float) -> dict:
    step = y_after - y_before
    step_norm = _norms(step)
    row = {
        "epoch": epoch,
        **{f"before_{k}": v for k, v in grid_stats(y_before, core_fraction).items()},
        "step_norm_max": float(step_norm.max()),
        "step_norm_p999": float(np.percentile(step_norm, 99.9)),
        "step_norm_median": float(np.median(step_norm)),
        "n_step_gt_4": int((step_norm > 4.0).sum()),
    }
    if attr is not None:
        attr_norm = _norms(attr)
        repl_norm = _norms(step - attr)
        row.update({
            "attr_norm_max": float(attr_norm.max()),
            "attr_norm_p999": float(np.percentile(attr_norm, 99.9)),
            "repl_norm_max": float(repl_norm.max()),
            "repl_norm_p999": float(np.percentile(repl_norm, 99.9)),
            "argmax_attr": int(attr_norm.argmax()),
            "argmax_repl": int(repl_norm.argmax()),
        })
    metrics = bbox_metrics(y_after, core_fraction, far_multiple)
    for key in ("bbox_area", "core_bbox_area", "bbox_area_ratio", "radius_max_over_core", "far_count"):
        row[f"after_{key}"] = metrics.get(key)
    return row


@contextmanager
def epoch_trace_hook(rows: list, core_fraction: float, far_multiple: float,
                     snapshot_steps: set | None = None, snapshot_dir=None):
    """Record one trace row per update; optionally save the state after t updates."""
    import ibumap.optimizers.ibumap_optimizer as module

    original = module.ibFFT_repulsive_sampling
    state = {"calls": 0, "prev": None, "prev_attr": None}
    snapshot_steps = snapshot_steps or set()

    def wrapped(y, *args, **kwargs):
        y_now = np.array(y, dtype=np.float64, copy=True)
        if state["calls"] in snapshot_steps and snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.save(snapshot_dir / f"step_{state['calls']:04d}.npy", np.array(y, dtype=np.float32, copy=True))
        if state["prev"] is not None:
            rows.append(trace_row(state["calls"] - 1, state["prev"], y_now, state["prev_attr"],
                                  core_fraction, far_multiple))
        fused = kwargs.get("fused_update")
        attr = None if fused is None else np.array(fused["attr_force"], dtype=np.float64, copy=True)
        state.update(prev=y_now, prev_attr=attr, calls=state["calls"] + 1)
        return original(y, *args, **kwargs)

    module.ibFFT_repulsive_sampling = wrapped
    try:
        yield state
    finally:
        module.ibFFT_repulsive_sampling = original


def run_traced(graph, init, params, core_fraction, far_multiple, snapshot_steps=None, snapshot_dir=None):
    """Optimize from the prepared graph under the trace hook; return (embedding, rows, extras)."""
    from common.algorithm_adapters import run_ibumap_optimize_from_prepared_graph

    rows: list = []
    with epoch_trace_hook(rows, core_fraction, far_multiple, snapshot_steps, snapshot_dir) as state:
        # Fresh copies: every run starts from the unmodified fixed inputs.
        embedding, extras = run_ibumap_optimize_from_prepared_graph(
            None, graph.copy(), np.asarray(init, dtype=np.float32, order="C").copy(), params,
            algorithm="ibumap", device="cpu")
    if state["prev"] is not None:
        rows.append(trace_row(state["calls"] - 1, state["prev"], np.asarray(embedding, dtype=np.float64),
                              state["prev_attr"], core_fraction, far_multiple))
    if state["calls"] != int(params["n_epochs"]):
        raise RuntimeError(f"expected {params['n_epochs']} field evaluations, observed {state['calls']}")
    return np.asarray(embedding, dtype=np.float32, order="C"), rows, extras


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_inputs(dataset: str, graph, init) -> dict:
    digests = input_digests(graph, init)
    paper = (CFG.get("paper_fixed_input_digests") or {}).get(dataset, {})
    match = bool(paper) and all(digests[k] == paper.get(k) for k in ("optimizer_graph", "init_embedding"))
    print(f"{dataset}: fixed inputs {'match' if match else 'DIFFER FROM'} the paper's "
          f"(graph {digests['optimizer_graph'][:12]}, init {digests['init_embedding'][:12]})")
    return {**digests, "match_paper": match}


def run_one(layout: Layout, dataset: str, variant: str, seed: int, graph, init, inputs: dict,
            overwrite: bool) -> None:
    out = layout.embedding(dataset, variant, seed)
    if out.exists() and not overwrite:
        print(f"skip existing {repo_relative(out)}")
        return
    params = dict(CFG["params"])
    params.update(CFG["variants"][variant] or {})
    params["random_state"] = int(seed)
    metrics = CFG["metrics"]
    plotted = int(seed) == int(CFG["plot_seed"])
    snapshot_steps = {int(v) for v in CFG.get("snapshot_steps", [])} if plotted else set()

    start = time.perf_counter()
    embedding, rows, extras = run_traced(graph, init, params, metrics["core_fraction"],
                                         metrics["far_radius_multiple"], snapshot_steps,
                                         layout.snapshots(dataset, variant, seed))
    wall = time.perf_counter() - start
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embedding)
    write_csv(layout.trace(dataset, variant, seed), rows)
    used = (extras.get("effective_config") or {}).get("used_parameters", {})
    write_json(layout.metadata(dataset, variant, seed), {
        "dataset": dataset, "variant": variant, "seed": int(seed),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "params": params, "used_parameters": used,
        "fixed_input_dir": repo_relative(fixed_input_dir(CFG, dataset)),
        "fixed_inputs": inputs,
        "embedding_sha256": sha256_array(embedding),
        "field_evaluations": len(rows),
        "snapshot_steps": sorted(snapshot_steps),
        "wall_seconds_provenance_only": wall,
        "bbox_metrics": bbox_metrics(embedding, metrics["core_fraction"], metrics["far_radius_multiple"]),
        "environment": {"python": platform.python_version(), "packages": package_versions(),
                        "numba_threads": numba.get_num_threads(),
                        "fft_threads": int(os.environ["IBUMAP_FFT_THREADS"]), "machine": machine_info()},
    })
    print(f"{dataset:12s} {variant:22s} seed={seed:<5d} wall={wall:6.1f}s "
          f"finite={bool(np.isfinite(embedding).all())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="execute (default: print the plan)")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--variant", action="append", choices=list(CFG["variants"]))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = args.dataset or CFG["datasets"]
    variants = args.variant or list(CFG["variants"])
    seeds = args.seed or CFG["seeds"]
    print(f"{len(datasets) * len(variants) * len(seeds)} runs, numba/fft threads={THREADS}")
    if not args.run:
        for d in datasets:
            for v in variants:
                for s in seeds:
                    print("  would run", d, v, s)
        return
    numba.set_num_threads(THREADS)
    layout = Layout(CFG)
    for dataset in datasets:
        graph, init = load_fixed_inputs(CFG, dataset)
        inputs = check_inputs(dataset, graph, init)
        for variant in variants:
            for seed in seeds:
                run_one(layout, dataset, variant, seed, graph, init, inputs, args.overwrite)


if __name__ == "__main__":
    main()
