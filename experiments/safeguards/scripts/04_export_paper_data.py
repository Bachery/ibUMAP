#!/usr/bin/env python3
"""Convert results/ to the paper-data layout (``<output>/safeguards/``).

    python scripts/04_export_paper_data.py                 # -> paper/rerun/safeguards/
    python scripts/04_export_paper_data.py --output DIR    # -> DIR/safeguards/

Writes the files that ``scripts/paper/ch4_safeguard_cases.py`` reads:

  runs.csv.gz            results/bbox_summary.csv
  cases.csv.gz           results/case_summary.csv
  traces.csv.gz          per-update traces of the plotted seed, dataset/variant/seed prepended
  escaped_points.csv.gz  results/escaped_points.csv
  reference.csv.gz       results/reference_bbox.csv
  embeddings.npz         final embeddings of the plotted seed, keys <dataset>__<variant>
  protocol.json          datasets, seeds, parameters, variants, metric definitions,
                         digests of the fixed inputs, run environment

CSV values are copied as strings. The export refuses incomplete results (run
01_run_embeddings.py for all runs, then 02_summarize.py and 03_reference_bbox.py).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from _common import PROTOCOL, Layout, load_config, read_json, resolve
from common.paper_data import DEFAULT_RERUN_ROOT, read_csv, write_csv_gz, write_json, write_npz

GROUP = "safeguards"
CFG = load_config()
DESCRIPTION = ("Production ibUMAP (all safeguards) versus the same optimizer with the repulsion-norm clip "
               "or the attraction degree damping switched off, on the fixed graphs and spectral "
               "initializations of the mechanism study.")
DEFINITIONS = {
    "bbox_area_ratio": "bounding-box area of all points / that of the core_fraction points nearest the "
                       "coordinate-wise median",
    "far": "distance to the median > far_radius_multiple x core radius (distance of the last core point)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_RERUN_ROOT,
                        help="paper-data root; the group is written to OUTPUT/safeguards (default: paper/rerun)")
    args = parser.parse_args()
    layout = Layout(CFG)
    results = layout.results
    datasets, variants, seeds = CFG["datasets"], list(CFG["variants"]), [int(s) for s in CFG["seeds"]]
    seed = int(CFG["plot_seed"])

    for name in ("bbox_summary.csv", "case_summary.csv", "escaped_points.csv", "reference_bbox.csv"):
        if not (results / name).exists():
            raise SystemExit(f"Missing results/{name}; run 02_summarize.py and 03_reference_bbox.py first.")
    r_fields, runs = read_csv(results / "bbox_summary.csv")
    expected = {(d, v, str(s)) for d in datasets for v in variants for s in seeds}
    found = {(r["dataset"], r["variant"], r["seed"]) for r in runs}
    if found != expected:
        raise SystemExit(f"Incomplete runs: missing {sorted(expected - found)}; rerun 01 and 02.")
    c_fields, cases = read_csv(results / "case_summary.csv")
    if {(r["dataset"], r["variant"]) for r in cases} != {(d, v) for d in datasets for v in variants}:
        raise SystemExit("case_summary.csv does not cover every dataset and variant; rerun 02_summarize.py.")
    e_fields, escaped = read_csv(results / "escaped_points.csv")
    f_fields, reference = read_csv(results / "reference_bbox.csv")
    if not reference:
        raise SystemExit("reference_bbox.csv is empty; run the mechanism experiment and 03_reference_bbox.py.")

    out = args.output.resolve() / GROUP
    files = {
        "runs.csv.gz": write_csv_gz(out / "runs.csv.gz", r_fields, runs),
        "cases.csv.gz": write_csv_gz(out / "cases.csv.gz", c_fields, cases),
        "escaped_points.csv.gz": write_csv_gz(out / "escaped_points.csv.gz", e_fields, escaped),
        "reference.csv.gz": write_csv_gz(out / "reference.csv.gz", f_fields, reference),
    }

    trace_fields, trace_rows, arrays, digests = None, [], {}, {}
    for dataset in datasets:
        for variant in variants:
            fields, rows = read_csv(layout.trace(dataset, variant, seed))
            fields = ["dataset", "variant", "seed"] + fields
            if trace_fields is None:
                trace_fields = fields
            if fields != trace_fields:
                raise SystemExit(f"{dataset}/{variant}: trace columns differ from the other traces")
            trace_rows += [{"dataset": dataset, "variant": variant, "seed": str(seed), **row} for row in rows]
            y = np.load(layout.embedding(dataset, variant, seed))
            wanted = next(r["embedding_sha256"] for r in runs
                          if (r["dataset"], r["variant"], r["seed"]) == (dataset, variant, str(seed)))
            if hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest() != wanted:
                raise SystemExit(f"{dataset}/{variant}: embedding does not match bbox_summary.csv")
            arrays[f"{dataset}__{variant}"] = y
        inputs = read_json(layout.metadata(dataset, "full", seed))["fixed_inputs"]
        digests[dataset] = {k: inputs[k] for k in ("optimizer_graph", "init_embedding", "n", "optimizer_graph_nnz")}
    files["traces.csv.gz"] = write_csv_gz(out / "traces.csv.gz", trace_fields, trace_rows)
    files["embeddings.npz"] = write_npz(out / "embeddings.npz", arrays)

    mechanism = resolve(CFG["reference"]["mechanism_root"])
    entries = yaml.safe_load((mechanism / "configs" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    meta = read_json(layout.metadata(datasets[0], "full", seed))
    protocol = {
        "protocol": PROTOCOL,
        "description": DESCRIPTION,
        "datasets": datasets,
        "seeds": CFG["seeds"],
        "plot_seed": seed,
        "params": CFG["params"],
        "variants": CFG["variants"],
        "metrics": {**CFG["metrics"], **DEFINITIONS},
        "fixed_input_digests": digests,
        "reference": {"suites": CFG["reference"]["suites"], "algorithms": CFG["reference"]["algorithms"],
                      "excluded_datasets": sorted(e["name"] for e in entries if e.get("excluded"))},
        "environment": {k: meta["environment"][k] for k in ("python", "packages", "numba_threads", "fft_threads")},
    }
    files["protocol.json"] = write_json(out / "protocol.json", protocol)
    print(f"Exported {len(runs)} runs, {len(trace_rows)} trace rows, {len(reference)} reference rows -> {out}")
    for name, digest in files.items():
        print(f"  {name}  {digest}")


if __name__ == "__main__":
    main()
