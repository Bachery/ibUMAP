#!/usr/bin/env python3
"""Combine the three size suites into the paper-data layout (``<output>/mechanism/``).

    python scripts/08_export_paper_data.py                 # -> paper/rerun/mechanism/
    python scripts/08_export_paper_data.py --output DIR    # -> DIR/mechanism/

Writes the files read by ``scripts/paper/ch4_*.py``:

  scores.csv.gz        one row per retained dataset, variant A-H plus sync_sampled_clip, and optimizer seed:
                       trustworthiness, continuity, neighborhood preservation, RTA, distance Spearman
  workloads.json       retained datasets, excluded datasets with reasons, variants, seeds, metric configuration
  calendar_audit.json  per-dataset sampling-calendar replay (07_calendar_audit.py)

Every suite summary must be complete (``05_summarize_plot.py`` without
``--allow-incomplete``), and every score and embedding hash is checked against
its metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _common import ROOT, SIZE_SUITES, read_json, sha
from common.paper_data import DEFAULT_RERUN_ROOT, write_csv_gz, write_json

GROUP = "mechanism"
VARIANTS = ("umap_learn", "sync_sampled", "sync_expected", "sync_direct", "sync_direct_capped", "sync_fft",
            "sync_fft_guarded", "ibumap_production", "sync_sampled_clip")
LOCAL_METRICS = ("trustworthiness", "continuity", "neighborhood_preservation")
GLOBAL_METRICS = ("rta", "distance_spearman")
FIELDS = ("suite", "dataset", "n", "family", "algorithm", "seed", *LOCAL_METRICS, *GLOBAL_METRICS,
          "embedding_sha256")
GLOBAL_CONFIG_KEYS = ("rta_n_triplets", "rta_random_state", "distance_spearman_n_pairs",
                      "distance_spearman_random_state", "source_metric", "dtype")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output", type=Path, default=DEFAULT_RERUN_ROOT,
                        help="paper-data root; the group is written to OUTPUT/mechanism (default: paper/rerun)")
    parser.add_argument("--suite", action="append", choices=list(SIZE_SUITES),
                        help="export only these suites (smoke tests); the paper uses all three")
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    cfg = {n: yaml.safe_load((config_dir / f"{n}.yaml").read_text()) for n in ("experiment", "datasets", "evaluation")}
    results = (config_dir / cfg["experiment"]["results_root"]).resolve()
    seeds = [int(s) for s in cfg["experiment"]["seeds"]]
    suites = args.suite or list(SIZE_SUITES)

    rows, datasets = [], []
    for suite in suites:
        root = results / suite
        manifest = read_json(root / "05_summary" / "manifest.json")
        if manifest.get("status") != "complete" or manifest.get("missing_or_stale"):
            raise SystemExit(f"{suite}: summary is incomplete; rerun 05_summarize_plot.py --suite {suite}")
        for d in manifest["datasets"]:
            datasets.append({"name": d["name"], "n": d["n"], "family": d["family"], "size_suite": suite})
            for algorithm in VARIANTS:
                for seed in seeds:
                    run = f"{algorithm}__seed_{seed}"
                    score_path = root / "04_quality" / d["name"] / run / "scores.json"
                    meta = read_json(score_path.with_name("metadata.json"))
                    if meta.get("status") != "ok" or meta["output_hashes"]["scores.json"] != sha(score_path):
                        raise SystemExit(f"Score record does not match its metadata: {score_path}")
                    scalars = read_json(score_path)["scalars"]
                    run_meta = read_json(root / "02_runs" / d["name"] / run / "metadata.json")
                    embedding = run_meta["output_hashes"]["embedding.npy"]
                    if meta.get("embedding_sha256") != embedding:
                        raise SystemExit(f"Scores were computed on a different embedding: {score_path}")
                    rows.append({"suite": suite, "dataset": d["name"], "n": d["n"], "family": d["family"],
                                 "algorithm": algorithm, "seed": seed,
                                 **{m: repr(float(scalars[m])) for m in LOCAL_METRICS + GLOBAL_METRICS},
                                 "embedding_sha256": embedding})
    names = [d["name"] for d in datasets]
    if len(names) != len(set(names)):
        raise SystemExit("A dataset appears in more than one suite")

    excluded = [{"name": d["name"], "n": d["n"], "reason": d["excluded"]}
                for d in cfg["datasets"]["datasets"] if d.get("excluded")
                and any(SIZE_SUITES[s][0] < d["n"] <= SIZE_SUITES[s][1] for s in suites)]
    evaluation = cfg["evaluation"]
    workloads = {
        "description": "Mechanism experiment (paper Section 4 and appendix): fixed graph and initialization, "
                       "eight variants A-H plus the sampled-norm-clip control, optimizer seeds "
                       + ", ".join(map(str, seeds)) + ".",
        "variants": list(VARIANTS), "seeds": seeds,
        "local_metrics": list(LOCAL_METRICS), "global_metrics": list(GLOBAL_METRICS),
        "global_configuration": {"rows": "all", **{k: evaluation.get(k) for k in GLOBAL_CONFIG_KEYS}},
        "selection": {"suites": suites, "candidate_count": len(datasets) + len(excluded),
                      "retained_count": len(datasets), "excluded_datasets": excluded},
        "datasets": datasets,
    }

    audit_dir = results / "07_calendar_audit" / "per_dataset"
    calendar = [json.loads((audit_dir / f"{name}.json").read_text()) for name in sorted(names)]

    out = args.output.resolve() / GROUP
    files = {
        "scores.csv.gz": write_csv_gz(out / "scores.csv.gz", FIELDS, rows),
        "workloads.json": write_json(out / "workloads.json", workloads),
        "calendar_audit.json": write_json(out / "calendar_audit.json", calendar),
    }
    print(f"Exported {len(rows)} score rows for {len(datasets)} datasets "
          f"({len(excluded)} excluded) -> {out}")
    for name, digest in files.items():
        print(f"  {name}  {digest}")


if __name__ == "__main__":
    main()
