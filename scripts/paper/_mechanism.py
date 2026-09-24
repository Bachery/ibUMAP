"""Load the frozen mechanism experiment (paper/data/mechanism)."""
from __future__ import annotations

import pandas as pd

from _common import Paths, check_group, read_csv, read_json

GROUP = "mechanism"
LOCAL_METRICS = ["trustworthiness", "continuity", "neighborhood_preservation"]
GLOBAL_METRICS = ["rta", "distance_spearman"]
ALL_METRICS = LOCAL_METRICS + GLOBAL_METRICS
VARIANTS = [("A", "umap_learn", "UMAP"), ("B", "sync_sampled", "Synchronous sampled"),
            ("C", "sync_expected", "Synchronous event expectation"),
            ("D", "sync_direct", "Direct degree-weighted field"),
            ("E", "sync_direct_capped", "D + kernel cap"), ("F", "sync_fft", "E with default FFT"),
            ("G", "sync_fft_guarded", "F + norm clip and damping"), ("H", "ibumap_production", "Production ibUMAP")]
TIE = 1e-12  # paired differences within +/- TIE count as neither improved nor worse


def load_scores(paths: Paths):
    """Return (raw DataFrame with one row per dataset/algorithm/seed, workloads record, source hashes)."""
    hashes = check_group(paths, GROUP)
    root = paths.data / GROUP
    workloads = read_json(root / "workloads.json")
    raw = [dict(dataset=r["dataset"], n=int(r["n"]), family=r["family"], algorithm=r["algorithm"],
                seed=int(r["seed"]), **{m: float(r[m]) for m in ALL_METRICS})
           for r in read_csv(root / "scores.csv.gz")]
    names = [d["name"] for d in workloads["datasets"]]
    assert len(names) == len(set(names)) == workloads["selection"]["retained_count"]
    assert {r["dataset"] for r in raw} == set(names)
    assert len(raw) == len(names) * len(workloads["variants"]) * len(workloads["seeds"])
    return pd.DataFrame(raw), workloads, hashes


def per_dataset_means(raw: pd.DataFrame) -> pd.DataFrame:
    """Average optimizer seeds within each dataset (deterministic aliases are not independent repeats)."""
    return raw.groupby(["dataset", "n", "family", "algorithm"])[ALL_METRICS].mean()
