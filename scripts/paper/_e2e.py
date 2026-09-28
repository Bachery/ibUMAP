"""Load the frozen end-to-end benchmark (paper/data/e2e_benchmark) and build plot rows.

Aggregation rules (unchanged from the experiment scripts):
  runtime   per dataset/algorithm: median of five ``e2e_call_wall_s``; speedup = baseline / ibUMAP
  quality   per dataset/algorithm: mean of the five unseeded runs (seeded: repeat 1)
  stability per dataset/algorithm: median over reference repeat 1 vs repeats 2-5
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median

from _common import Paths, check_group, read_csv, read_json
from _style import COMPARISONS, ORANGE, BLUE, STABILITY_METRICS

GROUP = "e2e_benchmark"
LOCAL_METRICS = ["trustworthiness", "continuity", "neighborhood_preservation"]
GLOBAL_METRICS = ["rta", "distance_spearman"]
QUALITY_METRICS = LOCAL_METRICS + GLOBAL_METRICS


@dataclass
class E2E:
    bundle: dict
    runs: list[dict[str, str]]
    quality: list[dict[str, str]]
    stability: list[dict[str, str]]
    datasets: list[dict[str, str]]
    source_sha256: dict[str, str]


def load(paths: Paths) -> E2E:
    hashes = check_group(paths, GROUP)
    root = paths.data / GROUP
    bundle = read_json(root / "bundle.json")
    if not all(bundle.get(flag) for flag in ("strict_passed", "full_scope", "published")):
        raise ValueError("The paper figures require the published, strict, full-scope benchmark.")
    data = E2E(bundle=bundle, runs=read_csv(root / "runs.csv.gz"), quality=read_csv(root / "quality.csv.gz"),
               stability=read_csv(root / "stability.csv.gz"), datasets=read_csv(root / "datasets.csv.gz"),
               source_sha256=hashes)
    keys = [(r["dataset"], r["algorithm_id"], r["repeat"]) for r in data.runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset/algorithm/repeat in runs.csv.gz")
    if sum(r["status"] == "ok" for r in data.runs) != bundle["successful_runs"]:
        raise ValueError("Run count does not match bundle.json")
    return data


def _groups(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["dataset"], run["algorithm_id"]].append(run)
    return groups


def _skipped(scope, dataset, *algorithms) -> bool:
    skip = scope.get("resource_skip_rules", {})
    return dataset in skip.get("datasets", []) and any(a in skip.get("algorithms", []) for a in algorithms)


def _median_time(groups, scope, dataset, algorithm, profile=None):
    records = groups.get((dataset, algorithm), [])
    if {int(r["repeat"]) for r in records} != {int(r) for r in scope["repeats"]}:
        raise ValueError(f"Incomplete repeats: {dataset}/{algorithm}")
    if any(r["status"] != "ok" for r in records):
        raise ValueError(f"Non-successful run: {dataset}/{algorithm}")
    if profile is not None and any(r["execution_profile"] != profile for r in records):
        raise ValueError(f"Mismatched execution profile: {dataset}/{algorithm}")
    values = [float(r["e2e_call_wall_s"]) for r in records]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"Invalid E2E time: {dataset}/{algorithm}")
    sizes = {int(r["n_samples"]) for r in records}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent sample count: {dataset}/{algorithm}")
    return median(values), sizes.pop()


def runtime_rows(data: E2E) -> tuple[list[dict], list[dict]]:
    """Unseeded speedups for the three comparisons, and CPU seeded/unseeded time ratios."""
    groups, scope = _groups(data.runs), data.bundle["scope"]
    unseeded, seed_cost = [], []
    for comparison in COMPARISONS:
        baseline = comparison["baseline"] + "_seed_none"
        candidate = comparison["candidate"] + "_seed_none"
        for dataset in scope["datasets"]:
            if _skipped(scope, dataset, baseline, candidate):
                continue
            b, n = _median_time(groups, scope, dataset, baseline)
            c, n_c = _median_time(groups, scope, dataset, candidate)
            if n != n_c:
                raise ValueError(f"Different input sizes in pair: {dataset}")
            unseeded.append({
                "comparison": comparison["key"], "dataset": dataset, "n_samples": n,
                "baseline_algorithm": baseline, "candidate_algorithm": candidate,
                "baseline_median_e2e_s": b, "ibumap_median_e2e_s": c, "speedup": b / c,
            })
    for implementation, algorithm, color in (("umap-learn", "umap_learn", ORANGE), ("ibUMAP CPU", "ibumap_cpu", BLUE)):
        for dataset in scope["datasets"]:
            seeded, unseeded_id = algorithm + "_seed_42", algorithm + "_seed_none"
            if _skipped(scope, dataset, seeded, unseeded_id):
                continue
            s, n = _median_time(groups, scope, dataset, seeded)
            u, n_u = _median_time(groups, scope, dataset, unseeded_id)
            if n != n_u:
                raise ValueError(f"Different input sizes in profile pair: {dataset}")
            seed_cost.append({
                "implementation": implementation, "algorithm": algorithm, "dataset": dataset,
                "n_samples": n, "seeded_median_e2e_s": s, "unseeded_median_e2e_s": u,
                "seeded_over_unseeded": s / u, "color": color,
            })
    return unseeded, seed_cost


def cpu_seeded_rows(data: E2E, datasets: set[str]) -> list[dict]:
    """Seeded CPU speedups on exactly the unseeded CPU population."""
    groups = defaultdict(list)
    for row in data.runs:
        if row["algorithm_id"] in ("umap_learn_seed_42", "ibumap_cpu_seed_42"):
            groups[row["dataset"], row["algorithm_id"]].append(row)
    for algorithm in ("umap_learn_seed_42", "ibumap_cpu_seed_42"):
        if {d for d, a in groups if a == algorithm} != datasets:
            raise ValueError(f"Seeded/unseeded CPU coverage differs for {algorithm}.")
    output = []
    for dataset in sorted(datasets):
        times, sizes = {}, set()
        for algorithm in ("umap_learn_seed_42", "ibumap_cpu_seed_42"):
            part = groups[dataset, algorithm]
            if len(part) != 5 or {int(r["repeat"]) for r in part} != set(range(1, 6)):
                raise ValueError(f"Incomplete seeded repetitions: {dataset}/{algorithm}")
            if any(r["status"] != "ok" for r in part):
                raise ValueError(f"Failed seeded run: {dataset}/{algorithm}")
            values = [float(r["e2e_call_wall_s"]) for r in part]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Invalid seeded runtime: {dataset}/{algorithm}")
            times[algorithm] = median(values)
            sizes.update(int(r["n_samples"]) for r in part)
        if len(sizes) != 1:
            raise ValueError(f"Input sizes differ within seeded CPU pair: {dataset}")
        b, c = times["umap_learn_seed_42"], times["ibumap_cpu_seed_42"]
        output.append({
            "comparison": "cpu_umap", "dataset": dataset, "n_samples": sizes.pop(),
            "baseline_algorithm": "umap_learn_seed_42", "candidate_algorithm": "ibumap_cpu_seed_42",
            "baseline_median_e2e_s": b, "ibumap_median_e2e_s": c, "speedup": b / c,
            "execution_profile": "seeded",
        })
    return output


def profile_rows(data: E2E, comparisons: dict) -> list[dict]:
    """Seeded and unseeded speedups for the CPU/umap-learn and GPU/cuML pairs (appendix Figure A)."""
    groups, scope = _groups(data.runs), data.bundle["scope"]
    output = []
    for device, comparison in comparisons.items():
        for profile, suffix in (("seeded", "42"), ("unseeded", "none")):
            baseline = comparison["baseline_id"] + "_seed_" + suffix
            candidate = comparison["candidate_id"] + "_seed_" + suffix
            for dataset in scope["datasets"]:
                if _skipped(scope, dataset, baseline, candidate):
                    continue
                b, n = _median_time(groups, scope, dataset, baseline, profile)
                c, n_c = _median_time(groups, scope, dataset, candidate, profile)
                if n != n_c:
                    raise ValueError(f"Inconsistent sample count: {dataset}")
                output.append({
                    "device": device, "dataset": dataset, "execution_profile": profile,
                    "n_samples": n, "baseline_algorithm": baseline, "candidate_algorithm": candidate,
                    "repeats": len(scope["repeats"]), "baseline_median_e2e_s": b,
                    "ibumap_median_e2e_s": c, "speedup": b / c,
                })
    return output


def quality_delta_rows(data: E2E, metrics) -> list[dict]:
    """Unseeded paired quality differences (candidate mean - baseline mean over five runs)."""
    repeats = {int(r) for r in data.bundle["scope"]["repeats"]}
    by_key = defaultdict(list)
    for row in data.quality:
        if row["status"] == "ok" and row["execution_profile"] == "unseeded":
            by_key[row["dataset"], row["algorithm_id"], row["metric"]].append(row)
    output = []
    for comparison in COMPARISONS:
        candidate = comparison["candidate"] + "_seed_none"
        baseline = comparison["baseline"] + "_seed_none"
        for metric, _ in metrics:
            candidates = {d: rows for (d, a, m), rows in by_key.items() if a == candidate and m == metric}
            baselines = {d: rows for (d, a, m), rows in by_key.items() if a == baseline and m == metric}
            for dataset in sorted(set(candidates) & set(baselines)):
                c_rows, b_rows = candidates[dataset], baselines[dataset]
                if {int(r["repeat"]) for r in c_rows} != repeats or {int(r["repeat"]) for r in b_rows} != repeats:
                    raise ValueError(f"Incomplete quality repetitions for {dataset}/{metric}")
                c_mean = mean(float(r["value"]) for r in c_rows)
                b_mean = mean(float(r["value"]) for r in b_rows)
                output.append({"comparison": comparison["key"], "dataset": dataset, "metric": metric,
                               "candidate_mean": c_mean, "baseline_mean": b_mean, "paired_delta": c_mean - b_mean})
    return output


def stability_delta_rows(data: E2E) -> list[dict]:
    groups = defaultdict(list)
    for row in data.stability:
        if row["status"] == "ok" and row["execution_profile"] == "unseeded":
            groups[row["dataset"], row["algorithm_id"]].append(row)
    per_dataset = {}
    for (dataset, algorithm), rows in groups.items():
        if {(int(r["reference_repeat"]), int(r["candidate_repeat"])) for r in rows} != {(1, k) for k in range(2, 6)}:
            raise ValueError(f"Unexpected stability comparisons for {dataset}/{algorithm}")
        per_dataset[dataset, algorithm] = {m: median(float(r[m]) for r in rows) for m, _ in STABILITY_METRICS}
    output = []
    for comparison in COMPARISONS:
        candidate = comparison["candidate"] + "_seed_none"
        baseline = comparison["baseline"] + "_seed_none"
        for (dataset, algorithm), values in per_dataset.items():
            if algorithm != candidate or (dataset, baseline) not in per_dataset:
                continue
            base = per_dataset[dataset, baseline]
            for metric, _ in STABILITY_METRICS:
                output.append({"comparison": comparison["key"], "dataset": dataset, "metric": metric,
                               "candidate_value": values[metric], "baseline_value": base[metric],
                               "paired_delta": values[metric] - base[metric]})
    return output
