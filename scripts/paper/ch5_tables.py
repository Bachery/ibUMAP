#!/usr/bin/env python3
"""Section 5 / Appendix C tables from the frozen end-to-end benchmark.

Writes to paper/build/tables/:
  ch5_quality_common_median_rows.tex   Table tab:e2e-fidelity-common (main text)
  ch5_runtime_rows.tex                 tab:benchmark-runtime
  ch5_profile_rows.tex                 tab:benchmark-profiles
  ch5_quality_absolute_{mean,median}_rows.tex, ch5_quality_rows.tex,
  ch5_quality_paired_median_rows.tex, ch5_collection_rows.tex,
  ch5_stability_rows.tex, ch5_stability_pair_rows.tex, ch5_dataset_rows.tex
  ch5_fidelity_plot_data.csv, ch5_stability_plot_data.csv, ch5_evidence.json
"""
import collections
import statistics as st

import _e2e
from _common import parse_args, write_csv, write_json, write_rows

BUILDER = "ch5_tables.py"


def main() -> None:
    _, paths = parse_args(__doc__.splitlines()[0])
    data = _e2e.load(paths)
    runs, quality, stability, datasets = data.runs, data.quality, data.stability, data.datasets
    assert data.bundle["strict_passed"] and data.bundle["successful_runs"] == 3400
    metric_names = _e2e.QUALITY_METRICS
    assert len(runs) == 3400
    assert len({(r["dataset"], r["run_id"]) for r in runs}) == len(runs)
    assert all(r["status"] == "ok" for r in runs + quality + stability)
    assert len({r["hardware_signature"] for r in runs}) == 1
    local_records = sum(r["source"] == "local" for r in quality)
    assert local_records == 10200

    times = collections.defaultdict(list)
    hashes = collections.defaultdict(set)
    scores = collections.defaultdict(list)
    sizes = {d["dataset"]: int(d["n_samples"]) for d in datasets}
    for r in runs:
        key = r["dataset"], r["algorithm_id"]
        times[key].append(float(r["e2e_call_wall_s"]))
        hashes[key].add(r["embedding_sha256"])
    run_hash = {(r["dataset"], r["algorithm_id"], r["repeat"]): r["embedding_sha256"] for r in runs}
    for r in quality:
        assert r["metric"] in metric_names
        assert r["embedding_sha256"] == run_hash[r["dataset"], r["algorithm_id"], r["repeat"]]
        # Seeded profiles are scored on repeat 1 (seeded runs are bitwise repeatable).
        if r["algorithm_id"].endswith("_seed_42") and int(r["repeat"]) != 1:
            continue
        scores[r["dataset"], r["algorithm_id"], r["metric"]].append(float(r["value"]))
    assert all(len(v) == 5 for v in times.values())
    assert all(len(v) == (1 if a.endswith("_seed_42") else 5) for (d, a, m), v in scores.items())
    assert {m for (_, _, m) in scores} == set(metric_names)

    comp = [("CPU/UMAP", "ibumap_cpu", "umap_learn"),
            ("GPU/cuML", "ibumap_cuda", "cuml_umap"),
            ("GPU/TorchDR", "ibumap_cuda", "torchdr_umap")]

    def write(name, rows):
        write_rows(paths, name, rows, BUILDER)

    def row(*values):
        return " & ".join(map(str, values)) + r" \\"

    def signed(v):
        return f"${v:+.4f}$"

    def collection(name):
        return next((p for p in ["tabula_sapiens", "c_elegans", "whole_mouse_brain"] if name.startswith(p)), name)

    audit = {"data": "paper/data/e2e_benchmark", "source_sha256": data.source_sha256,
             "metrics": metric_names,
             "global_metric_configuration": data.bundle["quality_metrics"]["global_configuration"],
             "reported_quality_records": sum(len(v) for v in scores.values()),
             "quality_aggregation": {"seeded": "repeat 1", "unseeded": "mean of five runs"},
             "runtime": [], "quality": [], "collection_quality": [], "profiles": [],
             "stability": [], "stability_pairs": []}

    # Per-size speed ratios, paired separately within each profile.
    runtime_rows = []
    for label, candidate, baseline in comp:
        bins = [("All", 0, float("inf"))]
        if candidate == "ibumap_cpu":
            bins += [(r"$N<10^3$", 0, 1000), (r"$10^3\leq N<10^4$", 1000, 10000),
                     (r"$10^4\leq N<10^5$", 10000, 100000), (r"$10^5\leq N<10^6$", 100000, 1000000),
                     (r"$N\geq10^6$", 1000000, float("inf"))]
        elif baseline == "cuml_umap":
            bins += [(r"$N<10^4$", 0, 10000), (r"$10^4\leq N<10^5$", 10000, 100000),
                     (r"$10^5\leq N<10^6$", 100000, 1000000), (r"$N\geq10^6$", 1000000, float("inf"))]
        for bin_label, lo, hi in bins:
            cells = []
            for seed in ["42", "none"]:
                values = [st.median(times[d, baseline + "_seed_" + seed]) / st.median(v)
                          for (d, a), v in times.items() if a == candidate + "_seed_" + seed
                          and (d, baseline + "_seed_" + seed) in times and lo <= sizes[d] < hi]
                entry = {"comparison": label, "scope": bin_label, "seed": seed, "n": len(values),
                         "median": st.median(values), "wins": sum(v > 1 for v in values)}
                audit["runtime"].append(entry)
                cells += [entry["median"], entry["wins"]]
            runtime_rows.append(row(label, bin_label, len(values), f"{cells[0]:.3f}", f"{cells[2]:.3f}",
                                    f"{cells[1]}/{len(values)}", f"{cells[3]}/{len(values)}"))
    write("ch5_runtime_rows.tex", runtime_rows)

    # Paired fidelity differences; all five metrics are higher-is-better.
    quality_rows, paired_median_rows, collection_rows, fidelity_points = [], [], [], []
    audit["quality_paired_medians"] = []
    for label, candidate, baseline in comp:
        for seed in ["42", "none"]:
            means, paired_medians, balanced = [], [], []
            for metric in metric_names:
                values = {}
                for d in sizes:
                    c, b = (d, candidate + "_seed_" + seed, metric), (d, baseline + "_seed_" + seed, metric)
                    if c in scores and b in scores:
                        values[d] = st.mean(scores[c]) - st.mean(scores[b])
                        fidelity_points.append({"dataset": d, "comparison": label, "seed": seed, "metric": metric,
                                                "candidate_mean": st.mean(scores[c]),
                                                "baseline_mean": st.mean(scores[b]),
                                                "paired_delta": values[d], "collection": collection(d)})
                means.append(st.mean(values.values()))
                paired_medians.append(st.median(values.values()))
                blocks = collections.defaultdict(list)
                for d, v in values.items():
                    blocks[collection(d)].append(v)
                balanced.append(st.mean(st.mean(v) for v in blocks.values()))
            profile = "Seeded" if seed == "42" else "Unseeded"
            quality_rows.append(row(label, profile, len(values), *(signed(v) for v in means)))
            paired_median_rows.append(row(label, profile, len(values),
                                          *(f"${v:+.6f}$" if m == "continuity" else signed(v)
                                            for m, v in zip(metric_names, paired_medians))))
            collection_rows.append(row(label, profile, len(blocks), *(signed(v) for v in balanced)))
            audit["quality"].append({"comparison": label, "profile": profile, "n": len(values),
                                     **dict(zip(metric_names, means))})
            audit["quality_paired_medians"].append({"comparison": label, "profile": profile, "n": len(values),
                                                    **dict(zip(metric_names, paired_medians))})
            audit["collection_quality"].append({"comparison": label, "profile": profile, "n_blocks": len(blocks),
                                                **dict(zip(metric_names, balanced))})
    write("ch5_quality_rows.tex", quality_rows)
    write("ch5_quality_paired_median_rows.tex", paired_median_rows)
    write("ch5_collection_rows.tex", collection_rows)
    assert all(r[m] < 0 for r in audit["collection_quality"]
               if r["comparison"] == "CPU/UMAP" or (r["comparison"] == "GPU/cuML" and r["profile"] == "Unseeded")
               for m in ["trustworthiness", "neighborhood_preservation"])

    algorithms = [("umap-learn", "umap_learn"), ("ibUMAP CPU", "ibumap_cpu"), ("cuML", "cuml_umap"),
                  ("ibUMAP CUDA", "ibumap_cuda"), ("TorchDR", "torchdr_umap")]

    # Common-population quality table and complete absolute score summaries.
    available = {a: {d for d, algo in times if algo == a + "_seed_none"} for _, a in algorithms}
    common = set.intersection(*available.values())
    full_gpu = available["cuml_umap"] & available["ibumap_cuda"]
    assert len(common) == 66 and len(full_gpu) == 71
    assert common == available["umap_learn"] == available["ibumap_cpu"] == available["torchdr_umap"]
    assert all({d for d, algo in times if algo == a + "_seed_42"} == available[a] for _, a in algorithms)
    audit["global_paired"] = []
    for label, candidate, baseline in comp:
        for metric in _e2e.GLOBAL_METRICS:
            delta = [st.mean(scores[d, candidate + "_seed_none", metric])
                     - st.mean(scores[d, baseline + "_seed_none", metric]) for d in sorted(common)]
            audit["global_paired"].append({"comparison": label, "metric": metric, "n": len(delta),
                                           "mean": st.mean(delta), "median": st.median(delta),
                                           "wins": sum(v > 0 for v in delta)})
    cpu_methods = [("umap-learn", "umap_learn"), ("ibUMAP", "ibumap_cpu")]
    gpu_methods = [("cuML", "cuml_umap"), ("TorchDR", "torchdr_umap"), ("ibUMAP", "ibumap_cuda")]
    quality_scopes = [("CPU: common 66 datasets", common, cpu_methods),
                      ("GPU: common 66 datasets", common, gpu_methods),
                      ("GPU: full 71 datasets", full_gpu, [("cuML", "cuml_umap"), ("ibUMAP", "ibumap_cuda")])]

    def quality_summary(ds, algorithm, seed, reducer):
        return [reducer(st.mean(scores[d, algorithm + "_seed_" + seed, metric]) for d in sorted(ds))
                for metric in metric_names]

    def quality_cells(values, best, *, compact=False):
        cells = []
        for i, v in enumerate(values):
            digits = 4 if compact or metric_names[i] != "continuity" else 6
            value = f"{v:.{digits}f}"
            is_best = value == f"{best[i]:.{digits}f}" if compact else v == best[i]
            cells.append(r"\textbf{" + value + "}" if is_best else value)
        return cells

    def column_best(matrix):
        return [max(v[i] for v in matrix) for i in range(len(matrix[0]))]

    common_rows = []
    audit["quality_common_population"] = {
        "datasets": sorted(common), "excluded_from_common": sorted(full_gpu - common),
        "within_dataset": "mean of five runs", "across_datasets": "median", "display_decimals": 4,
        "bold": "highest displayed value including rounding ties", "rows": []}
    for platform, methods in [("CPU", cpu_methods), ("GPU", gpu_methods)]:
        matrix = [quality_summary(common, a, "none", st.median) for _, a in methods]
        best = column_best(matrix)
        if common_rows:
            common_rows.append(r"\midrule")
        for j, ((label, algorithm), values) in enumerate(zip(methods, matrix)):
            common_rows.append(row(platform if j == 0 else "", label, *quality_cells(values, best, compact=True)))
            audit["quality_common_population"]["rows"].append({"platform": platform, "algorithm": algorithm,
                                                               "n": len(common), **dict(zip(metric_names, values))})
    write("ch5_quality_common_median_rows.tex", common_rows)
    audit["quality_absolute"] = []
    for name, reducer in [("mean", st.mean), ("median", st.median)]:
        rows = []
        for title, ds, methods in quality_scopes:
            if rows:
                rows.append(r"\midrule")
            rows.append(r"\multicolumn{7}{@{}l}{\emph{" + title + r"}} \\")
            for seed in ["none", "42"]:
                matrix = [quality_summary(ds, a, seed, reducer) for _, a in methods]
                best = column_best(matrix)
                for (label, algorithm), values in zip(methods, matrix):
                    rows.append(row(label, "Unseeded" if seed == "none" else "Seeded", *quality_cells(values, best)))
                    audit["quality_absolute"].append({"scope": title, "summary": name, "algorithm": algorithm,
                                                      "seed": seed, "n": len(ds), **dict(zip(metric_names, values))})
        write(f"ch5_quality_absolute_{name}_rows.tex", rows)

    profile_rows = []
    for label, algo in algorithms:
        values = [(d, st.median(v) / st.median(times[d, algo + "_seed_none"]))
                  for (d, a), v in times.items() if a == algo + "_seed_42"]
        large = [v for d, v in values if sizes[d] >= 1000000]
        identical = sum(len(hashes[d, algo + "_seed_42"]) == 1 for d, v in values)
        profile_rows.append(row(label, len(values), f"{st.median(v for d, v in values):.3f}", len(large),
                                f"{st.median(large):.3f}", f"{identical}/{len(values)}"))
        audit["profiles"].append({"algorithm": label, "n": len(values),
                                  "median_ratio": st.median(v for d, v in values), "n_large": len(large),
                                  "large_ratio": st.median(large), "identical": identical})
    write("ch5_profile_rows.tex", profile_rows)

    # Each dataset score is the median of comparisons of repeat 1 against 2--5.
    groups = collections.defaultdict(list)
    for r in stability:
        if r["execution_profile"] == "unseeded":
            groups[r["dataset"], r["algorithm_id"]].append(r)
    assert all(len(rows) == 4 for rows in groups.values())
    stability_metrics = ["neighbor_overlap_at_15", "pairwise_distance_spearman", "procrustes_rms"]
    stats_by_dataset = {(d, a): {m: st.median(float(r[m]) for r in rows) for m in stability_metrics}
                        for (d, a), rows in groups.items()}
    stable_rows = []
    for label, algo in algorithms:
        selected = [v for (d, a), v in stats_by_dataset.items() if a == algo + "_seed_none"]
        medians = [st.median(v[m] for v in selected) for m in stability_metrics]
        stable_rows.append(row(label, len(selected), f"{medians[0]:.4f}", f"{medians[1]:.4f}", f"{medians[2]:.6f}"))
        audit["stability"].append({"algorithm": label, "n": len(selected), **dict(zip(stability_metrics, medians))})
    write("ch5_stability_rows.tex", stable_rows)
    pair_rows, stable_points = [], []
    for label, candidate, baseline in comp:
        deltas = {m: [] for m in stability_metrics}
        for (d, a), v in stats_by_dataset.items():
            if a != candidate + "_seed_none" or (d, baseline + "_seed_none") not in stats_by_dataset:
                continue
            base = stats_by_dataset[d, baseline + "_seed_none"]
            for m in stability_metrics:
                delta = v[m] - base[m]
                deltas[m].append(delta)
                stable_points.append({"dataset": d, "comparison": label, "metric": m, "candidate_value": v[m],
                                      "baseline_value": base[m], "paired_delta": delta, "collection": collection(d)})
        counts = [sum(v > 0 for v in deltas[m]) for m in stability_metrics[:2]]
        n = len(deltas[stability_metrics[0]])
        pair_rows.append(row(label, n, f"{counts[0]}/{n}", signed(st.median(deltas[stability_metrics[0]])),
                             f"{counts[1]}/{n}", signed(st.median(deltas[stability_metrics[1]]))))
        audit["stability_pairs"].append({"comparison": label, "n": n, "neighbor_wins": counts[0],
                                         "distance_wins": counts[1]})
    write("ch5_stability_pair_rows.tex", pair_rows)

    # Compact readable identifiers, reversible using the prefix definitions in the appendix.
    # Daggers mark the five datasets outside the common 66-dataset population.
    manifest_rows = []
    for d in datasets:
        label = (d["dataset"].replace("tabula_sapiens_v2_", "TS: ").replace("c_elegans_embryogenesis_", "CE: ")
                 .replace("whole_mouse_brain_merfish_", "MERFISH: ").replace("scdeed_", "scDEED: "))
        label = label.replace("__", " / ").replace("_", " ")
        if d["dataset"] in full_gpu - common:
            label += r"$^{\dagger}$"
        manifest_rows.append(row(label, f"{int(d['n_samples']):,}", f"{int(d['n_features']):,}"))
    write("ch5_dataset_rows.tex", manifest_rows)
    write_csv(paths.tables / "ch5_fidelity_plot_data.csv", fidelity_points)
    write_csv(paths.tables / "ch5_stability_plot_data.csv", stable_points)
    write_json(paths.tables / "ch5_evidence.json", audit)
    print(f"{BUILDER}: wrote 11 tables, 2 plot-data files and ch5_evidence.json to {paths.tables}")


if __name__ == "__main__":
    main()
