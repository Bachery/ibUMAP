#!/usr/bin/env python3
"""Appendix table tab:mechanism-psweep: interpolation-order sweep on 30 datasets, paired against p = 1.

Optimizer only (fixed graph and spectral initialization). Quality is averaged over the
three repeats (bitwise identical for these deterministic variants); runtime uses the
per-dataset median of the three repeats. Writes paper/build/tables/ch4_psweep_rows.tex.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

from _common import check_group, parse_args, read_csv, write_rows

METRICS = [("trustworthiness", "TW"), ("continuity", "C"), ("neighborhood_preservation", "NP")]


def main() -> None:
    _, paths = parse_args(__doc__.splitlines()[0])
    check_group(paths, "psweep")
    quality = defaultdict(list)
    for row in read_csv(paths.data / "psweep" / "quality.csv.gz"):
        if row["status"] == "ok":
            quality[(row["dataset"], row["device"], row["variant"], row["metric"])].append(float(row["value"]))
    qmean = {k: st.mean(v) for k, v in quality.items()}
    runtime = defaultdict(list)
    for row in read_csv(paths.data / "psweep" / "runs.csv.gz"):
        if row["status"] == "ok":
            runtime[(row["dataset"], row["device"], row["variant"])].append(float(row["optimization_wall_s"]))
    rmed = {k: st.median(v) for k, v in runtime.items()}

    datasets = sorted({k[0] for k in rmed})
    lines = []
    for device, label in [("cpu", "CPU"), ("cuda", "CUDA")]:
        for variant in ["p2", "p3"]:
            cells = []
            for metric, _ in METRICS:
                diffs = [qmean[(d, device, variant, metric)] - qmean[(d, device, "p1", metric)]
                         for d in datasets if (d, device, variant, metric) in qmean]
                cells.append(f"${st.median(diffs):+.5f}$ ({sum(x > 0 for x in diffs)}/{len(diffs)})")
            ratios = [rmed[(d, device, variant)] / rmed[(d, device, "p1")]
                      for d in datasets if (d, device, variant) in rmed]
            head = f"{label} & $p={variant[1]}$" if variant == "p2" else f" & $p={variant[1]}$"
            lines.append(f"{head} & " + " & ".join(cells) + f" & ${st.median(ratios):.3f}\\times$ \\\\")
        if device == "cpu":
            lines.append("\\midrule")
    write_rows(paths, "ch4_psweep_rows.tex", lines, "ch4_psweep_table.py")
    print(f"ch4_psweep_table.py: {len(datasets)} datasets")


if __name__ == "__main__":
    main()
