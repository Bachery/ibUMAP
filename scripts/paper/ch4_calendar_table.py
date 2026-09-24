#!/usr/bin/env python3
"""Appendix table tab:calendar-audit: replayed umap-learn sampling calendars on the 59 mechanism datasets.

The replay (T = 200 epochs, m = 5 negative samples) runs on the saved optimizer graphs,
without random draws, optimization or quality evaluation; its per-dataset records are
frozen in paper/data/mechanism/calendar_audit.json. Per-point quantities are summarized by
their median within each dataset; the table gives the median and range across datasets.

Writes paper/build/tables/ch4_calendar_rows.tex and ch4_calendar_summary.json (all
statistics quoted in the appendix text, e.g. the 0.91--1.10 band from t = 10 onward).
"""
from __future__ import annotations

import numpy as np

from _common import check_group, parse_args, read_json, write_json, write_rows


def q(x):
    return {k: float(np.percentile(x, p)) for k, p in (("p5", 5), ("median", 50), ("p95", 95))} | \
           {"min": float(np.min(x)), "max": float(np.max(x))}


def summarize(rows):
    med = lambda key, stat="median": np.array([r[key][stat] for r in rows])  # noqa: E731
    epoch = lambda key: np.array([r["epoch_total_over_mean_t_ge_1"][key] for r in rows])  # noqa: E731
    s = {
        "n_datasets": len(rows), "n_range": [min(r["n"] for r in rows), max(r["n"] for r in rows)],
        "all_symmetric": all(r["symmetric_max_abs_diff"] == 0 for r in rows),
        "v_max_values": sorted({r["v_max"] for r in rows}),
        "min_v_min_over_v_max": min(r["v_min_over_v_max"] for r in rows),
        "isolated_vertices_total": sum(r["isolated_vertices"] for r in rows),
        "first_active_epochs": sorted({r["first_active_epoch"] for r in rows}),
        "epoch0_draws_total": sum(r["epoch0_scheduled_draws"] for r in rows),
        "head_tail_activation_max_abs_diff": max(r["head_tail_activation_max_abs_diff"] for r in rows),
        "first_activation_share_m_minus_1": q(np.array([r["first_activation_draws_share_m_minus_1"] for r in rows])),
        "first_activation_draws_range": [min(r["first_activation_draws_min"] for r in rows),
                                         max(r["first_activation_draws_max"] for r in rows)],
    }
    for key in ("cbar_over_m_d", "share_epochs_with_zero_draws", "E_over_cbar", "rho_DG_over_ref", "rho_H_over_ref"):
        s[key] = {"dataset_medians": q(med(key)), "pooled_p5_min": float(med(key, "p5").min()),
                  "pooled_p95_max": float(med(key, "p95").max())}
    s["epoch_total_over_mean"] = {
        "min_over_datasets": q(epoch("min")), "max_over_datasets": q(epoch("max")), "cv": q(epoch("cv")),
        **{k: q(epoch(k)) for k in ("epoch1", "min_t_ge_2", "max_t_ge_2", "min_t_ge_10", "max_t_ge_10")},
        "argmin_epochs": sorted({r["epoch_total_over_mean_t_ge_1"]["argmin_epoch"] for r in rows}),
        "argmax_epochs": sorted({r["epoch_total_over_mean_t_ge_1"]["argmax_epoch"] for r in rows})}
    return s


def main() -> None:
    _, paths = parse_args(__doc__.splitlines()[0])
    check_group(paths, "mechanism")
    rows = sorted(read_json(paths.data / "mechanism" / "calendar_audit.json"), key=lambda r: r["dataset"])
    s = summarize(rows)
    epochs = s["epoch_total_over_mean"]
    table = [
        (r"Mean scheduled count, $\bar c_i/(m d_i^{(v)})$", s["cbar_over_m_d"]["dataset_medians"]),
        (r"First activations drawing $m-1$ negatives (share of entries)", s["first_activation_share_m_minus_1"]),
        (r"Total count at $t=1$ / mean over $t\geq1$", epochs["epoch1"]),
        (r"Minimum total count over $t\geq10$ / mean over $t\geq1$", epochs["min_t_ge_10"]),
        (r"Maximum total count over $t\geq1$ / mean over $t\geq1$", epochs["max_over_datasets"]),
        (r"$E_i/(\bar c_i/n)$, equal to the D--G balance / reference", s["E_over_cbar"]["dataset_medians"]),
        (r"Repulsion-to-attraction balance of H / reference", s["rho_H_over_ref"]["dataset_medians"]),
    ]
    lines = [f"{label} & ${v['median']:.3f}$ & ${v['min']:.3f}$--${v['max']:.3f}$ \\\\" for label, v in table]
    write_rows(paths, "ch4_calendar_rows.tex", lines, "ch4_calendar_table.py")
    write_json(paths.tables / "ch4_calendar_summary.json", s)
    print(f"ch4_calendar_table.py: {s['n_datasets']} datasets")


if __name__ == "__main__":
    main()
