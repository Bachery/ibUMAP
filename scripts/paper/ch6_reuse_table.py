#!/usr/bin/env python3
"""Appendix table tab:braque-reuse: complete CPU reuse summary of the BRAQUE application.

Times are five-run medians in seconds; distance correlation rho, ARI and the changed-assignment
fraction summarize ten unordered run pairs; "Exact" means identical full coordinates and raw
labels across all five runs within the profile. Writes paper/build/tables/ch6_reuse_rows.tex
and ch6_reuse_summary.json (seeded cost ratios and fidelity values quoted in the text).
"""
from __future__ import annotations

from _common import check_group, parse_args, read_csv, write_json, write_rows

PROFILES = (("umap_unseeded", "UMAP unseeded"), ("umap_seeded", "UMAP seeded"),
            ("ibumap_unseeded", "ibUMAP unseeded"), ("ibumap_seeded", "ibUMAP seeded"))


def main() -> None:
    _, paths = parse_args(__doc__.splitlines()[0])
    check_group(paths, "braque")
    summary = {r["profile"]: r for r in read_csv(paths.data / "braque" / "summary.csv.gz")}
    assert set(summary) == {p for p, _ in PROFILES}
    lines = []
    for profile, label in PROFILES:
        r = summary[profile]
        assert r["run_count"] == "5" and r["pair_count"] == "10"
        exact = r["all_embeddings_identical"] == "True" and r["all_labels_identical"] == "True"
        lines.append(" & ".join([
            label, f"{float(r['embedding_fit_median']):.3f}", f"{float(r['embedding_plus_hdbscan_median']):.3f}",
            f"{float(r['distance_spearman_median']):.3f}", f"{float(r['ari_including_noise_median']):.3f}",
            f"{100 * float(r['assignment_disagreement_median']):.2f}\\%", "Yes" if exact else "No"]) + r" \\")
    write_rows(paths, "ch6_reuse_rows.tex", lines, "ch6_reuse_table.py")
    ratios = read_csv(paths.data / "braque" / "cost_ratios.csv.gz")
    write_json(paths.tables / "ch6_reuse_summary.json", {
        "cost_ratios": [{k: (float(v) if k in ("numerator_seconds", "denominator_seconds", "ratio") else v)
                         for k, v in r.items()} for r in ratios],
        "seeded_fidelity": {p: {"trustworthiness": float(summary[p]["trustworthiness_median"]),
                                "high_dim_neighbor_overlap": float(summary[p]["high_dim_neighbor_overlap_median"])}
                            for p in ("umap_seeded", "ibumap_seeded")},
    })
    print("ch6_reuse_table.py: wrote ch6_reuse_rows.tex")


if __name__ == "__main__":
    main()
