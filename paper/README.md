# Paper figures and tables

This directory regenerates every figure and table of the paper from frozen result
summaries, without rerunning any experiment. Figure 1 (method overview) is a
drawing and is not generated.

```bash
python -m pip install -e ".[paper]"      # numpy, scipy, pandas, matplotlib, scikit-learn
python scripts/paper/make_all.py         # paper/data -> paper/build, then verify
```

No GPU, dataset download or compiled extension is needed; the whole build takes
about a minute on a laptop. `make_all.py` finishes by running
`scripts/paper/verify.py`, which checks the output against the manuscript:

- every LaTeX table-row file and plot-data CSV must equal the manuscript version
  (comment lines and line endings aside);
- every manuscript figure must be produced;
- selected numbers quoted in the text must round to the printed value.

PDFs are not compared byte for byte because glyph shapes depend on the installed
fonts. The manuscript figures use Times New Roman; without it matplotlib falls
back to STIXGeneral, which changes glyphs and text widths slightly but not the
plotted data. `verify.py --compare-pdf <dir with the manuscript PDFs>` reports a
raster difference per figure when poppler's `pdftoppm` is available.

## Outputs

`paper/build/` (not tracked by git):

| Manuscript | Label | File | Builder |
| --- | --- | --- | --- |
| Sec. 4 figure | `fig:mechanism-effects` | `figures/figure_ch4_mechanism_effects.pdf` | `ch4_effect_figure.py` |
| App. table | `tab:mechanism-quality` | `tables/ch4_quality_rows.tex` | `ch4_tables.py` |
| App. table | `tab:mechanism-effects` | `tables/ch4_effect_rows.tex` | `ch4_tables.py` |
| App. table | `tab:mechanism-family` | `tables/ch4_family_rows.tex` | `ch4_tables.py` |
| App. table | `tab:mechanism-psweep` | `tables/ch4_psweep_rows.tex` | `ch4_psweep_table.py` |
| App. table | `tab:calendar-audit` | `tables/ch4_calendar_rows.tex` | `ch4_calendar_table.py` |
| Sec. 5 figure | `fig:e2e-results` | `figures/figure_BE_runtime_stability_merged_h166.pdf` | `ch5_main_figure.py` |
| Sec. 5 table | `tab:e2e-fidelity-common` | `tables/ch5_quality_common_median_rows.tex` | `ch5_tables.py` |
| App. table | `tab:benchmark-runtime` | `tables/ch5_runtime_rows.tex` | `ch5_tables.py` |
| App. table | `tab:benchmark-profiles` | `tables/ch5_profile_rows.tex` | `ch5_tables.py` |
| App. tables | `tab:benchmark-quality-absolute-{mean,median}` | `tables/ch5_quality_absolute_{mean,median}_rows.tex` | `ch5_tables.py` |
| App. table | `tab:benchmark-quality` | `tables/ch5_quality_rows.tex`, `tables/ch5_quality_paired_median_rows.tex` | `ch5_tables.py` |
| App. table | `tab:benchmark-collection-quality` | `tables/ch5_collection_rows.tex` | `ch5_tables.py` |
| App. tables | `tab:benchmark-stability`, `tab:benchmark-stability-pairs` | `tables/ch5_stability_rows.tex`, `tables/ch5_stability_pair_rows.tex` | `ch5_tables.py` |
| App. table | dataset list | `tables/ch5_dataset_rows.tex` | `ch5_tables.py` |
| App. figure | `fig:appendix-speedups` | `figures/figure_A_side_by_side.pdf` | `ch5_profile_figures.py` |
| App. figure | `fig:appendix-runtime-profiles` | `figures/figure_A_runtime_seeded_unseeded_side_by_side.pdf` | `ch5_profile_figures.py` |
| App. figure | `fig:appendix-cpu-seed-cost` | `figures/figure_C_cpu_seeded_time_cost.pdf` | `ch5_seed_cost_figure.py` |
| App. figure | `fig:e2e-fidelity` | `figures/figure_D_unseeded_fidelity.pdf` | `ch5_fidelity_figure.py` |
| App. figure | `fig:appendix-stage-share` | `figures/figure_F_ibumap_stage_share_{cpu,cuda}_{unseeded,seeded}.pdf` | `ch5_stage_share_figures.py` |
| Sec. 6 figure | `fig:braque-repeatability` | `figures/ch6_run1.pdf`, `ch6_run2.pdf`, `ch6_changes.pdf` | `ch6_panels.py` |
| App. table | `tab:braque-reuse` | `tables/ch6_reuse_rows.tex` | `ch6_reuse_table.py` |

The benchmark-environment table in Appendix B describes the experiment machine
and is written by hand. Each builder also writes the plotted values
(`figures/data/*.csv`, `tables/*_plot_data.csv`) and a summary JSON with the
statistics quoted in the text. Every builder runs on its own, e.g.
`python scripts/paper/ch5_tables.py`.

## Frozen data

`paper/data/` holds the result summaries the paper was built from. Values are
copied verbatim from the experiment outputs; only local paths and internal
names were removed. `MANIFEST.json` lists the SHA-256 of every file, and the
builders refuse to run on modified files.

| Group | Files | Content |
| --- | --- | --- |
| `e2e_benchmark/` | `runs.csv.gz` | 3,400 end-to-end runs: 71 datasets × {umap-learn, cuML, TorchDR, ibUMAP CPU, ibUMAP CUDA} × {seeded, unseeded} × 5 repeats (resource-skipped combinations excluded); wall times, ibUMAP stage timers, embedding hashes |
| | `quality.csv.gz` | per-run quality: trustworthiness, continuity, neighborhood preservation (`source=local`, all runs) and random-triplet accuracy, distance Spearman (`source=global`, 10^6 sampled triplets/pairs; unseeded all repeats, seeded repeat 1) |
| | `stability.csv.gz` | run-to-run stability, repeat 1 vs repeats 2–5 |
| | `datasets.csv.gz`, `bundle.json` | dataset sizes; benchmark scope, skip rules and metric configuration |
| `mechanism/` | `scores.csv.gz` | 1,593 runs: 59 datasets × 9 variants × optimizer seeds {42, 137, 2026}; five quality metrics |
| | `workloads.json` | retained datasets, the one excluded dataset and why, metric configuration |
| | `calendar_audit.json` | per-dataset replay of the umap-learn sampling calendar |
| `psweep/` | `quality.csv.gz`, `runs.csv.gz` | interpolation-order and stage-schedule sweep, 30 datasets, CPU and CUDA |
| `braque/` | `summary.csv.gz`, `pairwise_metrics.csv.gz`, `run_metrics.csv.gz`, `cost_ratios.csv.gz` | BRAQUE case study, four execution profiles × 5 runs |
| | `plot_data.npz`, `panels.json` | coordinates and HDBSCAN labels of the two plotted runs; rendering parameters |

The geodesic and persistent-homology scores computed during the experiments are
not part of the paper and were dropped.

## Using new results

A full rerun of the experiments (see the top-level README) writes summaries with
the same layout. Point the builders at them with

```bash
python scripts/paper/make_all.py --data-dir <new data> --build-dir <out>
```

Verification against the manuscript is skipped in that case, because timings
and unseeded runs are not expected to reproduce bit for bit.
