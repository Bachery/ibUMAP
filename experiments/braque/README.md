# BRAQUE case study (paper Section 6 and appendix)

BRAQUE (Dall'Olio et al., 2023) clusters spatial single-cell proteomics data with
HDBSCAN on a 2-D UMAP embedding. This case study asks how much of that partition
changes when the same UMAP analysis is simply run again, and what a seeded,
bitwise-repeatable run costs with umap-learn and with ibUMAP. The Section 6 figure
`fig:braque-repeatability` and the appendix table `tab:braque-reuse` are built by
`scripts/paper/ch6_panels.py` and `ch6_reuse_table.py` from the data this
experiment exports (frozen copy in `paper/data/braque/`).

- Paper: L. Dall'Olio, M. Bolognesi, S. Borghesi, G. Cattoretti, G. Castellani.
  BRAQUE: Bayesian Reduction for Amplified Quantization in UMAP Embedding.
  *Entropy* 25(2):354, 2023. https://doi.org/10.3390/e25020354 (CC BY 4.0;
  `literature/braque.bib`)
- Data: Mendeley Data, https://doi.org/10.17632/j8xbwb93x9.1, sample `L2`

## Input (stages 01–04)

```bash
python scripts/01_download_data.py --run                 # BRAQUE-RawCSVdata.zip, ListOfPrimaryAntibodies.xlsx
python scripts/02_check_and_prepare_data.py --run --sample L2
python scripts/03_prepare_braque_features.py --run
python scripts/04_run_and_freeze_lns.py --run            # Lognormal Shrinkage, checkpointed per marker
```

Stage 02 reads the antibody workbook with pandas, which needs `openpyxl`
(included in `environments/cuda-linux-64.yml`; when the environment is created
from the lock file, install it as shown in `environments/README.md`).

01 verifies the official sizes and SHA-256 of the downloads. 02 removes the CSV
export index and keeps raw marker intensities; the public CSVs contain no spatial
coordinates, and none are invented. 03 selects markers with a documented proxy of
BRAQUE's paper-specific reference table (exact name matching against the public
antibody workbook, dropping missing or `IGNORE` significance), because that table
is not public. 04 runs BRAQUE's Lognormal Shrinkage (Bayesian Gaussian mixtures,
seed 42) and robust standardization and freezes the UMAP input
`data/processed/L2/lns/umap_input.npy` (56,962 cells × 62 features) with a
checksum ledger. The paper used the input with SHA-256
`8e3698f8c350dac7489edf2be8257c6d4c1a499d1ba6897339f23c92bf84abb8`; stage 05
reports whether the rebuilt input matches. See `data/README.md`.

This is therefore a BRAQUE-derived workflow on public data, not an exact
reproduction of the BRAQUE paper, and there are no expert phenotype labels:
the results concern the repeatability of computational partitions, not their
biological correctness.

## Protocol (stages 05–06)

| Profile | Estimator | Seed | Jobs |
|---|---|---|---|
| `umap_unseeded` | umap-learn | none | 8 |
| `umap_seeded` | umap-learn | `random_state=42` | 1 |
| `ibumap_unseeded` | ibUMAP CPU | none | 8 |
| `ibumap_seeded` | ibUMAP CPU | `random_state=42`, `deterministic=True` | 8 |

- Embedding: 2-D, `n_neighbors=50`, `min_dist=0`, Euclidean, spectral
  initialization, 200 epochs; every run rebuilds graph, initialization and layout.
- HDBSCAN: `min_cluster_size = min_samples = max(int(0.00005 N), 10)`,
  `cluster_selection_epsilon=0.1`, `eom`, one job.
- Five repeats per profile, each in a fresh worker process; the four profiles are
  interleaved in a shuffled order within each repeat block (fixed schedule seed).
  BLAS/OpenMP/Numba thread counts are set to `--threads` (8).
- Each worker first warms up the same configuration on a fixed 5,000-cell subset,
  then times `fit_transform` of a new estimator on all cells and HDBSCAN's
  `fit_predict`. Imports, loading, estimator construction, I/O and diagnostics are
  outside the timers; all five measurements are kept. `embedding_plus_hdbscan` is
  the sum of the two calls, not the full BRAQUE runtime. Peak RSS covers the whole
  worker including warmup.
- Evaluation (06): label agreement on all cells for all 10 run pairs per profile
  (ARI with noise as a label, ARI on cells non-noise in both runs, AMI, disagreement
  after maximum-overlap one-to-one cluster matching with noise kept as noise,
  cluster→cluster changes, both noise transitions); geometry on fixed samples
  (15-NN overlap on 2,048 cells, distance Spearman on 200,000 pairs); quality
  (trustworthiness on 3,000 cells, input-neighbor overlap on 2,048 cells). Pair
  values are summarized by median and range; pairs share runs and are not
  independent samples. The seeded profiles must reproduce coordinates and labels
  bit for bit (exit code 2 otherwise).

The figure shows the first two scheduled unseeded umap-learn runs (no selection
by effect size): run 2 is rotated/reflected and translated onto run 1 for display
only (no scaling; HDBSCAN runs on the original coordinates) and its clusters are
matched to run 1 over all cells.

## Running

From this directory in the `ibumap-cuda` environment (CPU only; needs `hdbscan`).
Nothing is overwritten: a batch, analysis or figure ID that exists is refused.

```bash
python scripts/05_run_repeatability.py --run-id l2_v1 --threads 8 --repeats 5          # dry run: checks only
python scripts/05_run_repeatability.py --run-id l2_v1 --threads 8 --repeats 5 --run    # 20 workers
python scripts/06_evaluate_repeatability.py --run-id l2_v1 --run
python scripts/07_plot_repeatability.py --run-id l2_v1 --run    # diagnostic figures (not in the paper)
python scripts/08_export_paper_data.py --run-id l2_v1           # -> paper/rerun/braque/
```

After an interruption, rerun stage 05 with the same options and `--resume`:
verified runs are skipped and failed ones retried in a new `attempt_<n>/`
directory. Resuming is refused if the configuration, inputs, source code
(`src/ibumap`, the runner), key package versions or the machine changed.
`--input-dir` / `--cells-file` point stage 05 to an input elsewhere; the batch
keeps its own copy under `inputs/`, so it can be evaluated on another machine.

Small check (not for paper numbers; outputs are marked as a smoke subset):

```bash
python -m pytest tests -q
python scripts/05_run_repeatability.py --run-id smoke_v1 --max-cells 256 --warmup-cells 256 --repeats 2 --threads 2 --run
python scripts/06_evaluate_repeatability.py --run-id smoke_v1 --quality-cells 256 --structure-cells 256 --distance-pairs 5000 --run
python scripts/07_plot_repeatability.py --run-id smoke_v1 --run
python scripts/08_export_paper_data.py --run-id smoke_v1 --output /tmp/braque_smoke
```

A live browser demo of the same pipeline on a cell subset is in `demo_live/`
(`scripts/09_serve_live_demo.py`).

## Outputs

```text
data/raw/, data/processed/L2/{markers_raw.npy, cells.csv.gz, features_selected/, lns/}
results/repeatability/<run-id>/
  experiment.json, status.json                   frozen contract, schedule, input/source hashes
  inputs/                                        copy of the input actually used, SHA256SUMS
  runs/<profile>/repeat_<r>/attempt_<n>/         embedding.npy, labels.npy, run.json, effective_config.json,
                                                 internal_timings.json, cluster_sizes.csv, logs, SHA256SUMS
  analysis/<analysis-id>/                        summary.csv, run_metrics.csv, pairwise_metrics.csv,
                                                 cost_ratios.csv, validation.json, SUMMARY.md, SHA256SUMS
  figures/<figure-id>/                           diagnostic figures of stage 07
```
