# Mechanism experiment (paper Section 4 and appendix)

Separates the effects that turn UMAP's online sampled optimizer into ibUMAP's
FFT-based expected field: synchronous updates, the expectation of negative
sampling, the degree-weighted dense field, the kernel cap, the FFT approximation
and ibUMAP's safeguards. All variants start from the same graph and spectral
initialization of each dataset and differ only in the optimizer. The Section 4
figure and the appendix tables `tab:mechanism-{quality,effects,family}` and
`tab:calendar-audit` are built by `scripts/paper/ch4_*.py` from the data this
experiment exports (frozen copy in `paper/data/mechanism/`).

## Variants

`configs/algorithms.yaml`; the letters are those of the paper.

| | ID | Update rule |
|---|---|---|
| A | `umap_learn` | umap-learn's `optimize_layout_euclidean` (online, sampled) |
| — | `ibumap_async` | ibUMAP's asynchronous CPU optimizer; must reproduce A bit for bit (implementation control, not scored) |
| B | `sync_sampled` | synchronous epochs: all forces of an epoch are accumulated, then applied |
| — | `sync_sampled_clip` | B + repulsion norm clip at `4*alpha` |
| C | `sync_expected` | B with the exact expectation of the scheduled uniform negative samples (per-pair component clip kept) |
| D | `sync_direct` | same attraction; repulsion = full degree-weighted sum `alpha * m*d_i/(2N) * sum_j K_ij (y_i - y_j)` |
| E | `sync_direct_capped` | D with the scalar kernel capped at 4, as in ibUMAP |
| F | `sync_fft` | E with the exact sum replaced by the CPU ibFFT approximation |
| G | `sync_fft_guarded` | F + repulsion norm clip at `4*alpha` and p99 degree damping |
| H | `ibumap_production` | production ibUMAP (head-wise attraction, own arithmetic) |

with `K_ij = 2*gamma*b / ((epsilon + |y_i-y_j|^2) (1 + a |y_i-y_j|^(2b)))`.
B–G call the same synchronous single-epoch routine of ibUMAP for the attraction
and replace only the repulsion buffer; the learning rate follows umap-learn's
post-epoch schedule. Exact sums accumulate in float64 per target (Numba, O(N)
memory). The contrasts in `configs/experiment.yaml` read: B−A synchrony, C−B
sampling to event expectation, D−C event weights to the degree-weighted raw
field (a change of formula, not denoising), E−D kernel cap, F−E FFT
approximation at matched update rule, G−F the safeguard bundle, H−G the
production residual.

Seeds 42, 137 and 2026 change only the optimizer's sampling. C–G do not sample,
so under a fixed initialization they are computed once per dataset and copied for
the other seeds (`independent_seed_replicate: false`).

## Inputs and suites

`01_prepare_fixed_inputs.py` builds, once per dataset, the umap-learn kNN graph
and fuzzy graph, the optimizer graph pruned for 200 epochs and the spectral
initialization with `IBUMAP.prepare_fixed_inputs` (seed 42, `n_jobs=1`,
parameters in `experiment.yaml`, section `fixed_inputs`). With the package
versions of `environments/cuda-linux-64.*` these are the inputs used for the
paper.

The 60 candidate datasets (`configs/datasets.yaml`) form three size suites:

| `--suite` | N | Datasets |
|---|---|---:|
| `le10k` (also `≤10k`, `<=10k`) | N ≤ 10,000 | 19 |
| `10k_50k` | 10,000 < N ≤ 50,000 | 28 |
| `50k_100k` | 50,000 < N ≤ 100,000 | 13 (12 retained) |

`c_elegans_embryogenesis_global_qc_annotated` is marked `excluded`: on the
paper's machine the asynchronous control differed from umap-learn at seed 2026,
reproducibly, so the whole dataset was dropped before quality evaluation. Every
stage skips it unless `--include-excluded` is given, and the paper reports the
remaining 59 datasets.

Each suite writes to `results/<suite>/` and runs full 200-epoch trajectories for
all variants; the dense variants refuse datasets above the suite's upper N bound.
The exact sums of C–E grow as N² per epoch, so `50k_100k` is by far the most
expensive suite; estimate its cost with the pilot run below first.

## Quality

`04_evaluate_quality.py` scores all variants except the control with the five
paper metrics on all rows (`configs/evaluation.yaml`): trustworthiness,
continuity and neighborhood preservation (k = 15; kNN on the GPU when one is
visible, as for the paper) and random triplet accuracy and distance Spearman
correlation (10^6 triplets / pairs, seed 42, CPU). `05_summarize_plot.py`
averages the paired seed differences within each dataset and then summarizes
datasets; seeds are not pooled as independent samples.

## Diagnostics not used in the paper

- `02_run_mechanism.py` keeps snapshots after 0, 10, 50, 100 and 200 updates and,
  for B–G, force norms, clip rates, degree damping and geometry every 5 epochs
  (`dynamics.csv`).
- `03_check_field_accuracy.py` evaluates, on the seed-42 snapshots after 0, 50
  and 200 updates of B, D, E, F and G, the exact field at 256 random targets and
  3 × 32 stress targets (high degree, dense region, radial tail) against all N
  sources, for the raw and the capped kernel, and compares CPU ibFFT at four
  grids `(p, box scale) = (1,1), (1,2), (3,1), (3,2)`. The FFT workspace estimate
  is capped at `fft.max_estimated_gib` (16 GiB); larger grids fail explicitly.
- `05_summarize_plot.py` plots dynamics, field errors and paired effects.

## Running

Run from this directory in the `ibumap-cuda` environment (umap-learn 0.5.12 is
required). Stages 01–04 print a plan unless `--run` is given.

```bash
python scripts/06_smoke_test.py                      # synthetic end-to-end check, a few minutes

for suite in le10k 10k_50k 50k_100k; do
  python scripts/00_check_environment.py --suite $suite --strict-data
  python scripts/01_prepare_fixed_inputs.py --suite $suite --run
  python scripts/02_run_mechanism.py --suite $suite --run
  python scripts/04_evaluate_quality.py --suite $suite --run
  python scripts/05_summarize_plot.py --suite $suite
done
python scripts/07_calendar_audit.py && python scripts/07_calendar_audit.py --summarize
python scripts/08_export_paper_data.py               # -> paper/rerun/mechanism/
```

Optional field diagnostics for a suite: `03_check_field_accuracy.py --suite <suite> --run`,
then `05_summarize_plot.py --suite <suite> --require-fields`.

Every stage accepts `--dataset` and `--seed` (repeatable) and `--algorithm`; the
summary must be given the same filters. A cost estimate for the dense variants
on one dataset (a separate 5-epoch run under `02_pilot/`, not a result):

```bash
python scripts/02_run_mechanism.py --suite 10k_50k --dataset tabula_sapiens_v2_heart --seed 42 \
  --algorithm sync_expected --algorithm sync_direct --algorithm sync_direct_capped --pilot --run
```

Numba uses `threads` from `experiment.yaml` (8; `--threads N` overrides it for a
run, and all stages must use the same value). FFT threads follow unless
`IBUMAP_FFT_THREADS` is set. Completed outputs are reused only when their key
(configuration, source code of `src/ibumap`, `scripts/common` and these scripts,
environment, input hashes) and all output hashes match; anything else stops with
a request for `--overwrite`. An interrupted trajectory restarts from epoch 0.
Tests: `python -m pytest tests -q`.

## Outputs

```text
results/01_fixed_inputs/<dataset>/neighbors_15/      kNN, fuzzy and optimizer graphs, initialization, metadata.json
results/<suite>/
  00_environment.json
  01_inputs/<dataset>/binding.json                    hashes verified by every later stage
  02_runs/<dataset>/<variant>__seed_<seed>/           embedding.npy, snapshots/, dynamics.csv, metadata.json
  02_pilot/                                           cost-estimate runs only
  03_fields/<dataset>/<variant>__seed_42/step_<t>/    targets, exact and FFT fields, errors.csv
  04_quality/<dataset>/<variant>__seed_<seed>/        scores.json, metadata.json
  05_summary/                                         completeness.csv, manifest.json, quality_long.csv,
                                                      paired_quality.csv, dataset_effects.csv, suite_summary.csv,
                                                      dynamics.csv, field_errors.csv, *.png
results/07_calendar_audit/                            per_dataset/<dataset>.json, summary.json, per_dataset.csv
results/logs/
```

Quality differences are signed so that positive means better.
`wall_seconds_provenance_only` in the run metadata includes diagnostic overhead
and is not a speed measurement.
