# Interpolation-order sweep (paper appendix, `tab:mechanism-psweep`)

Compares fixed interpolation orders p = 1, 2, 3 of the FFT repulsion (and a
p1→p2→p3 schedule) on CPU and CUDA with everything except the optimizer held
fixed. The appendix table is built by `scripts/paper/ch4_psweep_table.py` from the
summaries this experiment writes (frozen copy in `paper/data/psweep/`); it reports
p = 2 and p = 3 against p = 1. The schedule variants are part of the experiment
but not of the paper.

## Design

- 30 datasets from 6 families and 5 size bins (150–129,062 rows,
  `configs/datasets.yaml`). `00_validate_experiment.py` checks every entry
  against `datasets/catalog.json` and the processed arrays and rejects datasets
  above 200,000 rows.
- The kNN graph, fuzzy graph and spectral initialization are built once per
  dataset (`01_prepare_fixed_inputs.py`, seed 42, `n_neighbors=15`). All eight
  device/variant combinations read the same fixed inputs, so the comparison
  boundary is `IBUMAP.optimize_from_graph`.
- Eight variants (`configs/algorithms.yaml`): CPU and CUDA × fixed p = 1, 2, 3 and
  the schedule, which runs p = 1 for epochs [0, 180), p = 2 for [180, 190) and
  p = 3 for [190, 200) through an explicit `FFTConfig.interpolation_schedule`.
  `p2m_mode=auto` is required because the CPU atomic/segmented and CUDA
  block-atomic P2M paths support only p = 1. All variants use `n_epochs=200` and
  `deterministic=true`.
- Three repeats with paired seeds 42/43/44: the eight tasks of a dataset and
  repeat share the seed, and their order is shuffled deterministically.
- Every task first runs a warmup worker process (same variant, at most 512 rows),
  then a fresh measurement process. `optimization_wall_s` covers the full
  `optimize_from_graph` call including CUDA synchronization, but not input
  loading, warmup or serialization. ibUMAP's memory diagnostics sample every
  20 ms: the CPU measure is the process RSS peak, the CUDA measure the NVML
  process memory (CuPy pool size when NVML is unavailable).
- Quality: trustworthiness, continuity and neighborhood preservation (k = 15) on
  all rows (`configs/evaluation.yaml`). Stability: repeat 1 against repeats 2 and 3
  on up to 10,000 points (Procrustes RMS, pairwise-distance Spearman,
  15-NN overlap).

## Running

Run from this directory in the `ibumap-cuda` environment. The expensive scripts
only list their tasks unless `--run` is given.

```bash
python scripts/00_validate_experiment.py --strict --write-json
python scripts/01_prepare_fixed_inputs.py --run
python scripts/02_run_benchmark.py --run
python scripts/03_run_evaluations.py --run
python scripts/04_summarize_results.py --strict
python scripts/05_plot_results.py                # diagnostic plots (not in the paper)
python scripts/06_export_paper_data.py           # -> paper/rerun/psweep/
```

CPU and CUDA tasks can run separately and in any slice; results do not depend on
the slicing:

```bash
python scripts/02_run_benchmark.py --run --device cpu
python scripts/02_run_benchmark.py --run --device cuda --variant p3 --repeat 1
python scripts/02_run_benchmark.py --run --dataset cifar10 --overwrite
```

A worker killed by the out-of-memory killer is recorded as a failure and the
parent continues. Without `--overwrite`, a task whose record matches the current
configuration hash is skipped and a stale one stops the run.

Tests of the configuration contract, the memory-diagnostics parser, the
stability alignment and the plots (no data needed):

```bash
python -m pytest tests -q
```

## Outputs

```text
results/00_validation.json
results/01_fixed_inputs/<dataset>/neighbors_15/
results/02_embeddings/{per_run,metadata,failures,memory}/<dataset>/<algorithm>__repeat_<NN>.*
results/03_evaluations/{per_run,failures}/<dataset>/, sample_indices/, metric_cache/
results/04_summaries/
    completeness.csv  runs.csv  runtime_memory_summary.csv  quality_scores.csv  quality_summary.csv
    stability.csv  stability_summary.csv  paired_schedule_runtime_memory.csv  paired_schedule_quality.csv
    manifest.json
results/05_plots/{runtime_memory_scaling,runtime_memory_distribution,evaluation_quality,embedding_stability}.{png,svg}
```

`paired_schedule_*` pair the schedule with each fixed order on the same dataset,
device and repeat.
