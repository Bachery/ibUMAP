# End-to-end benchmark (paper Section 5, Appendix C)

Measures the end-to-end time, quality and run-to-run stability of five UMAP
implementations on 71 datasets. The paper's Section 5 figure and tables and the
Appendix C tables and figures are built from the summaries this experiment
writes (`scripts/paper/ch5_*.py`, frozen copy in `paper/data/e2e_benchmark/`).

## Scope

`configs/datasets.yaml` lists the 71 datasets (prepared under
`datasets/processed/`, see `datasets/README.md`). `configs/algorithms.yaml` has
ten variants:

| Implementation | Device | Seeded | Unseeded |
| --- | --- | --- | --- |
| umap-learn | CPU | `umap_learn_seed_42` | `umap_learn_seed_none` |
| cuML UMAP | CUDA | `cuml_umap_seed_42` | `cuml_umap_seed_none` |
| ibUMAP | CPU | `ibumap_cpu_seed_42` | `ibumap_cpu_seed_none` |
| ibUMAP | CUDA | `ibumap_cuda_seed_42` | `ibumap_cuda_seed_none` |
| TorchDR UMAP | CUDA | `torchdr_umap_seed_42` | `torchdr_umap_seed_none` |

All variants share `common_umap_params` (`n_neighbors=15`, `min_dist=0.1`,
`n_epochs=200`, ...). A seeded variant passes `random_state=42`; an unseeded one
omits it. umap-learn and cuML may only override `random_state`, so they run with
their library defaults (cuML chooses its own `build_algo` and
`force_serial_epochs`). ibUMAP uses its automatic production paths; the seeded
ibUMAP and TorchDR variants also set `deterministic: true`. TorchDR uses the FAISS
backend on CUDA with distributed and compiled execution disabled; seeded runs use
deterministic PyTorch/cuDNN/cuBLAS settings, unseeded runs PyTorch's fast mode.
"Seeded" and "unseeded" are grouping labels, not claims of bitwise determinism or
maximum speed.

Each task runs five repeats (3,550 nominal runs). umap-learn, ibUMAP CPU and
TorchDR are not run on five datasets that exhaust host or GPU memory
(`gist_960_euclidean`, `google_news_300d` and three `whole_mouse_brain_merfish_*`
sections). The task builder, evaluation and summaries apply this rule
consistently, so the 150 omitted runs are reported as resource-skipped rather than
missing; 3,400 runs are expected.

## What is timed

`e2e_call_wall_s` covers model construction, nearest-neighbor search, graph
construction, initialization, optimization, CUDA synchronization and copying the
embedding to host memory. Dataset loading, hashing, warmup, serialization,
evaluation and plotting are outside the timed region.

The controller (`01_run_e2e_benchmark.py`) runs every task in a fresh worker
process and waits for it before starting the next, so allocator, FFTW, Numba, RMM
and CUDA state never carries over between tasks. Each worker first runs the
configured warmup (one call on a 512-row subset). Task order is shuffled within
each repeat block with a fixed seed. A worker killed by a signal is recorded as a
failure (exit code 137 / `SIGKILL` is flagged as a likely out-of-memory kill) and
the controller continues.

## Result identity

Every successful run record stores:

- a benchmark configuration hash (datasets, algorithms, timing policy);
- a protocol hash over the controller, adapters, shared helpers and, for ibUMAP,
  the complete `src/ibumap` source (the Cython-generated
  `_cuml_graph_ext.cpp` is excluded because it depends on the build host);
- a task signature (dataset, algorithm, repeat, `n_neighbors`, requested
  parameters, protocol hash, feature-file SHA-256);
- package/runtime-variable and hardware signatures, and the embedding SHA-256.

A rerun skips a task only when all of these still match; otherwise it stops and
asks for `--overwrite`. Evaluation records repeat the run contract and add the
evaluation request and runtime signature. The strict summary rejects mixed
hardware, mixed per-algorithm runtimes and mixed evaluation environments.

CUDA ibUMAP requires the cuML graph extension (`ibumap.graph._cuml_graph_ext`,
built with `IBUMAP_BUILD_CUML_GRAPH_EXT=1`, see `environments/README.md`). The
environment check and every CUDA ibUMAP worker fail before timing if it cannot
be imported; a silent fallback to the generic graph builder is not allowed.

## Quality evaluation

`02_run_evaluations.py` scores every successful embedding with the paper's five
metrics (`configs/evaluation.yaml`, protocol in `scripts/common/README.md`):
trustworthiness, continuity and neighborhood preservation with k = 15, and random
triplet accuracy and distance Spearman correlation with 10^6 sampled triplets /
pairs (seed 42). Datasets with more than 10^6 rows are scored on 10^6 rows sampled
with seed 42, the same rows for every algorithm and repeat. The kNN searches of
the local metrics run on the GPU when one is visible (`device: auto`, as for the
paper); the global metrics always run on the CPU. An embedding that is bitwise
identical to an already scored repeat reuses its scores.

`03_summarize_results.py` also computes repeat stability: for each variant and
dataset, repeat 1 is compared with repeats 2–5 on up to 10,000 sampled points
(Procrustes RMS, pairwise-distance Spearman on 20,000 pairs and 15-NN overlap).

## Running

Run from this directory. The benchmark and evaluation scripts are dry runs unless
`--run` is given. The CUDA and TorchDR stacks live in separate environments
(`environments/`); both write to the same `results/` tree.

```bash
./check.sh                          # task grid, data availability, resource skips (no writes)

conda activate ibumap-cuda
./run.sh                            # 8 umap-learn / cuML / ibUMAP variants

conda activate ibumap-torchdr
./run_torchdr.sh                    # 2 TorchDR variants

conda activate ibumap-cuda
python scripts/02_run_evaluations.py --run
python scripts/03_summarize_results.py --strict
python scripts/07_export_paper_data.py          # -> paper/rerun/e2e_benchmark/
```

Then build the paper outputs from the new summaries (after the other experiments
have exported their groups to the same directory):

```bash
python ../../scripts/paper/make_all.py --data-dir ../../paper/rerun --build-dir ../../paper/build_rerun
```

The builders expect the complete grid (3,400 runs) and stop on a partial one.

Useful options of `01_run_e2e_benchmark.py`, `02_run_evaluations.py` and
`03_summarize_results.py`: `--dataset`, `--algorithm`, `--profile seeded|unseeded`
and `--repeat` (all repeatable) restrict the grid; `--overwrite` replaces stale
results. A quick CPU check:

```bash
python scripts/00_check_environment.py --dataset iris --algorithm umap_learn_seed_42 --probe --strict
python scripts/01_run_e2e_benchmark.py --dataset iris --algorithm umap_learn_seed_42 --repeats 1 --run
python scripts/02_run_evaluations.py --dataset iris --algorithm umap_learn_seed_42 --repeat 1 --run
python scripts/03_summarize_results.py --dataset iris --algorithm umap_learn_seed_42 --repeat 1 --strict
```

A filtered strict pass is written as a scoped bundle and is not published; only
an unfiltered strict pass over the full grid updates
`results/03_summaries/published.json`, which the plotting and export scripts read.

Diagnostic plots (not in the paper), from the published bundle:

```bash
python scripts/04_plot_embeddings.py            # per-run embeddings, per-dataset grids, stability
python scripts/05_plot_time_comparisons.py      # runtime scaling, speedups, ibUMAP stage breakdowns
python scripts/06_plot_metrics_comparisons.py   # quality distributions, paired deltas, ranks, Pareto
```

## Outputs

```text
results/01_embeddings/{per_run,metadata,failures,warmups}/<dataset>/<run_id>.*
results/02_evaluations/{per_run,failures}/<dataset>/<run_id>.json, sample_indices/, metric_cache/
results/03_summaries/{bundles,scoped_bundles,validation_failures,drafts}/<bundle id>/
    run_summary.csv  runtime_summary.csv  speedup_pairs.csv  quality_long.csv  quality_summary.csv
    quality_pairs.csv  stability_summary.csv  dataset_summary.csv  runs.jsonl  completeness.json
results/03_summaries/published.json
results/04_embedding_plots/  results/05_time_plots/  results/06_metrics_plots/
logs/<script>_<timestamp>_pid<pid>.log
```

Run IDs have the form `<algorithm>__nn_<n_neighbors>__repeat_<rr>`, e.g.
`ibumap_cuda_seed_none__nn_15__repeat_01`. Paths inside the records are relative
to the repository root. Every invocation of scripts `00`–`06` writes its complete
terminal output (stdout, stderr, warnings, tracebacks, worker output) to one log
file under `logs/`.
