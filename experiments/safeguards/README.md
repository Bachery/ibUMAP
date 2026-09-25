# Safeguard failure cases (paper Appendix B)

ibUMAP guards its synchronous update in two places besides the kernel cap: the
repulsive update is norm-clipped at `4*alpha`, and the attraction of high-degree
vertices is damped by `min(1, (d99 / d_i)^0.5)`. This experiment switches off one
of the two at a time and shows what happens without it. The appendix paragraph
"Safeguard failure cases", `tab:safeguard-cases` and `fig:safeguard-cases` are
built by `scripts/paper/ch4_safeguard_cases.py` from the data this experiment
exports (frozen copy in `paper/data/safeguards/`).

| Variant | Change from production ibUMAP (`ibumap_production` of the mechanism experiment) |
|---|---|
| `full` | none: kernel cap 4, repulsion-norm clip `4*alpha`, p99 attraction degree damping |
| `no_repulsion_clip` | `repulsion_clip_norm: null` |
| `no_attraction_damping` | `attraction_degree_damping: false` |

Datasets: CIFAR-10 (60,000 points), where the clip was introduced, and scDEED CART
(62,167 points), where the damping was introduced. Both start from the fixed
optimizer graph and spectral initialization of the mechanism experiment; all
other parameters are those of `ibumap_production` (200 epochs, deterministic CPU
optimizer, `configs/experiment.yaml`).

## Measures

- **Bounding-box area ratio**: area of the axis-aligned bounding box of all points
  divided by that of the core, the `ceil(0.99 n)` points nearest the
  coordinate-wise median. A layout without outliers has a ratio near 1; the
  mechanism experiment's production ibUMAP and umap-learn embeddings stay below
  2 on all 59 datasets (`03_reference_bbox.py`).
- **Far points**: points farther from the median than twice the core radius
  (the distance of the last core point).
- **Per-update trace**: `01_run_embeddings.py` wraps the CPU ibFFT call read-only
  and records, for every update, the mesh built for the current state, the norms
  of the applied update, attraction and repulsion, and the bounding-box statistics
  after the update. The wrapper only copies arrays; the test suite checks that the
  result is bitwise equal to an unwrapped run. Library diagnostics are not used
  because they disable the fused execution path.

With fixed inputs the optimizer draws no random numbers: the three seeds give
bitwise-identical embeddings (`case_summary.csv`, `seeds_bitwise_identical`), and
the paper shows seed 42.

## Running

Run from this directory in `ibumap-cuda` (CPU only; any environment with the
package versions of `environments/cuda-linux-64.*` works). First build the fixed
inputs of the two datasets with the mechanism experiment, and for the reference
run its three suites (the reference needs `ibumap_production` and `umap_learn`
only):

```bash
cd ../mechanism
python scripts/01_prepare_fixed_inputs.py --suite 50k_100k --dataset cifar10 --dataset scdeed_cart --run
# reference: the mechanism experiment's own runs (see its README), at least
#   python scripts/02_run_mechanism.py --suite <suite> --algorithm ibumap_production --algorithm umap_learn --run
cd ../safeguards

python scripts/01_run_embeddings.py --run     # 2 datasets x 3 variants x 3 seeds, under a minute each (CPU)
python scripts/02_summarize.py
python scripts/03_reference_bbox.py           # reads ../mechanism/results/<suite>/02_runs
python scripts/04_export_paper_data.py        # -> paper/rerun/safeguards/
python ../../scripts/paper/ch4_safeguard_cases.py --data-dir ../../paper/rerun --build-dir ../../paper/build_rerun
```

`01_run_embeddings.py` prints whether the fixed inputs equal those of the paper
(content digests in `configs/experiment.yaml`). With the paper's package versions
the mechanism experiment rebuilds them bit for bit, and `full` then reproduces
the mechanism experiment's `ibumap_production` embeddings bit for bit. Existing
embeddings are skipped; `--overwrite` recomputes them; `--dataset`, `--variant`
and `--seed` (repeatable) select a subset. Without `--run` the script prints the
plan. Tests: `python -m pytest tests -q`.

The frozen results were computed on a Linux x86_64 machine other than the one of
the other experiments, with 2 Numba/FFTW threads and the package versions of
`environments/cuda-linux-64.*`. The CPU optimizer is deterministic, and
`full` equals the paper's `ibumap_production` embeddings of both datasets bit for
bit. Wall times in the metadata are provenance only.

## Outputs

```text
results/embeddings/<dataset>/<variant>__seed_<seed>.npy     final embeddings
results/metadata/<dataset>/<variant>__seed_<seed>.json      parameters, input digests, embedding hash, environment
results/traces/<dataset>/<variant>__seed_<seed>.csv         per-update trace
results/snapshots/<dataset>/<variant>__seed_42/step_<t>.npy states after t updates (not used by the paper)
results/bbox_summary.csv, case_summary.csv, escaped_points.csv, summary.json     02_summarize.py
results/reference_bbox.csv, reference_bbox_summary.json                          03_reference_bbox.py
```

`04_export_paper_data.py` converts them to `paper/data/safeguards/`
(`runs`, `cases`, `traces`, `escaped_points`, `reference`, `embeddings.npz`,
`protocol.json`; see `paper/README.md`).

## Scope

Two datasets and one fixed graph and initialization each: the cases show the
failures the safeguards prevent; they do not calibrate the thresholds (`4*alpha`,
the 99th percentile, the exponent 0.5), which were fixed during the development
of an earlier version of the optimizer. No quality metrics are computed here.
