# BRAQUE live computation demo

A local browser interface that runs a fresh dimensionality reduction and
clustering on request:

```text
fixed subset of the frozen BRAQUE input → ibUMAP or umap-learn → HDBSCAN → browser plot
```

It is a demonstration, not part of the paper's results, and writes nothing to
disk. It needs the frozen input of stages 01–04 (`data/processed/L2/`). From
`experiments/braque`:

```bash
python scripts/09_serve_live_demo.py --check      # validate dependencies and data
python scripts/09_serve_live_demo.py              # serve on http://127.0.0.1:8010/
```

The page fits one screen (browser viewport of 1280×720 or larger); the run history, the effective
configuration of the last run and the method notes are below it.

## Controls

| Control | Range | Default |
|---|---|---|
| Algorithm | ibUMAP, umap-learn | ibUMAP |
| Cells in subset | 1,000, 2,500, 5,000, 10,000, 20,000 | 2,500 |
| `n_neighbors` | 2 – 200 | 50 |
| `min_dist` | 0 – 0.99 | 0 |
| HDBSCAN `min_cluster_size` (`min_samples` is set to the same value) | 2 – 500 | 10 |
| Deterministic mode and seed | on/off, 0 – 2³¹−1 | off, 42 |

Defaults are the case-study values; the HDBSCAN default is the case-study rule
`max(int(0.00005 N), 10)`, which is 10 for every offered subset size. Fixed
settings: 200 epochs, spectral initialisation, Euclidean metric, HDBSCAN
`cluster_selection_epsilon = 0.1` with `eom` selection.

Subsets are prefixes of one fixed permutation (seed `20260820`), so the cells do
not change when the algorithm seed changes, and a smaller subset is contained in
every larger one. With deterministic mode enabled the selected seed is passed as
`random_state`; umap-learn then uses one worker and ibUMAP its deterministic
execution path. Only one job runs at a time.

## Change vs previous run

From the second run on, each run is compared with the previous completed run of
the same browser session. Clusters are matched one-to-one by maximum overlap
(Hungarian assignment, noise kept as noise; `match_labels` / `label_metrics` in
`scripts/_common.py`, as in `06_evaluate_repeatability.py`), and the page shows
the share of cells whose matched cluster differs, including changes to or from
noise. The card becomes more strongly coloured as the share grows (full
intensity at 50 %). When the two runs use different cell counts, the comparison
uses the cells both contain.

The share includes cluster splits and merges and is not a misclassification
rate. After matching, clusters keep the colour of their matched cluster in the
previous run; "Highlight changed cells" marks the cells that changed.

## Command-line smoke test

Without starting the server:

```bash
python scripts/09_serve_live_demo.py --smoke-test --algorithm ibumap --cells 1000 --deterministic --seed 42
python scripts/09_serve_live_demo.py --smoke-test --algorithm ibumap --cells 1000 \
    --n-neighbors 30 --min-dist 0.1 --min-cluster-size 15 --repeats 2   # also prints the comparison
```
