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

The default subset has 2,500 cells (1,000 to 20,000 selectable). Subsets are
prefixes of one fixed permutation (seed `20260820`), so the cells do not change
when the algorithm seed changes. With deterministic mode enabled the selected
seed is passed as `random_state`; umap-learn then uses one worker and ibUMAP its
deterministic execution path. HDBSCAN always uses the case study's configuration
(`min_cluster_size = min_samples = max(int(0.00005 N), 10)`, `cluster_selection_epsilon=0.1`).
Only one job runs at a time.

Command-line smoke test without starting the server:

```bash
python scripts/09_serve_live_demo.py --smoke-test --algorithm ibumap --cells 1000 --deterministic --seed 42
```
