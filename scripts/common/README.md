# Shared experiment helpers (`scripts/common`)

Code shared by the experiment scripts under `experiments/`. Experiment scripts
put `scripts/` on `sys.path` and import the package as `common`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # experiments/<id>/scripts/<script>.py
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common.dataset_io import load_processed_dataset, select_row_indices
```

`ibumap` itself should be installed (`pip install -e .`, see
`environments/README.md`). If it is not, `common.paths.ensure_ibumap_importable()`
falls back to `src/` without the compiled extensions.

| Module | Purpose |
| --- | --- |
| `paths.py` | repository root and standard directories (`datasets/`, `experiments/`), repo-relative paths, `ensure_ibumap_importable()` |
| `config.py` | YAML loading (PyYAML, or a small built-in parser) and config-relative path resolution |
| `dataset_io.py` | processed-dataset loading from `datasets/processed/<id>/` and `datasets/catalog.json`, labels, deterministic row subsampling (`select_row_indices`) |
| `algorithm_adapters.py` | one call per method: `run_umap_learn`, `run_cuml_umap`, `run_torchdr_umap`, `run_ibumap`, fixed-graph optimization (`run_ibumap_optimize_from_graph`, `run_ibumap_optimize_from_prepared_graph`, `run_torchdr_umap_optimize_from_graph`) and `run_algorithm_entry` for config entries |
| `fixed_inputs.py` | kNN graph, fuzzy graph and spectral initialization through `IBUMAP.prepare_fixed_inputs*`; read/write of a fixed-input directory |
| `evaluation_cache.py` | the paper's five quality metrics with cached source/embedding states (see below) |
| `gpu_runtime.py` | CuPy/PyTorch synchronization and memory cleanup, GPU and machine information, array conversion |
| `run_metadata.py` | JSON read/write, package versions, git commit, timestamps |
| `task_grid.py` | dataset/algorithm entry selection and task grids |
| `embedding_diagnostics.py` | bounding-box and radial-tail summaries of embeddings, label sidecars |
| `plotting.py` | headless Matplotlib, embedding overview grids and heatmaps for the non-paper diagnostic plots |
| `callable_utils.py` | pass only the keyword arguments a callable accepts |

## Quality evaluation

`evaluation_cache.py` computes the metrics reported in the paper on top of
`ibumap.evaluation`:

| Metric | Key | Protocol (`PAPER_PROTOCOL`) |
| --- | --- | --- |
| Trustworthiness | `trustworthiness` | k = 15 |
| Continuity | `continuity` | k = 15 |
| Neighborhood preservation | `neighborhood_preservation` | k = 15 |
| Random triplet accuracy | `rta` | 10^6 triplets, seed 42 |
| Distance Spearman correlation | `distance_spearman` | 10^6 pairs, seed 42 |

Datasets with more than 10^6 rows are evaluated on 10^6 rows sampled without
replacement with seed 42 (`select_row_indices(n, 1_000_000, seed=42)`); the
caller applies this sampling. Unspecified values in an evaluation config fall
back to this protocol. Source-side states are cached per dataset and reused for
every embedding of that dataset; the cache key covers the full request, so a
changed parameter never reuses stale states.

```python
from common.evaluation_cache import (cache_store_from_config, request_from_config, scalar_scores,
                                     source_cache_key, embedding_cache_key, summarize_evaluation_scores)

request = request_from_config(config)                    # e.g. the experiment's evaluation.yaml
store = cache_store_from_config(config, base_dir=config_dir)
source_key = source_cache_key(dataset_id=name, n_rows=X.shape[0], n_features=X.shape[1], request=request)
result = store.evaluate_embedding(X, embedding, source_key=source_key,
                                  embedding_key=embedding_cache_key(embedding_id=run_id, embedding=embedding,
                                                                    request=request),
                                  request=request)
scores = scalar_scores(summarize_evaluation_scores(result.scores))   # {metric: float}
```

The store keeps two namespaces under `<root>/<version>/`: `source/` (source kNN,
sampled triplets and pairs with their source-side comparisons and distances,
shared by all embeddings of one sampled dataset) and `embedding/` (embedding kNN
of one embedding). Evaluation config options:

```yaml
device: auto              # GPU when CuPy/CUDA is visible, else CPU; "gpu" requires a GPU
gpu_batch_size: 128       # batch size of the pairwise GPU kernels
cache:
  root: ../results/02_evaluations/metric_cache
  delete_embedding_after_score: true   # drop embedding states once scored
  delete_source_after_dataset: true    # honored by store.cleanup_after_dataset(source_key)
timing:
  enabled: true           # per-stage timings in result.timings (adds GPU synchronization)
```

Scoring-only options such as `rank_low_memory_batch_size` do not invalidate
cached states. `evaluation_device_status()` reports the requested and resolved
device, which is useful in dry runs.

Geodesic and persistent-homology scores were computed during development but
are not part of the paper; requesting them raises `ValueError`.

## TorchDR

`run_torchdr_umap` maps the shared names `n_epochs` and `learning_rate` to
TorchDR's `max_iter` and `lr` (explicit TorchDR names take precedence) and
defaults to the FAISS backend of the `ibumap-torchdr` environment
(`backend: null` selects the pure PyTorch path). The adapter-only
`deterministic` flag switches cuDNN/cuBLAS/PyTorch between deterministic and
maximum-throughput execution for the duration of one call and restores the
previous settings afterwards.

TorchDR 0.4 has no public `simplicial_set_embedding`.
`run_torchdr_umap_optimize_from_graph(X, fuzzy_graph, init_embedding, params, device="cuda")`
injects a precomputed SciPy/CuPy CSR fuzzy graph through TorchDR's
`SparseAffinity` extension point, keeps the supplied initialization unscaled and
skips kNN/affinity construction, for the fixed-graph optimization-only
comparisons. It runs on one CPU or CUDA device; distributed execution and
duplicate processing are disabled because they would break the graph's row
alignment.

## Tests

```bash
python -m pytest scripts/common/tests -q
```

TorchDR tests that need `torch`/`torchdr` or a CUDA device are skipped when those
are unavailable; the others use fakes and run on CPU.
