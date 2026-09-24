# ibUMAP

Reference implementation for the paper *ibUMAP: Coherent and Scalable Field
Evaluation for UMAP Optimization*.

> **Note:** this README is a placeholder. Installation, reproduction commands,
> dataset preparation and hardware notes still need to be written.

## Installation

```bash
pip install -e .
```

Optional backends:

```bash
pip install -e ".[cuda]"      # CUDA runtime (CuPy + cuML); see environments/README.md
pip install -e ".[metal]"     # Apple Silicon (MLX)
pip install -e ".[datasets]"  # dataset download / preparation scripts
```

## Quick start

```python
from ibumap import IBUMAP

model = IBUMAP(n_components=2, algorithm="ibumap", device="cpu")
embedding = model.fit_transform(X)
```

## Paper figures and tables

Every figure and table of the paper can be rebuilt from the frozen result
summaries in `paper/data/`, without a GPU or rerunning experiments:

```bash
pip install -e ".[paper]"
python scripts/paper/make_all.py
```

See [`paper/README.md`](paper/README.md) for the mapping from manuscript
labels to builders and data files.

## Layout

```
src/ibumap/
├── api.py            # public IBUMAP estimator
├── config.py         # configuration dataclasses
├── graph/            # kNN graph construction (CPU / cuML)
├── init/             # spectral initialization
├── kernels/          # ibFFT field-evaluation kernels (CPU / CUDA / Metal)
├── optimizers/       # ibUMAP, UMAP and tFDP optimizers
├── pipeline/         # per-device orchestration
└── evaluation/       # embedding-quality metrics
```

## License

Intentionally unset for now. The license will be decided after the
third-party code provenance audit (see the migration record kept with the
research repository).
