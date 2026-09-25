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

## Rerunning the experiments

The experiments behind the paper live under `experiments/`; each has a README
with its protocol, commands and outputs. Environments are described in
`environments/README.md`, datasets in `datasets/README.md`.

| Directory | Paper | Result |
| --- | --- | --- |
| `experiments/mechanism/` | Section 4 and appendix | optimizer variants A–H on fixed inputs, 59 datasets |
| `experiments/psweep/` | appendix `tab:mechanism-psweep` | interpolation order p = 1, 2, 3 on 30 datasets |
| `experiments/e2e_benchmark/` | Section 5 and Appendix B | end-to-end runtime, quality and stability, 71 datasets |
| `experiments/braque/` | Section 6 and appendix | BRAQUE rerun repeatability and seeded cost |

Each experiment ends with an export step that writes its summaries in the layout
of `paper/data/` to `paper/rerun/`; then

```bash
python scripts/paper/make_all.py --data-dir paper/rerun --build-dir paper/build_rerun
```

rebuilds the figures and tables from the new results. Shared helpers are in
`scripts/common/` (see its README).

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
