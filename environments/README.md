# Environments

Run every command below from the repository root unless stated otherwise.

## Overview

| Environment | Platform | Used for | Specification |
| --- | --- | --- | --- |
| `ibumap-cuda` | Linux x86-64 + NVIDIA GPU | **All experiments in the paper**: ibUMAP on CPU and CUDA, umap-learn, cuML UMAP, tFDP, the HDBSCAN case study, and dataset preparation | [`cuda-linux-64.yml`](./cuda-linux-64.yml) |
| `ibumap-torchdr` | Linux x86-64 + NVIDIA GPU | TorchDR UMAP baseline (FAISS backend) only | [`torchdr-linux-64.yml`](./torchdr-linux-64.yml) |
| `ibumap-metal` | Apple Silicon macOS | ibUMAP on MLX/Metal | [`metal-osx-arm64.yml`](./metal-osx-arm64.yml) |

All paper results were produced on a single Linux machine: CPU and CUDA
timings in `ibumap-cuda`, TorchDR timings in `ibumap-torchdr`. The Metal path
is implemented and smoke-tested, but it is **not evaluated in the paper** and
has not been run at scale.

TorchDR lives in its own environment because its CUDA 12.6 PyTorch wheel and
the pinned `faiss-gpu` 1.8.0 build do not co-exist with the RAPIDS stack.
`ibumap` is not installed there.

Each environment has three files:

| File | Content |
| --- | --- |
| `<env>.yml` | Authoritative, portable specification: conda packages, pip packages and saved environment variables |
| `<env>.conda.lock` | Exact conda package set (`conda list --explicit`) **of the environment that produced the paper results** on the experiment machine. Does not contain pip packages or environment variables |
| `<env>.pip.txt` | Every Python distribution in that same environment as `name==version` (`pip list --format=freeze`), for auditing |

The Linux lock files were exported from the environments the experiments ran
in, not from a new solve of the YAML files. A new solve reproduces every
direct dependency exactly, but conda may pick newer builds of transitive
packages (for example the OpenMP, TBB and OpenBLAS runtimes). For timing
comparisons with the paper, create the environment from the lock file.

One entry in `cuda-linux-64.conda.lock`, `ca-certificates`, comes from
Anaconda's `pkgs/main` channel rather than conda-forge; it was pulled in
by an update of the experiment environment. It only provides the TLS root
certificates and does not affect any computation. If your setup blocks that
channel, replace the line with any conda-forge `ca-certificates` build.

All environments install `pyFFTW` 0.15.1 from PyPI. Do not replace it with the
conda-forge build, which reports `0.0.0` as its package version.

## CPU only, without conda

The CPU path needs no conda environment. Any Python >= 3.11 with a C++
compiler works:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[datasets]"
python scripts/smoke_test.py cpu
```

This is convenient for trying the method and for preparing datasets, but the
CPU numbers in the paper were measured in `ibumap-cuda` (dependency versions
differ).

## Prerequisites

- A recent conda (or mamba) with the libmamba solver.
- Linux GPU environments: an NVIDIA driver visible through `nvidia-smi` and
  compatible with CUDA 12.x.
- Metal environment: Apple Silicon and a native arm64 conda installation (not
  Rosetta/x86-64).

## Create an environment

```bash
# Linux, CUDA/cuML (paper environment)
conda env create -f environments/cuda-linux-64.yml
conda activate ibumap-cuda

# Linux, TorchDR baseline
conda env create -f environments/torchdr-linux-64.yml
conda activate ibumap-torchdr

# macOS, MLX/Metal
conda env create -f environments/metal-osx-arm64.yml
conda activate ibumap-metal
```

`-n <name>` overrides the environment name, e.g. to test a new solve next to an
existing environment.

### Exact conda package set from the lock file

To recreate the exact conda package set of the paper environment, use the
lock file and then add the parts a lock file cannot hold:

```bash
# ibumap-cuda
conda create -n ibumap-cuda --file environments/cuda-linux-64.conda.lock
conda env config vars set -n ibumap-cuda \
  'LD_LIBRARY_PATH=$ORIGIN/../lib:$ORIGIN/../targets/x86_64-linux/lib'
conda activate ibumap-cuda
python -m pip install --no-deps pyfftw==0.15.1
python -m pip install --no-deps pyreadr==0.5.6   # optional: scripts/datasets/scdeed/ only

# ibumap-torchdr
conda create -n ibumap-torchdr --file environments/torchdr-linux-64.conda.lock
conda env config vars set -n ibumap-torchdr 'LD_PRELOAD=$ORIGIN/../lib/libstdc++.so.6'
conda activate ibumap-torchdr
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  pyfftw==0.15.1 torch==2.12.1+cu126 torchdr==0.4
```

Compare the result with the corresponding `<env>.pip.txt`.

## Install ibUMAP

The YAML files install third-party dependencies only.

**`ibumap-cuda`** — build both native extensions (the CPU atomic P2M kernel and
the optional cuML NN-Descent graph extension):

```bash
conda activate ibumap-cuda
IBUMAP_BUILD_CUML_GRAPH_EXT=1 \
python -m pip install --no-build-isolation --no-deps -e .
```

- Do not set `IBUMAP_DISABLE_CPU_ATOMIC_EXT`. Without the atomic extension the
  CPU path silently falls back to a slower P2M implementation.
- `--no-build-isolation` is required: the build uses the pinned NumPy, Cython,
  CCCL, `nvcc` and conda GCC from the active environment.
- `setup.py` selects `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++`
  automatically; do not override it with the system `/usr/bin/g++`.

**`ibumap-metal`**:

```bash
conda activate ibumap-metal
python -m pip install --no-deps -e .
```

**`ibumap-torchdr`**: do not install `ibumap`.

## Smoke tests

```bash
conda activate ibumap-cuda    && python scripts/smoke_test.py cuda
conda activate ibumap-torchdr && python scripts/smoke_test.py torchdr
conda activate ibumap-metal   && python scripts/smoke_test.py metal
```

The `cuda` and `metal` runs include the `cpu` checks. Each check prints
`PASS`/`FAIL`; the exit status is non-zero if any check fails. Differences
from the pinned versions are reported as `WARN` only. Checks include:

- `pyFFTW` distribution metadata;
- the CPU atomic extension is importable (no silent fallback);
- CUDA: RMM/cuML load from the active environment, the cuML graph extension is
  built and loadable, each hand-written CuPy kernel matches a NumPy reference,
  and `fit_transform` runs with `deterministic=False` and `True`;
- `fit_transform` on three separated Gaussian blobs gives finite output with
  leave-one-out 1-NN label accuracy >= 0.95 in the embedding.

Set `IBUMAP_SMOKE_TRACEBACK=1` to print full tracebacks for failed checks.

## Unit tests

```bash
python -m pytest -q -ra
```

Run in `ibumap-cuda` or `ibumap-metal`, not in `ibumap-torchdr`. Some tests
use mocks or platform skips, so a passing suite does not prove that a GPU was
used; run the smoke test as well.

## Refresh the snapshots

Only after `ibumap` was installed as above and the smoke test passed. For
the Linux environments, export from the environment the paper experiments
ran in (see the note under [Overview](#overview)). Run on the target
platform:

```bash
env=cuda-linux-64            # or torchdr-linux-64, metal-osx-arm64
conda list --explicit > environments/$env.conda.lock
python -m pip list --format=freeze --exclude-editable > environments/$env.pip.txt
```

Use `pip list --format=freeze`, not `pip freeze`: the latter records conda
packages as local `file://` build paths and editable installs as VCS URLs,
which are neither portable nor anonymous. Review `git diff` before
committing.

## Troubleshooting

### `pyfftw.__version__` prints `0.0.0`

`pyFFTW` came from conda-forge rather than pip:

```bash
conda list pyfftw
python -c 'from importlib.metadata import version; print(version("pyFFTW"))'
```

Reinstall with `python -m pip install pyfftw==0.15.1`.

### FAISS reports `undefined symbol: __libc_single_threaded`

Do not upgrade FAISS in `ibumap-torchdr`. Newer builds require glibc >= 2.32.
The YAML pins the verified build
`faiss-gpu=1.8.0=py3.11_h4c7d538_0_cuda12.1.1` from the `pytorch` channel.

### TorchDR / Matplotlib reports `CXXABI_1.3.15 not found`

The saved `LD_PRELOAD` was not restored:

```bash
conda env config vars list -n ibumap-torchdr
echo "$LD_PRELOAD"    # expected literal value: $ORIGIN/../lib/libstdc++.so.6
```

If it is missing:

```bash
conda env config vars set 'LD_PRELOAD=$ORIGIN/../lib/libstdc++.so.6' -n ibumap-torchdr
conda deactivate && conda activate ibumap-torchdr
```

`$ORIGIN` is expanded by the dynamic loader and should stay literal in the
shell.

System commands run inside the activated environment (`rm`, `ls`, ...) resolve
`$ORIGIN` to their own directory and print a harmless warning:
`ld.so: object '$ORIGIN/../lib/libstdc++.so.6' from LD_PRELOAD cannot be
preloaded ... ignored`. Only the environment's own executables need the preload.

### CUDA environment loads base Anaconda or cannot find `librmm.so`

```bash
conda activate ibumap-cuda
hash -r
command -v python          # expected: $CONDA_PREFIX/bin/python
echo "$LD_LIBRARY_PATH"    # expected literal: $ORIGIN/../lib:$ORIGIN/../targets/x86_64-linux/lib
python -c "import sys, rmm; print(sys.executable); print(rmm.__file__)"
```

`rmm.__file__` must be below `$CONDA_PREFIX/lib/python3.11/site-packages`. If
Python still resolves to the base installation, reactivate the environment and
check shell aliases and startup files. Do not copy libraries into the base
environment.

### CUDA extension build reports CCCL or C++20 errors

```bash
conda list --show-channel-urls | \
  grep -E '^(cccl|cuda-nvcc|cython|gxx_linux-64|sysroot_linux-64)[[:space:]]'
command -v nvcc && nvcc --version
command -v x86_64-conda-linux-gnu-g++ && x86_64-conda-linux-gnu-g++ --version
```

Expected: CCCL 3.3.4, Cython 3.1.8, CUDA/nvcc 12.6.85, GCC 13.4. A system GCC
(e.g. GCC 9 on Ubuntu 20.04) does not accept the `-std=c++20` flag used here.

### Linker reports `cannot find -lcuml++`

RAPIDS 26.06 ships `libcuml.so`, whose linker name is `cuml`. `setup.py` links
`cuml,rmm` by default; if an override is needed:

```bash
export IBUMAP_CUML_LIBRARIES="cuml,rmm"
```

### `ModuleNotFoundError: No module named 'ibumap'`

Install the package as described in [Install ibUMAP](#install-ibumap).
