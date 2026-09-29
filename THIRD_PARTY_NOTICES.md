# Third-party notices

ibUMAP is released under the BSD 3-Clause License (`LICENSE`). The files below
contain code adapted from other projects and modified by the ibUMAP authors.
Paths are relative to `src/ibumap/`; the upstream license texts are in
`LICENSES/`.

| Upstream | License | Files |
|---|---|---|
| [t-FDP](https://github.com/Ideas-Laboratory/t-fdp) (Zhong et al., IEEE TVCG 2024) | LGPL-2.1; the adapted code is distributed under BSD-3-Clause with the permission of its author | `kernels/cpu/ibfft.py`, `kernels/gpu/RawCudafloat.cu`, `kernels/gpu/cupy_ibfft.py`, `kernels/gpu/cupy_ibfft_fused.py`, `optimizers/ibumap_optimizer.py` |
| [FIt-SNE](https://github.com/KlugerLab/FIt-SNE) `nbodyfft` (Linderman et al., Nature Methods 2019), on which t-FDP's FFT kernels are based | MIT (`LICENSES/FIt-SNE.txt`) | the t-FDP files above |
| [umap-learn](https://github.com/lmcinnes/umap) 0.5.12 | BSD-3-Clause (`LICENSES/umap-learn.txt`) | `init/_spectral_cpu_impl.py`, `init/_spectral_cupy_impl.py`, `optimizers/umap_optimizer.py`, `optimizers/ibumap_optimizer.py`, `api.py`, `utils.py` |
| [CuPy](https://github.com/cupy/cupy) 13.3.0 | MIT (`LICENSES/CuPy.txt`) | `init/_eigsh_cupy.py` |
| [cuML](https://github.com/rapidsai/cuml) 26.06.00 | Apache-2.0 (`LICENSES/cuML.txt`); this file remains under Apache-2.0 | `graph/_cuml_graph_ext.pyx` |

Not included:

- BRAQUE (GPL-3.0): `experiments/braque` reimplements its Lognormal Shrinkage
  preprocessing; no BRAQUE code is included.
- scDEED (MIT): `evaluation/scdeed.py` is an independent implementation.
- FFTW (GPL-2.0-or-later) is used through pyFFTW and is not distributed with
  ibUMAP.
- Datasets are not redistributed; `paper/data` holds derived results only.
