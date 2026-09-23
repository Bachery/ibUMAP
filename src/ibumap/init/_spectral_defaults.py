from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ResolvedSpectralDefaults:
    method: str
    tol: float
    maxiter: int
    ncv: Optional[int]
    policy: str


def has_explicit_spectral_parameter(
    *,
    method=None,
    tol=None,
    maxiter=None,
    ncv=None,
) -> bool:
    return any(value is not None for value in (method, tol, maxiter, ncv))


def spectral_auto_defaults_enabled(
    *,
    init,
    spectral_auto_defaults: Optional[bool],
    method=None,
    tol=None,
    maxiter=None,
    ncv=None,
) -> bool:
    if not (isinstance(init, str) and init == "spectral"):
        return False
    if has_explicit_spectral_parameter(
        method=method,
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
    ):
        return False
    return True if spectral_auto_defaults is None else bool(spectral_auto_defaults)


def resolve_spectral_defaults(
    *,
    n_samples: int,
    n_components: int,
    backend: str = "cpu",
) -> ResolvedSpectralDefaults:
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    backend = str(backend).lower()
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be either 'cpu' or 'cuda'")
    k = int(n_components) + 1
    if n_samples >= 2_000_000:
        if backend == "cuda":
            return ResolvedSpectralDefaults(
                method="eigsh",
                tol=1e-4,
                maxiter=5 * n_samples,
                ncv=64,
                policy="ibumap_cuda_large_eigsh",
            )
        return ResolvedSpectralDefaults(
            method="lobpcg",
            tol=1e-4,
            maxiter=5 * n_samples,
            ncv=None,
            policy="umap_learn_large_lobpcg",
        )
    if n_samples >= 50_000:
        return ResolvedSpectralDefaults(
            method="eigsh",
            tol=1e-3,
            maxiter=5000,
            ncv=64,
            policy="ibumap_large_eigsh_fast",
        )
    return ResolvedSpectralDefaults(
        method="eigsh",
        tol=1e-4,
        maxiter=5 * n_samples,
        ncv=auto_ncv(n_samples, k),
        policy="umap_learn_small_eigsh",
    )


def auto_ncv(component_n: int, k: int) -> int:
    component_n = int(component_n)
    k = int(k)
    if component_n < 1:
        raise ValueError("component_n must be positive")
    if k < 1:
        raise ValueError("k must be positive")
    if component_n < 50_000:
        return max(2 * k + 1, int(np.sqrt(component_n)))
    return 64


def resolve_component_ncv(
    *,
    component_n: int,
    k: int,
    user_ncv: Optional[int],
) -> int:
    component_n = int(component_n)
    k = int(k)
    if component_n <= k + 1:
        raise ValueError("component_n must be greater than k + 1 for eigsh")
    if user_ncv is None:
        ncv = auto_ncv(component_n, k)
    else:
        if isinstance(user_ncv, (bool, np.bool_)) or int(user_ncv) != user_ncv:
            raise ValueError("spectral_ncv must be a positive integer")
        ncv = int(user_ncv)
        if ncv <= 0:
            raise ValueError("spectral_ncv must be a positive integer")
    ncv = min(ncv, component_n - 1)
    ncv = max(ncv, k + 1)
    return ncv
