"""Explicit, process-local JIT warmup without concurrent module imports."""

from __future__ import annotations

import threading
import warnings
from typing import Any


_warmup_lock = threading.Lock()
_warmup_state = "not_started"
_warmup_errors: tuple[str, ...] = ()


def _status() -> dict[str, Any]:
    return {
        "state": _warmup_state,
        "completed": _warmup_state in {"complete", "failed"},
        "successful": _warmup_state == "complete",
        "errors": list(_warmup_errors),
    }


def get_jit_warmup_status() -> dict[str, Any]:
    """Return a snapshot of this process's explicit warmup state."""
    with _warmup_lock:
        return _status()


def warmup_jit(*, raise_errors: bool = False) -> dict[str, Any]:
    """Synchronously warm CPU/optimizer and umap-learn JIT kernels once.

    Warmup is deliberately explicit and synchronous.  Starting a thread from
    ``umap_fft.__init__`` and importing package descendants in that thread can
    form a cycle between Python's parent-package and child-module locks when a
    caller immediately imports another submodule.  A process-wide lock makes
    concurrent explicit callers serialize safely, while the completed state
    keeps subsequent calls inexpensive.
    """

    global _warmup_errors, _warmup_state

    with _warmup_lock:
        if _warmup_state in {"complete", "failed"}:
            status = _status()
            if raise_errors and _warmup_errors:
                raise RuntimeError("; ".join(_warmup_errors))
            return status

        _warmup_state = "running"
        errors: list[str] = []
        try:
            from .kernels.cpu import ibfft as _ibfft  # noqa: F401
            from .optimizers import ibumap_optimizer as _optimizer  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency-specific
            errors.append(f"project kernels: {type(exc).__name__}: {exc}")

        try:
            import numpy as np
            from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors

            values = np.random.default_rng(0).random((20, 4), dtype=np.float32)
            knn_indices, knn_distances, _ = nearest_neighbors(
                values,
                5,
                "euclidean",
                {},
                False,
                42,
                True,
                use_pynndescent=True,
                n_jobs=1,
                verbose=False,
            )
            fuzzy_simplicial_set(
                values,
                5,
                42,
                "euclidean",
                {},
                knn_indices,
                knn_distances,
                False,
                1.0,
                1.0,
                True,
                False,
                False,
            )
        except Exception as exc:  # pragma: no cover - dependency-specific
            errors.append(f"umap-learn kernels: {type(exc).__name__}: {exc}")

        _warmup_errors = tuple(errors)
        _warmup_state = "failed" if errors else "complete"
        status = _status()

    if errors:
        message = "JIT warmup did not complete: " + "; ".join(errors)
        if raise_errors:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return status


__all__ = ["get_jit_warmup_status", "warmup_jit"]
