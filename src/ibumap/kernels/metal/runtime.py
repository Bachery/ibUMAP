"""Lazy MLX runtime helpers shared by ibUMAP Metal kernels."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

import numpy as np

from ...runtime.metal import ensure_metal_runtime


@dataclass(frozen=True, slots=True)
class MetalTiming:
    wall_time_s: float


def mlx_core(*, require_device: bool = True) -> Any:
    """Import MLX lazily and select its GPU device."""

    if require_device:
        ensure_metal_runtime()
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - policy normally fails first
        raise RuntimeError(
            f"Could not import mlx.core: {type(exc).__name__}: {exc}"
        ) from exc
    if require_device:
        mx.set_default_device(mx.gpu)
    return mx


def as_metal_array(
    value: Any,
    *,
    dtype: Any = None,
    mx: Any = None,
) -> Any:
    """Create an MLX array without exposing MLX at package import time."""

    mx = mlx_core() if mx is None else mx
    return mx.array(value, dtype=dtype)


def evaluate(*values: Any, mx: Any = None) -> None:
    mx = mlx_core() if mx is None else mx
    if values:
        mx.eval(*values)
    mx.synchronize()


def to_numpy(value: Any, *, mx: Any = None, copy: bool = True) -> np.ndarray:
    """Materialize one explicit device-to-host boundary."""

    mx = mlx_core() if mx is None else mx
    mx.eval(value)
    mx.synchronize()
    result = np.asarray(value)
    return np.array(result, copy=True) if copy else result


def timed_call(
    function: Callable[[], Any],
    *,
    mx: Any = None,
    evaluate_result: bool = True,
) -> tuple[Any, MetalTiming]:
    """Measure a synchronized MLX call using explicit device boundaries."""

    mx = mlx_core() if mx is None else mx
    mx.synchronize()
    started = perf_counter()
    result = function()
    if evaluate_result:
        values: Iterable[Any]
        if isinstance(result, (tuple, list)):
            values = result
        else:
            values = (result,)
        values = tuple(value for value in values if value is not None)
        if values:
            mx.eval(*values)
    mx.synchronize()
    return result, MetalTiming(wall_time_s=perf_counter() - started)


def memory_snapshot(*, mx: Any = None) -> dict[str, int]:
    mx = mlx_core() if mx is None else mx
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def finite_count(value: Any, *, mx: Any = None) -> int:
    """Return the non-finite count at an intentional validation boundary."""

    mx = mlx_core() if mx is None else mx
    count = mx.reshape(mx.sum(~mx.isfinite(value)).astype(mx.int32), (1,))
    return int(to_numpy(count, mx=mx, copy=False)[0])
