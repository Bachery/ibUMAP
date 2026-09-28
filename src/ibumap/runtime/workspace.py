from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, Tuple

import numpy as np

from .._fft_kernel_cache import FFTKernelLRUCache


@dataclass
class CPUWorkspace:
    buffers: Dict[str, np.ndarray] = field(default_factory=dict)
    fft_kernel_cache: FFTKernelLRUCache = field(
        default_factory=lambda: FFTKernelLRUCache(scope="workspace")
    )

    def get_or_alloc(self, key: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        arr = self.buffers.get(key)
        if arr is None or arr.shape != shape or arr.dtype != np.dtype(dtype):
            arr = np.zeros(shape, dtype=dtype)
            self.buffers[key] = arr
        else:
            arr.fill(0)
        return arr

    def clear(self) -> None:
        self.buffers.clear()
        self.fft_kernel_cache.clear()


@dataclass
class GPUWorkspace:
    buffers: Dict[str, Any] = field(default_factory=dict)
    fft_kernel_cache: FFTKernelLRUCache = field(
        default_factory=lambda: FFTKernelLRUCache(scope="workspace")
    )
    fft_plans: Dict[str, Tuple[Hashable, Any]] = field(default_factory=dict)
    fft_plan_hits: int = 0
    fft_plan_misses: int = 0

    def get_or_alloc(self, key: str, shape: Tuple[int, ...], dtype: str = "float32"):
        import cupy as cp

        arr = self.buffers.get(key)
        if arr is None or tuple(arr.shape) != tuple(shape) or arr.dtype != cp.dtype(dtype):
            arr = cp.zeros(shape, dtype=dtype)
            self.buffers[key] = arr
        else:
            arr.fill(0)
        return arr

    def get_fft_plan(
        self,
        role: str,
        key: Hashable,
        builder: Callable[[], Any],
    ) -> Tuple[Any, bool]:
        """Return the one retained cuFFT plan for a transform role."""
        cached = self.fft_plans.get(str(role))
        if cached is not None and cached[0] == key:
            self.fft_plan_hits += 1
            return cached[1], True
        plan = builder()
        self.fft_plans[str(role)] = (key, plan)
        self.fft_plan_misses += 1
        return plan, False

    def fft_snapshot(self) -> Dict[str, int]:
        return {
            "plan_entries": len(self.fft_plans),
            "plan_hits": int(self.fft_plan_hits),
            "plan_misses": int(self.fft_plan_misses),
        }

    def clear(self) -> None:
        self.buffers.clear()
        self.fft_kernel_cache.clear()
        self.fft_plans.clear()
        self.fft_plan_hits = 0
        self.fft_plan_misses = 0
        self.memory_pool().free_all_blocks()

    @staticmethod
    def memory_pool() -> Any:
        import cupy as cp

        return cp.get_default_memory_pool()


@dataclass
class MetalWorkspace:
    """Estimator-local MLX kernel and FFT-spectrum cache ownership."""

    kernels: Dict[str, Any] = field(default_factory=dict)
    fft_kernel_cache: OrderedDict[Hashable, Any] = field(default_factory=OrderedDict)
    fft_kernel_cache_max_entries: int = 64
    kernel_hits: int = 0
    kernel_misses: int = 0
    fft_hits: int = 0
    fft_misses: int = 0

    def get_kernel(self, key: str, builder: Callable[[], Any]) -> Any:
        kernel = self.kernels.get(key)
        if kernel is None:
            self.kernel_misses += 1
            kernel = builder()
            self.kernels[key] = kernel
        else:
            self.kernel_hits += 1
        return kernel

    def get_fft_kernel(self, key: Hashable) -> Any:
        value = self.fft_kernel_cache.get(key)
        if value is None:
            self.fft_misses += 1
            return None
        self.fft_hits += 1
        self.fft_kernel_cache.move_to_end(key)
        return value

    def store_fft_kernel(self, key: Hashable, value: Any) -> None:
        self.fft_kernel_cache[key] = value
        self.fft_kernel_cache.move_to_end(key)
        while len(self.fft_kernel_cache) > int(self.fft_kernel_cache_max_entries):
            self.fft_kernel_cache.popitem(last=False)

    def snapshot(self) -> dict[str, int]:
        return {
            "kernel_entries": len(self.kernels),
            "kernel_hits": self.kernel_hits,
            "kernel_misses": self.kernel_misses,
            "fft_kernel_entries": len(self.fft_kernel_cache),
            "fft_kernel_hits": self.fft_hits,
            "fft_kernel_misses": self.fft_misses,
        }

    def clear(self) -> None:
        self.kernels.clear()
        self.fft_kernel_cache.clear()
        self.kernel_hits = 0
        self.kernel_misses = 0
        self.fft_hits = 0
        self.fft_misses = 0
