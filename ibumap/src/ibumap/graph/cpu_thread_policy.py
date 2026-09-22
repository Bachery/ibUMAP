from __future__ import annotations

import os
import platform
import warnings
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Optional


_VALIDATED_JOBS = (8, 4, 2)
_VALIDATED_PLATFORMS = {
    ("Darwin", "arm64"),
    ("Linux", "x86_64"),
}
_VALIDATED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "scikit-learn": "1.8.0",
    "numba": "0.61.2",
    "llvmlite": "0.44.0",
    "pynndescent": "0.5.13",
    "umap-learn": "0.5.12",
}


@dataclass(frozen=True)
class CPUGraphRuntime:
    system: str
    machine: str
    threading_layer: Optional[str]
    thread_capacity: int
    versions: Mapping[str, Optional[str]]
    probe_error: Optional[str] = None

    @property
    def verified(self) -> bool:
        return bool(
            self.probe_error is None
            and (self.system, self.machine) in _VALIDATED_PLATFORMS
            and self.threading_layer == "tbb"
            and all(
                self.versions.get(name) == expected
                for name, expected in _VALIDATED_VERSIONS.items()
            )
        )


@dataclass(frozen=True)
class CPUGraphThreadPolicy:
    requested_n_jobs: int
    effective_n_jobs: int
    deterministic: bool
    policy: str
    runtime_verified: bool
    thread_capacity: Optional[int] = None
    threading_layer: Optional[str] = None
    fallback_reason: Optional[str] = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "graph_n_jobs_requested": int(self.requested_n_jobs),
            "graph_n_jobs_effective": int(self.effective_n_jobs),
            "graph_thread_policy": self.policy,
            "graph_parallel_runtime_verified": bool(self.runtime_verified),
            "graph_thread_capacity": self.thread_capacity,
            "graph_threading_layer": self.threading_layer,
            "graph_thread_fallback_reason": self.fallback_reason,
        }


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cpu_capacity() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 1))


def detect_cpu_graph_runtime() -> CPUGraphRuntime:
    versions = {name: _package_version(name) for name in _VALIDATED_VERSIONS}
    threading_layer: Optional[str] = None
    probe_error: Optional[str] = None
    numba_capacity = 1
    try:
        import numba

        numba_capacity = max(1, int(numba.config.NUMBA_NUM_THREADS))
        configured_layer = str(numba.config.THREADING_LAYER)
        priority = tuple(str(value) for value in numba.config.THREADING_LAYER_PRIORITY)
        tbb_is_selected = configured_layer == "tbb" or (
            configured_layer == "default" and priority and priority[0] == "tbb"
        )
        if tbb_is_selected:
            # Importing the backend extension verifies availability without
            # launching the TBB worker pool before PyNNDescent starts.
            import_module("numba.np.ufunc.tbbpool")
            threading_layer = "tbb"
        else:
            threading_layer = configured_layer
    except Exception as exc:  # pragma: no cover - environment dependent
        probe_error = f"{type(exc).__name__}: {exc}"

    return CPUGraphRuntime(
        system=platform.system(),
        machine=platform.machine(),
        threading_layer=threading_layer,
        thread_capacity=min(_cpu_capacity(), numba_capacity),
        versions=versions,
        probe_error=probe_error,
    )


def resolve_cpu_graph_thread_policy(
    requested_n_jobs: int,
    *,
    deterministic: bool,
    algorithm: str,
    device: str,
    runtime: Optional[CPUGraphRuntime] = None,
) -> CPUGraphThreadPolicy:
    requested = int(requested_n_jobs)
    if requested == 0 or requested < -1:
        raise ValueError("n_jobs must be a positive integer or -1")

    if not deterministic:
        return CPUGraphThreadPolicy(
            requested_n_jobs=requested,
            effective_n_jobs=requested,
            deterministic=False,
            policy="fast_passthrough",
            runtime_verified=False,
        )

    if str(device) != "cpu" or str(algorithm) != "ibumap":
        return CPUGraphThreadPolicy(
            requested_n_jobs=requested,
            effective_n_jobs=1,
            deterministic=True,
            policy="serial_unvalidated_route",
            runtime_verified=False,
            fallback_reason=(
                "parallel deterministic graph is validated for CPU ibumap only"
            ),
        )

    if requested == 1:
        return CPUGraphThreadPolicy(
            requested_n_jobs=1,
            effective_n_jobs=1,
            deterministic=True,
            policy="explicit_serial",
            runtime_verified=False,
        )

    runtime = detect_cpu_graph_runtime() if runtime is None else runtime
    if not runtime.verified:
        warnings.warn(
            "CPU deterministic parallel graph runtime is outside the validated "
            "platform/package/TBB allowlist; falling back to n_jobs=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return CPUGraphThreadPolicy(
            requested_n_jobs=requested,
            effective_n_jobs=1,
            deterministic=True,
            policy="serial_unverified_runtime",
            runtime_verified=False,
            thread_capacity=runtime.thread_capacity,
            threading_layer=runtime.threading_layer,
            fallback_reason=runtime.probe_error
            or "runtime fingerprint is not allowlisted",
        )

    if requested == -1:
        effective = next(
            (value for value in _VALIDATED_JOBS if value <= runtime.thread_capacity),
            1,
        )
        return CPUGraphThreadPolicy(
            requested_n_jobs=-1,
            effective_n_jobs=effective,
            deterministic=True,
            policy=("auto_fixed_parallel" if effective > 1 else "auto_serial_capacity"),
            runtime_verified=True,
            thread_capacity=runtime.thread_capacity,
            threading_layer=runtime.threading_layer,
            fallback_reason=(
                None if effective > 1 else "fewer than two threads are available"
            ),
        )

    if requested not in _VALIDATED_JOBS:
        warnings.warn(
            "CPU deterministic parallel graph only validates explicit n_jobs "
            "values 2, 4, and 8; falling back to n_jobs=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return CPUGraphThreadPolicy(
            requested_n_jobs=requested,
            effective_n_jobs=1,
            deterministic=True,
            policy="serial_unvalidated_thread_count",
            runtime_verified=True,
            thread_capacity=runtime.thread_capacity,
            threading_layer=runtime.threading_layer,
            fallback_reason="explicit thread count was not validated",
        )

    if requested > runtime.thread_capacity:
        raise ValueError(
            "explicit deterministic CPU graph n_jobs="
            f"{requested} exceeds the runtime thread capacity "
            f"{runtime.thread_capacity}; choose a supported value that fits or use -1"
        )

    return CPUGraphThreadPolicy(
        requested_n_jobs=requested,
        effective_n_jobs=requested,
        deterministic=True,
        policy="explicit_fixed_parallel",
        runtime_verified=True,
        thread_capacity=runtime.thread_capacity,
        threading_layer=runtime.threading_layer,
    )
