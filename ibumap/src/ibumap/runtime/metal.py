"""MLX/Metal runtime policy and the first-release capability matrix.

This module must remain safe to import on CPU-only and non-macOS systems.
MLX is imported only by :func:`ensure_metal_runtime` after the public API has
selected ``device="metal"``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
from types import MappingProxyType
import platform
import sys
from typing import Any, Mapping

import numpy as np

from .device import DeviceError


class MetalDeviceError(DeviceError):
    """Raised when the requested MLX/Metal execution path is unavailable."""


class MetalCapabilityError(NotImplementedError):
    """Raised when a configuration is outside the initial Metal contract."""


@dataclass(frozen=True, slots=True)
class MetalRuntimeInfo:
    system: str
    machine: str
    python_version: str
    backend: str
    backend_version: str | None
    metal_available: bool


METAL_CAPABILITIES: Mapping[str, Any] = MappingProxyType(
    {
        "backend": "mlx",
        "algorithm": ("ibumap",),
        "dtype": ("float32",),
        "n_components": (2,),
        "graph_backend": ("auto", "cpu"),
        "graph_device": "cpu",
        "init_device": "cpu",
        "optimizer_device": "metal",
        "n_interpolation_points": (1, 2, 3),
        "p2m_mode": ("auto", "atomic", "segmented"),
        "attraction_mode": ("sampling",),
        "repulsion_mode": ("true_loss",),
        "constraints": ("hard",),
        "noise_mode": ("none", "force", "embedding"),
        "force_clipping": True,
        "deterministic": True,
        "host_fixed_inputs": True,
        "public_entrypoints": (
            "fit",
            "fit_transform",
            "fit_transform_from_knn",
            "optimize_from_graph",
            "optimize_from_prepared_graph",
            "update_embedding",
        ),
        "pipeline_status": "hybrid_cpu_graph_init",
        "optimizer_status": "available",
    }
)


def metal_capability_matrix() -> dict[str, Any]:
    """Return a mutable, JSON-safe copy of the public Metal capabilities."""

    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in METAL_CAPABILITIES.items()
    }


def _mlx_version() -> str | None:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_metal_runtime() -> MetalRuntimeInfo:
    """Require a usable Apple Silicon MLX Metal runtime without fallback."""

    system = platform.system()
    machine = platform.machine().lower()
    python_version = platform.python_version()
    if sys.version_info < (3, 11):
        raise MetalDeviceError(
            "device='metal' requires Python 3.11 or newer"
        )
    if system != "Darwin":
        raise MetalDeviceError(
            "device='metal' requires macOS on Apple Silicon; no CPU fallback was used"
        )
    if machine not in {"arm64", "aarch64"}:
        raise MetalDeviceError(
            "device='metal' requires an Apple Silicon arm64 Python process"
        )
    if importlib.util.find_spec("mlx") is None:
        raise MetalDeviceError(
            "device='metal' requires MLX; install the optional dependency with "
            "pip install 'umap-fft[metal]'"
        )

    try:
        mx = importlib.import_module("mlx.core")
    except Exception as exc:
        raise MetalDeviceError(
            f"device='metal' could not import mlx.core: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        available = bool(mx.metal.is_available())
    except Exception as exc:
        raise MetalDeviceError(
            "device='metal' could not query the MLX Metal device: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not available:
        raise MetalDeviceError(
            "device='metal' requested MLX, but no Metal device is available; "
            "no CPU fallback was used"
        )

    return MetalRuntimeInfo(
        system=system,
        machine=machine,
        python_version=python_version,
        backend="mlx",
        backend_version=_mlx_version(),
        metal_available=True,
    )


def _is_default_noise(noise: Any) -> bool:
    return (
        noise.mode == "none"
        and noise.scale == 0.0
        and noise.decay == "linear"
        and noise.until_epoch is None
        and noise.seed is None
        and noise.hybrid_mode == "none"
        and noise.hybrid_switch_epoch is None
    )


def validate_metal_config(config: Any) -> None:
    """Reject features outside the frozen first Metal optimizer contract."""

    runtime = config.runtime
    fft = config.fft
    ibumap = config.ibumap
    experimental = ibumap.experimental

    if runtime.algorithm != "ibumap":
        raise MetalCapabilityError(
            "device='metal' initially supports algorithm='ibumap' only"
        )
    if runtime.graph_backend not in ("auto", "cpu"):
        raise MetalCapabilityError(
            "device='metal' uses CPU graph construction; graph_backend must be "
            "'auto' or 'cpu'"
        )
    if np.dtype(config.legacy_numerics.dtype) != np.dtype(np.float32):
        raise MetalCapabilityError(
            "device='metal' initially supports dtype='float32' only"
        )
    if config.umap.n_components != 2:
        raise MetalCapabilityError(
            "device='metal' initially supports n_components=2 only"
        )
    if ibumap.attraction_mode != "sampling":
        raise MetalCapabilityError(
            "device='metal' initially supports attraction_mode='sampling' only"
        )
    if ibumap.repulsion_mode != "true_loss":
        raise MetalCapabilityError(
            "device='metal' initially supports repulsion_mode='true_loss' only"
        )
    if fft.n_interpolation_points not in (1, 2, 3):
        raise MetalCapabilityError(
            "device='metal' supports n_interpolation_points 1, 2, or 3"
        )
    if fft.interpolation_schedule is not None and any(
        stage.n_interpolation_points not in (1, 2, 3)
        for stage in fft.interpolation_schedule
    ):
        raise MetalCapabilityError(
            "device='metal' interpolation_schedule supports p values 1, 2, or 3"
        )
    has_higher_order = (
        fft.n_interpolation_points > 1
        or fft.combine_stages
        or (
            fft.interpolation_schedule is not None
            and any(
                stage.n_interpolation_points > 1
                for stage in fft.interpolation_schedule
            )
        )
    )
    if fft.p2m_mode not in ("auto", "atomic", "segmented"):
        raise MetalCapabilityError(
            "device='metal' supports p2m_mode='auto', 'atomic', or 'segmented'"
        )
    if config.cpu_umap.deterministic and fft.p2m_mode == "atomic":
        raise MetalCapabilityError(
            "deterministic=True requires p2m_mode='auto' or 'segmented'"
        )
    if has_higher_order and config.cpu_umap.deterministic:
        raise MetalCapabilityError(
            "deterministic Metal P2M is currently supported for p=1 only"
        )
    if has_higher_order and fft.p2m_mode == "segmented":
        raise MetalCapabilityError(
            "p2m_mode='segmented' is currently supported for p=1 only"
        )
    if fft.kernel_cache_policy not in ("auto", "disabled"):
        raise MetalCapabilityError(
            "Metal FFT kernel caching is not implemented; use "
            "kernel_cache_policy='auto' or 'disabled'"
        )

    attraction_schedule = experimental.attraction_schedule
    if attraction_schedule.kernel_mode not in ("auto", "thread_per_row"):
        raise MetalCapabilityError(
            "device='metal' initially supports thread-per-row attraction only"
        )
    if attraction_schedule.schedule_mode != "row_scan":
        raise MetalCapabilityError(
            "device='metal' initially supports attraction_schedule_mode='row_scan' only"
        )
    if experimental.kernel_subsampling is not None:
        raise MetalCapabilityError(
            "ibFFT kernel sub-sampling is not implemented for device='metal'"
        )
    if experimental.local_exact_repulsion.enabled:
        raise MetalCapabilityError(
            "local_exact_repulsion is not implemented for device='metal'"
        )
    if experimental.local_density_pressure.enabled:
        raise MetalCapabilityError(
            "local_density_pressure is not implemented for device='metal'"
        )
    constraint = config.constraints
    if constraint.mode != "hard":
        raise MetalCapabilityError(
            "soft constraints are not implemented for device='metal'"
        )

    diagnostics = config.diagnostics
    if diagnostics.summary_path or diagnostics.timing_path or diagnostics.topk_path:
        raise MetalCapabilityError(
            "force/timing/top-k diagnostics are not implemented for device='metal'; "
            "process memory diagnostics remain available"
        )
