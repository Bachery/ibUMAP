from __future__ import annotations

from typing import Any, Literal


EvaluationDevice = Literal["cpu", "gpu", "auto"]


def _normalize_device(device: Any) -> EvaluationDevice:
    normalized = str(device).lower()
    if normalized not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be one of 'cpu', 'gpu', or 'auto'")
    return normalized  # type: ignore[return-value]


def gpu_available() -> tuple[bool, str]:
    try:
        import cupy as cp
    except Exception as exc:
        return False, f"CuPy import failed: {exc}"

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, f"CUDA device check failed: {exc}"
    if device_count < 1:
        return False, "no CUDA devices are visible"
    return True, ""


def resolve_device(device: Any) -> EvaluationDevice:
    normalized = _normalize_device(device)
    if normalized == "cpu":
        return "cpu"

    available, reason = gpu_available()
    if normalized == "gpu":
        if not available:
            detail = f" ({reason})" if reason else ""
            raise ImportError(
                "device='gpu' requires CuPy and a visible CUDA GPU for evaluation GPU mode"
                f"{detail}"
            )
        return "gpu"

    return "gpu" if available else "cpu"


def synchronize_gpu_if_loaded() -> None:
    try:
        import sys

        cp = sys.modules.get("cupy")
        if cp is None:
            return
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def free_gpu_memory_if_loaded() -> None:
    try:
        import sys

        cp = sys.modules.get("cupy")
        if cp is None:
            return
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return
