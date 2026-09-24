from __future__ import annotations

import gc
import os
import platform
from pathlib import Path
from typing import Any


def synchronize_gpu() -> None:
    """Synchronize the default CuPy CUDA stream when CuPy is available."""
    try:
        import cupy as cp  # type: ignore

        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def synchronize_torch_gpu(device: Any = None) -> None:
    """Synchronize a PyTorch CUDA device when PyTorch and CUDA are available."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
    except Exception:
        return


def clear_cupy_memory_pool() -> None:
    """Release CuPy default memory pools when CuPy is available."""
    try:
        import cupy as cp  # type: ignore

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def clear_torch_memory_cache() -> None:
    """Release unused blocks held by PyTorch's CUDA caching allocator."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        return


def force_cleanup(*, include_gpu: bool = False) -> None:
    """Run Python GC and optional GPU synchronization/memory-pool cleanup."""
    gc.collect()
    if include_gpu:
        synchronize_gpu()
        synchronize_torch_gpu()
        clear_cupy_memory_pool()
        clear_torch_memory_cache()
    gc.collect()


def to_numpy_array(value: Any) -> Any:
    """Convert common CPU/GPU array-like objects to a NumPy array."""
    import numpy as np

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return np.asarray(value.detach().cpu().numpy())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    if hasattr(value, "get"):
        try:
            return np.asarray(value.get())
        except TypeError:
            pass

    try:
        import cupy as cp  # type: ignore

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        if hasattr(value, "__cuda_array_interface__"):
            return cp.asnumpy(cp.asarray(value))
    except Exception:
        pass

    if hasattr(value, "to_output"):
        try:
            return np.asarray(value.to_output("numpy"))
        except TypeError:
            pass

    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())
    if hasattr(value, "values") and hasattr(value.values, "get"):
        return np.asarray(value.values.get())
    return np.asarray(value)


def device_for_algorithm(algorithm: str, *, explicit: str | None = None) -> str:
    """Infer the canonical CPU/CUDA device from an experiment algorithm name."""
    if explicit:
        return "cuda" if explicit == "gpu" else str(explicit)
    name = str(algorithm).lower()
    if "gpu" in name or name.startswith(("cuml", "torchdr")):
        return "cuda"
    return "cpu"


def _torch_gpu_info(*, fallback_reason: str | None = None) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "cupy_fallback_reason": fallback_reason,
        }

    try:
        device_count = int(torch.cuda.device_count())
        devices = []
        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)
            device: dict[str, Any] = {
                "id": device_id,
                "name": str(props.name),
                "total_memory_bytes": int(props.total_memory),
            }
            try:
                free_memory, total_runtime_memory = torch.cuda.mem_get_info(
                    device_id
                )
            except Exception:
                pass
            else:
                device["free_memory_bytes"] = int(free_memory)
                device["runtime_total_memory_bytes"] = int(total_runtime_memory)
            devices.append(device)
        return {
            "available": bool(torch.cuda.is_available() and device_count > 0),
            "backend": "torch",
            "device_count": device_count,
            "devices": devices,
            "cupy_fallback_reason": fallback_reason,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "cupy_fallback_reason": fallback_reason,
        }


def gpu_info() -> dict[str, Any]:
    """Return best-effort CUDA device information through CuPy or PyTorch."""
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        return _torch_gpu_info(
            fallback_reason=f"{type(exc).__name__}: {exc}"
        )

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        devices = []
        for device_id in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            total_memory = props.get("totalGlobalMem")
            device: dict[str, Any] = {
                "id": device_id,
                "name": name,
                "total_memory_bytes": int(total_memory) if total_memory is not None else None,
            }
            try:
                free_memory, total_runtime_memory = cp.cuda.runtime.memGetInfo()
            except Exception:
                pass
            else:
                device["free_memory_bytes"] = int(free_memory)
                device["runtime_total_memory_bytes"] = int(total_runtime_memory)
            devices.append(device)
        return {
            "available": device_count > 0,
            "backend": "cupy",
            "device_count": device_count,
            "devices": devices,
        }
    except Exception as exc:
        return _torch_gpu_info(
            fallback_reason=f"{type(exc).__name__}: {exc}"
        )


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def machine_info() -> dict[str, Any]:
    """Hardware description for run metadata (no host or user names)."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "memory_total_bytes": _memory_total_bytes(),
        "gpu": gpu_info(),
    }


def remove_if_empty(path: str | Path) -> bool:
    """Remove an empty directory. Useful for optional cache/log cleanup."""
    directory = Path(path)
    try:
        directory.rmdir()
    except OSError:
        return False
    return True
