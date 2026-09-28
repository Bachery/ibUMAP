from __future__ import annotations

import importlib.util
import platform


class DeviceError(RuntimeError):
    pass


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def ensure_gpu_runtime(graph_backend: str) -> None:
    ensure_cupy_runtime()

    if graph_backend in ("auto", "cuml") and not _has_module("cuml"):
        raise DeviceError(
            "GPU mode requires cuML for end-to-end GPU pipeline; install cuml or use device='cpu'"
        )


def ensure_cupy_runtime() -> None:
    if "Linux" not in platform.platform():
        raise DeviceError("GPU mode requires Linux runtime")
    if not _has_module("cupy"):
        raise DeviceError("GPU mode requires cupy")


def resolve_graph_backend(device: str, graph_backend: str) -> str:
    if device == "cpu":
        if graph_backend == "cuml":
            raise DeviceError("graph_backend='cuml' is invalid for device='cpu'")
        return "cpu"

    if device == "metal":
        if graph_backend == "cuml":
            raise DeviceError(
                "graph_backend='cuml' is invalid for device='metal'; the Metal "
                "hybrid pipeline builds its graph on CPU"
            )
        return "cpu"

    if graph_backend == "auto":
        return "cuml"
    if graph_backend == "cpu":
        raise DeviceError(
            "graph_backend='cpu' is not allowed for device='cuda' in end-to-end GPU mode"
        )
    return graph_backend
