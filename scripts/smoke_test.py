#!/usr/bin/env python
"""Smoke test for one ibUMAP execution path (or the TorchDR baseline env).

Usage, from the repository root inside the matching environment::

    python scripts/smoke_test.py cpu       # any env with ibumap installed
    python scripts/smoke_test.py cuda      # ibumap-cuda (runs the cpu checks too)
    python scripts/smoke_test.py metal     # ibumap-metal (runs the cpu checks too)
    python scripts/smoke_test.py torchdr   # ibumap-torchdr (does not import ibumap)

Exit status is 0 only if every check passes. Version pins are reported as
warnings, not failures, so the script also works outside the reference
environments.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np

# Reference-environment pins (environments/*.yml). Mismatches are warnings.
EXPECTED_VERSIONS = {
    "common": {"numpy": "1.26.4", "scipy": "1.16.3", "numba": "0.61.2",
               "umap-learn": "0.5.12", "pynndescent": "0.5.13", "pyFFTW": "0.15.1"},
    "cuda": {"cython": "3.1.8", "rmm": "26.6.0", "cuml": "26.6.0",
             "cuvs": "26.6.0", "cupy": "13.6.0"},
    "metal": {"mlx": "0.32.0"},
    "torchdr": {"torchdr": "0.4", "faiss": "1.8.0"},
}

_results: list[tuple[str, bool, str]] = []
_warnings: list[str] = []


def check(name: str) -> Callable[[Callable[[], str | None]], Callable[[], None]]:
    """Run the decorated function immediately and record PASS/FAIL."""
    def decorator(fn: Callable[[], str | None]) -> Callable[[], None]:
        def run() -> None:
            start = time.perf_counter()
            try:
                detail = fn() or ""
                ok = True
            except Exception as exc:  # noqa: BLE001 - report every failure
                detail = f"{type(exc).__name__}: {exc}"
                if os.environ.get("IBUMAP_SMOKE_TRACEBACK") == "1":
                    detail += "\n" + traceback.format_exc()
                ok = False
            elapsed = time.perf_counter() - start
            _results.append((name, ok, detail))
            status = "PASS" if ok else "FAIL"
            suffix = f" - {detail}" if detail else ""
            print(f"[{status}] {name} ({elapsed:.1f}s){suffix}", flush=True)

        run()  # checks execute where they are defined, in source order
        return run
    return decorator


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def report_versions(groups: list[str]) -> None:
    print(f"python    : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform  : {platform.platform()} / {platform.machine()}")
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        print(f"conda env : {prefix}")
        if not Path(sys.executable).resolve().is_relative_to(Path(prefix).resolve()):
            _warnings.append("sys.executable is not inside $CONDA_PREFIX")
    for group in groups:
        for name, expected in EXPECTED_VERSIONS[group].items():
            got = _dist_version(name)
            print(f"  {name:<12} {got}")
            if got is None:
                _warnings.append(f"{name} is not installed (expected {expected})")
            elif got != expected:
                _warnings.append(f"{name}=={got}, reference env pins {expected}")
    print()


# ---------------------------------------------------------------------------
# Synthetic data: three well-separated Gaussian blobs. The kNN graph is
# disconnected, which also exercises the multi-component spectral layout.
# ---------------------------------------------------------------------------

def make_blobs(n_per_cluster: int = 500, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=20.0, size=(3, dim))
    X = np.concatenate(
        [c + rng.normal(size=(n_per_cluster, dim)) for c in centers]
    ).astype(np.float32)
    y = np.repeat(np.arange(3), n_per_cluster)
    return X, y


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):  # CuPy
        array = array.get()
    elif hasattr(array, "detach"):  # torch
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def assert_good_embedding(embedding, y, min_accuracy: float = 0.95) -> str:
    from sklearn.neighbors import KNeighborsClassifier

    emb = _to_numpy(embedding)
    assert emb.shape == (y.size, 2), f"unexpected shape {emb.shape}"
    assert np.isfinite(emb).all(), "embedding contains non-finite values"
    # Leave-one-out 1-NN label accuracy in the embedding.
    knn = KNeighborsClassifier(n_neighbors=2).fit(emb, y)
    idx = knn.kneighbors(emb, return_distance=False)[:, 1]
    accuracy = float((y[idx] == y).mean())
    assert accuracy >= min_accuracy, f"1-NN accuracy {accuracy:.3f} < {min_accuracy}"
    return f"shape={emb.shape}, 1-NN acc={accuracy:.3f}"


def fit_ibumap(device: str, algorithm: str = "ibumap", **kwargs):
    from ibumap import IBUMAP

    X, y = make_blobs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IBUMAP(algorithm=algorithm, device=device, n_epochs=200,
                       random_state=0, **kwargs)
        embedding = model.fit_transform(X)
    return model, embedding, y


def _extension_path(module: str) -> str:
    path = Path(import_module(module).__file__).resolve()
    assert path.suffix in {".so", ".pyd"}, f"{module} resolved to {path}"
    return str(path)


# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------

def cpu_checks() -> None:
    @check("pyFFTW distribution metadata")
    def _():
        got = _dist_version("pyFFTW")
        assert got not in (None, "0.0.0"), f"pyFFTW metadata reports {got!r}"
        return got

    @check("import ibumap")
    def _():
        import ibumap

        missing = [n for n in getattr(ibumap, "__all__", []) if not hasattr(ibumap, n)]
        assert not missing, f"unresolved __all__ names: {missing}"
        return f"{len(ibumap.__all__)} public names"

    @check("CPU atomic P2M extension")
    def _():
        from ibumap.kernels.cpu import ibfft

        assert ibfft._native_p2m_atomic is not None, (
            "_p2m_atomic not importable; CPU ibFFT would silently fall back "
            "to the non-atomic path"
        )
        return _extension_path("ibumap.kernels.cpu._p2m_atomic")

    @check("fit_transform cpu / ibumap")
    def _():
        _, emb, y = fit_ibumap("cpu")
        return assert_good_embedding(emb, y)

    @check("fit_transform cpu / umap")
    def _():
        _, emb, y = fit_ibumap("cpu", algorithm="umap")
        return assert_good_embedding(emb, y)


# ---------------------------------------------------------------------------
# CUDA path
# ---------------------------------------------------------------------------

def _block_csr_graph(sizes=(40, 30, 50), seed: int = 0):
    """Symmetric positive-weight CSR graph made of disconnected random blocks."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    blocks = []
    for n in sizes:
        w = rng.uniform(0.1, 1.0, size=(n, n)) * (rng.random((n, n)) < 0.3)
        w = np.triu(w, 1)
        w = w + w.T
        w[np.arange(n - 1), np.arange(1, n)] = 0.5  # keep each block connected
        w[np.arange(1, n), np.arange(n - 1)] = 0.5
        blocks.append(sp.csr_matrix(w))
    graph = sp.block_diag(blocks, format="csr").astype(np.float32)
    graph.sort_indices()
    return graph


def cuda_checks() -> None:
    @check("CuPy device and arithmetic")
    def _():
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        assert count > 0, "no CUDA device visible"
        total = float(cp.sum(cp.arange(1024, dtype=cp.float32)).get())
        assert total == 523776.0, total
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        # runtimeGetVersion(): CUDA runtime CuPy was built against;
        # get_local_runtime_version(): libcudart actually loaded from the env.
        built = cp.cuda.runtime.runtimeGetVersion()
        local = getattr(cp.cuda, "get_local_runtime_version", lambda: None)()
        driver = cp.cuda.runtime.driverGetVersion()
        return (f"{count} device(s), {name}, CUDA runtime local={local} "
                f"(CuPy built with {built}), driver API {driver}")

    @check("RMM / cuML resolve inside the active env")
    def _():
        import cuml
        import rmm

        paths = [Path(m.__file__).resolve() for m in (rmm, cuml)]
        prefix = os.environ.get("CONDA_PREFIX")
        if prefix:
            outside = [p for p in paths if not p.is_relative_to(Path(prefix).resolve())]
            assert not outside, f"loaded from outside $CONDA_PREFIX: {outside}"
        return str(paths[0].parent.parent)

    @check("cuML NN-Descent graph extension")
    def _():
        path = _extension_path("ibumap.graph._cuml_graph_ext")
        os.environ["IBUMAP_REQUIRE_CUML_GRAPH_EXT"] = "1"
        try:
            from ibumap.graph.gpu_graph_cuml import _optional_graph_ext

            assert _optional_graph_ext() is not None
        finally:
            os.environ.pop("IBUMAP_REQUIRE_CUML_GRAPH_EXT", None)
        return path

    @check("RawKernel ibumap_csr_row_sum_float32")
    def _():
        import cupyx.scipy.sparse as cps
        from ibumap.kernels.gpu.cupy_sparse import deterministic_csr_row_sum_cupy

        graph = _block_csr_graph()
        got = _to_numpy(deterministic_csr_row_sum_cupy(cps.csr_matrix(graph)))
        np.testing.assert_allclose(got, np.asarray(graph.sum(axis=1)).ravel(), rtol=1e-5)

    @check("RawKernel ibumap_component_min_vertex_int32")
    def _():
        import cupy as cp
        from ibumap.kernels.gpu.cupy_sparse import canonicalize_component_labels_cupy

        # Native ids 2,0,1 in order of first appearance -> canonical 0,1,2.
        labels = cp.asarray([2, 2, 0, 1, 0, 1, 2], dtype=cp.int32)
        got = _to_numpy(canonicalize_component_labels_cupy(labels, 3))
        np.testing.assert_array_equal(got, [0, 0, 1, 2, 1, 2, 0])

    @check("RawKernel ibumap_csr_normalize_float32")
    def _():
        import cupyx.scipy.sparse as cps
        from ibumap.init._spectral_cupy_impl import _deterministic_normalized_laplacian_cupy

        graph = _block_csr_graph()
        lap, _sqrt_deg = _deterministic_normalized_laplacian_cupy(cps.csr_matrix(graph))
        d = 1.0 / np.sqrt(np.asarray(graph.sum(axis=1)).ravel())
        expected = np.eye(graph.shape[0]) - d[:, None] * graph.toarray() * d[None, :]
        np.testing.assert_allclose(_to_numpy(lap.toarray()), expected, atol=1e-5)

    @check("ElementwiseKernel ibumap_eigsh_normalize (eigsh_csr_alg2)")
    def _():
        import cupyx.scipy.sparse as cps
        import scipy.sparse as sp
        from ibumap.init._eigsh_cupy import eigsh_csr_alg2

        graph = _block_csr_graph(sizes=(80,))
        d = 1.0 / np.sqrt(np.asarray(graph.sum(axis=1)).ravel())
        lap = (np.eye(80) - d[:, None] * graph.toarray() * d[None, :]).astype(np.float32)
        vals = eigsh_csr_alg2(cps.csr_matrix(sp.csr_matrix(lap)), k=3, which="LA",
                              return_eigenvectors=False)
        expected = np.linalg.eigvalsh(lap.astype(np.float64))[-3:]
        np.testing.assert_allclose(np.sort(_to_numpy(vals)), expected, atol=1e-3)

    @check("fit_transform cuda / ibumap")
    def _():
        model, emb, y = fit_ibumap("cuda")
        backend = model.runtime.graph_backend
        assert backend == "cuml", f"graph_backend resolved to {backend!r}"
        return assert_good_embedding(emb, y) + f", graph_backend={backend}"

    @check("fit_transform cuda / ibumap / deterministic=True")
    def _():
        _, emb, y = fit_ibumap("cuda", deterministic=True)
        return assert_good_embedding(emb, y)

    @check("fit_transform cuda / umap")
    def _():
        _, emb, y = fit_ibumap("cuda", algorithm="umap")
        return assert_good_embedding(emb, y)


# ---------------------------------------------------------------------------
# Metal path
# ---------------------------------------------------------------------------

def metal_checks() -> None:
    @check("MLX device and arithmetic")
    def _():
        import mlx.core as mx

        total = mx.sum(mx.arange(1024, dtype=mx.float32))
        mx.eval(total)
        assert total.item() == 523776.0, total.item()
        return f"default device {mx.default_device()}"

    @check("fit_transform metal / ibumap")
    def _():
        _, emb, y = fit_ibumap("metal")
        return assert_good_embedding(emb, y)


# ---------------------------------------------------------------------------
# TorchDR baseline environment (ibumap is intentionally not installed here)
# ---------------------------------------------------------------------------

def torchdr_checks() -> None:
    @check("LD_PRELOAD keeps the conda libstdc++")
    def _():
        value = os.environ.get("LD_PRELOAD", "")
        assert "libstdc++" in value, f"LD_PRELOAD={value!r}"
        return value

    @check("FAISS GPU")
    def _():
        import faiss

        assert faiss.__version__.startswith("1.8."), faiss.__version__
        assert faiss.get_num_gpus() > 0, "FAISS sees no GPU"
        return f"faiss {faiss.__version__}, {faiss.get_num_gpus()} GPU(s)"

    @check("PyTorch CUDA")
    def _():
        import torch

        assert torch.cuda.is_available()
        return f"torch {torch.__version__}, CUDA {torch.version.cuda}"

    @check("TorchDR UMAP (faiss backend)")
    def _():
        from torchdr import UMAP

        X, y = make_blobs()
        emb = UMAP(n_neighbors=15, max_iter=200, device="cuda",
                   backend="faiss", random_state=0).fit_transform(X)
        return assert_good_embedding(emb, y)


PATHS = {
    "cpu": (["common"], [cpu_checks]),
    "cuda": (["common", "cuda"], [cpu_checks, cuda_checks]),
    "metal": (["common", "metal"], [cpu_checks, metal_checks]),
    "torchdr": (["torchdr"], [torchdr_checks]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", choices=sorted(PATHS))
    args = parser.parse_args()

    groups, suites = PATHS[args.path]
    report_versions(groups)
    for suite in suites:
        print(f"== {suite.__name__}")
        suite()
    return summarize()


def summarize() -> int:
    print()
    for message in _warnings:
        print(f"[WARN] {message}")
    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
