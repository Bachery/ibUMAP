"""Deterministic CuPy sparse primitives used by the CUDA pipeline."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _csr_row_sum_float32_kernel():
    import cupy as cp

    return cp.RawKernel(
        r"""
        extern "C" __global__
        void ibumap_csr_row_sum_float32(
            const int n_rows,
            const int* indptr,
            const float* data,
            float* degree
        ) {
            const int row = blockDim.x * blockIdx.x + threadIdx.x;
            if (row >= n_rows) {
                return;
            }
            float total = 0.0f;
            for (
                int position = indptr[row];
                position < indptr[row + 1];
                ++position
            ) {
                total += data[position];
            }
            degree[row] = total;
        }
        """,
        "ibumap_csr_row_sum_float32",
    )


@lru_cache(maxsize=None)
def _component_min_vertex_int32_kernel():
    import cupy as cp

    return cp.RawKernel(
        r"""
        extern "C" __global__
        void ibumap_component_min_vertex_int32(
            const int n_vertices,
            const int* labels,
            int* component_min_vertex
        ) {
            const int vertex = blockDim.x * blockIdx.x + threadIdx.x;
            if (vertex >= n_vertices) {
                return;
            }
            atomicMin(&component_min_vertex[labels[vertex]], vertex);
        }
        """,
        "ibumap_component_min_vertex_int32",
    )


def deterministic_csr_row_sum_cupy(graph):
    """Sum each float32 CSR row in its stored left-to-right order.

    One CUDA thread owns one row, so floating-point additions for a row have a
    fixed order and do not depend on atomic scheduling.
    """

    import cupy as cp
    from cupyx.scipy import sparse as cupyx_sparse

    if not cupyx_sparse.isspmatrix_csr(graph):
        raise TypeError("deterministic CUDA row sum requires a CSR matrix")
    if graph.data.dtype != cp.float32:
        raise TypeError("deterministic CUDA row sum requires float32 data")
    if graph.indptr.dtype != cp.int32 or graph.indices.dtype != cp.int32:
        raise TypeError("deterministic CUDA row sum requires int32 CSR indices")

    n_rows = int(graph.shape[0])
    degree = cp.empty(n_rows, dtype=cp.float32)
    if n_rows == 0:
        return degree

    threads = 256
    blocks = (n_rows + threads - 1) // threads
    _csr_row_sum_float32_kernel()(
        (blocks,),
        (threads,),
        (
            np.int32(n_rows),
            graph.indptr,
            graph.data,
            degree,
        ),
    )
    return degree


def canonicalize_component_labels_cupy(labels, n_components: int):
    """Renumber int32 component labels by ascending minimum vertex index.

    ``atomicMin`` is order independent for integers, and different components
    have distinct minimum vertices. The resulting ordering is therefore
    invariant to the native label ids returned by ``connected_components``.
    """

    import cupy as cp

    labels = cp.asarray(labels)
    n_components = int(n_components)
    if labels.ndim != 1:
        raise ValueError("component labels must be one-dimensional")
    if labels.dtype != cp.int32:
        raise TypeError("component labels must have dtype int32")
    if not labels.flags.c_contiguous:
        labels = cp.ascontiguousarray(labels)
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components == 1:
        return labels

    n_vertices = int(labels.size)
    if n_vertices > np.iinfo(np.int32).max:
        raise ValueError("component label canonicalization exceeds int32 range")

    component_min_vertex = cp.full(
        n_components,
        np.iinfo(np.int32).max,
        dtype=cp.int32,
    )
    if n_vertices:
        threads = 256
        blocks = (n_vertices + threads - 1) // threads
        _component_min_vertex_int32_kernel()(
            (blocks,),
            (threads,),
            (
                np.int32(n_vertices),
                labels,
                component_min_vertex,
            ),
        )

    component_order = cp.argsort(component_min_vertex)
    old_to_new = cp.empty(n_components, dtype=cp.int32)
    old_to_new[component_order] = cp.arange(n_components, dtype=cp.int32)
    return old_to_new[labels]


__all__ = [
    "canonicalize_component_labels_cupy",
    "deterministic_csr_row_sum_cupy",
]
