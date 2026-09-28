from __future__ import annotations

from typing import Optional

import numba
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse, isspmatrix_csr
from sklearn.utils import check_random_state

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
CPU_SAMPLING_DTYPE = np.dtype(np.float64)
DEVICE_SAMPLING_DTYPE = np.dtype(np.float32)

_RANDOM_STREAMS = {
    "graph": 0,
    "init": 1,
    "sampling": 2,
    "noise": 3,
    "refinement": 4,
}


def derive_random_seed(random_state, stream: str):
    """Derive an independent stable seed without mutating ``random_state``.

    ``None`` keeps the unseeded fast path.
    """
    if stream not in _RANDOM_STREAMS:
        raise ValueError(f"unknown random stream: {stream}")
    if random_state is None:
        return None

    source = check_random_state(random_state)
    copied = np.random.RandomState()
    copied.set_state(source.get_state())
    seeds = copied.randint(1, 2**31 - 1, size=len(_RANDOM_STREAMS))
    return int(seeds[_RANDOM_STREAMS[stream]])


def preprocess_graph(
    graph,
    n_epochs: Optional[int],
    *,
    return_stats: bool = False,
):
    """Apply umap-learn's ordered COO graph preprocessing protocol."""
    if not issparse(graph):
        raise TypeError("fuzzy_graph must be a scipy sparse matrix")

    # umap-learn operates on COO row/col/data arrays after sum_duplicates().
    # Copy first because optimization-only inputs must not be mutated in place.
    graph = graph.tocoo(copy=True)
    graph.sum_duplicates()

    if graph.shape[0] != graph.shape[1]:
        raise ValueError("fuzzy_graph must be square")

    default_epochs = 500 if graph.shape[0] <= 10000 else 200
    n_epochs = default_epochs if n_epochs is None else int(n_epochs)
    divisor = default_epochs if n_epochs <= 10 else n_epochs
    input_nnz = int(graph.nnz)

    if graph.nnz > 0:
        graph.data[graph.data < (graph.data.max() / float(divisor))] = 0.0
        graph.eliminate_zeros()

    if return_stats:
        return graph, n_epochs, {
            "input_nnz": input_nnz,
            "nnz": int(graph.nnz),
            "removed_edges": int(input_nnz - graph.nnz),
            "reused_csr": False,
        }
    return graph, n_epochs


def _resolve_preprocess_epochs(n_vertices: int, n_epochs: Optional[int]) -> tuple[int, int]:
    default_epochs = 500 if int(n_vertices) <= 10000 else 200
    resolved_epochs = default_epochs if n_epochs is None else int(n_epochs)
    divisor = default_epochs if resolved_epochs <= 10 else resolved_epochs
    return resolved_epochs, divisor


def _csr_filter_mask_counts(indptr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    row_lengths = np.diff(indptr)
    counts = np.zeros(row_lengths.shape[0], dtype=indptr.dtype)
    nonempty = row_lengths > 0
    if np.any(nonempty):
        counts[nonempty] = np.add.reduceat(
            mask.astype(indptr.dtype, copy=False),
            indptr[:-1][nonempty],
        )
    return counts


def preprocess_graph_csr(
    graph,
    n_epochs: Optional[int],
    *,
    dtype=None,
    return_degrees: bool = False,
):
    """Preprocess a fuzzy graph directly in CSR form.

    This is equivalent to :func:`preprocess_graph` followed by ``.tocsr()``,
    but avoids the CSR -> COO -> CSR round-trip used by the original path.
    The input graph is reused only when it is already canonical CSR with the
    requested dtype and no threshold/zero filtering is needed.
    """

    if not issparse(graph):
        raise TypeError("fuzzy_graph must be a scipy sparse matrix")

    target_dtype = None if dtype is None else np.dtype(dtype)
    if isspmatrix_csr(graph):
        needs_dtype_cast = target_dtype is not None and graph.data.dtype != target_dtype
        needs_copy = needs_dtype_cast or not graph.has_canonical_format
        if needs_dtype_cast:
            graph_csr = graph.astype(target_dtype, copy=True)
        elif needs_copy:
            graph_csr = graph.copy()
        else:
            graph_csr = graph
    else:
        graph_csr = graph.tocsr()
        if target_dtype is not None and graph_csr.data.dtype != target_dtype:
            graph_csr = graph_csr.astype(target_dtype, copy=False)

    if not graph_csr.has_canonical_format:
        graph_csr.sum_duplicates()
        graph_csr.sort_indices()
    elif not graph_csr.has_sorted_indices:
        if graph_csr is graph:
            graph_csr = graph_csr.copy()
        graph_csr.sort_indices()

    if graph_csr.shape[0] != graph_csr.shape[1]:
        raise ValueError("fuzzy_graph must be square")

    resolved_epochs, divisor = _resolve_preprocess_epochs(
        graph_csr.shape[0],
        n_epochs,
    )
    input_nnz = int(graph_csr.nnz)
    reused_csr = graph_csr is graph

    if graph_csr.nnz > 0:
        data = graph_csr.data
        threshold = data.max() / float(divisor)
        keep = ~(data < threshold) & (data != 0)
        if not bool(np.all(keep)):
            counts = _csr_filter_mask_counts(graph_csr.indptr, keep)
            new_indptr = np.empty_like(graph_csr.indptr)
            new_indptr[0] = 0
            np.cumsum(counts, out=new_indptr[1:])
            graph_csr = csr_matrix(
                (
                    graph_csr.data[keep],
                    graph_csr.indices[keep],
                    new_indptr,
                ),
                shape=graph_csr.shape,
            )
            reused_csr = False
    elif graph_csr.nnz == 0 and graph_csr is graph and target_dtype is not None:
        graph_csr = graph_csr.astype(target_dtype, copy=False)

    degrees = (
        cal_degrees(graph_csr.shape[0], graph_csr.indptr, graph_csr.data)
        if return_degrees
        else None
    )
    stats = {
        "input_nnz": input_nnz,
        "nnz": int(graph_csr.nnz),
        "removed_edges": int(input_nnz - graph_csr.nnz),
        "reused_csr": bool(reused_csr),
    }
    return graph_csr, resolved_epochs, degrees, stats


def make_epochs_per_sample(
    weights: np.ndarray,
    n_epochs: int,
    dtype=CPU_SAMPLING_DTYPE,
) -> np.ndarray:
    """Build a host sampling calendar.

    CPU calendars default to float64 to match umap-learn, independently of
    the model's numeric dtype. Accelerator callers must opt in to float32 (or
    use :func:`make_epochs_per_sample_gpu`) because their kernels consume
    single-precision calendars.
    """
    dtype = CPU_SAMPLING_DTYPE if dtype is None else np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = DEVICE_SAMPLING_DTYPE
    epochs_per_sample = -1.0 * np.ones(weights.shape[0], dtype=dtype)
    n_samples = n_epochs * (weights / max(weights.max(), 1e-12))
    mask = n_samples > 0
    # Cast the denominator array explicitly. With NumPy 1.x, dividing a
    # float64 scalar by a float32 array can still produce float32 values that
    # are only widened on assignment. umap-learn explicitly promotes this
    # operand, so do the same to preserve its CPU sampling schedule.
    epochs_per_sample[mask] = dtype.type(n_epochs) / np.asarray(
        n_samples[mask], dtype=dtype
    )
    return epochs_per_sample


def make_epochs_per_sample_gpu(weights, n_epochs: int):
    import cupy as cp

    epochs_per_sample = -1.0 * cp.ones(weights.shape[0], dtype=cp.float32)
    max_w = cp.maximum(weights.max(), cp.float32(1e-12))
    n_samples = cp.float32(n_epochs) * (weights / max_w)
    mask = n_samples > 0
    epochs_per_sample[mask] = cp.float32(n_epochs) / n_samples[mask]
    return epochs_per_sample


def get_rng_state(random_state) -> np.ndarray:
    return random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)


def normalize_rng_state(rng_state) -> np.ndarray:
    """Return a private ``(3,)`` int64 copy for the low-level UMAP RNG."""
    state = np.asarray(rng_state)
    if state.shape != (3,):
        raise ValueError(f"rng_state must have shape (3,), got {state.shape}")
    if not np.issubdtype(state.dtype, np.integer):
        raise TypeError("rng_state must contain integers")
    if np.issubdtype(state.dtype, np.unsignedinteger) and np.any(
        state > np.iinfo(np.int64).max
    ):
        raise ValueError("rng_state values must fit in int64")
    return np.array(state, dtype=np.int64, order="C", copy=True)


def prepare_known_point_arrays(
    n_vertices: int,
    n_components: int,
    known_points_indices,
    known_points_positions,
    dtype=np.float32,
):
    dtype = np.dtype(dtype)
    whether_known_points = np.zeros(n_vertices, dtype=bool)
    reverse_index = np.full((n_vertices,), -1, dtype=np.int64)
    if known_points_indices is None or known_points_positions is None:
        # Kernels consult positions/reverse_index only when the corresponding
        # mask entry is true. Empty placeholders avoid two unused O(N)
        # allocations when constraints are disabled.
        positions = np.empty((0, n_components), dtype=dtype)
        return whether_known_points, positions, np.empty(0, dtype=np.int64)

    idx = np.asarray(known_points_indices, dtype=np.int64)
    pos = np.asarray(known_points_positions, dtype=dtype)
    whether_known_points[idx] = True
    reverse_index[idx] = np.arange(idx.shape[0], dtype=np.int64)
    return whether_known_points, pos, reverse_index


def cal_degrees(n_vertices, edgesrc, weights):
    degrees = np.zeros(int(n_vertices), dtype=np.asarray(weights).dtype)
    row_lengths = np.diff(edgesrc)
    nonempty = row_lengths > 0
    starts = np.asarray(edgesrc[:-1])[nonempty]
    if starts.size:
        degrees[nonempty] = np.add.reduceat(weights, starts)[: starts.size]
    return degrees
