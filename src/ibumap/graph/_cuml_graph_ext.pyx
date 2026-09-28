# distutils: language = c++

"""Optional cuML graph-only wrappers for CUDA ibUMAP.

This module is compiled only when ``IBUMAP_BUILD_CUML_GRAPH_EXT=1`` is set
during package build. It mirrors cuML.UMAP's large dense unseeded graph path by
calling ML::UMAP::get_graph with UMAPParams.build_algo=NN_DESCENT.
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse
import numpy as np

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibraft.common.handle cimport handle_t
from pylibraft.common.handle import Handle

cimport cuml.manifold.umap.lib as lib
from cuml.internals.array import CumlArray
from cuml.internals.logger cimport level_enum
from cuml.internals.validation import check_array
from cuml.metrics.distance_type cimport DistanceType


cdef class _RaftCOO:
    """Own a cuML RAFT COO graph and expose zero-copy CuPy views."""

    cdef unique_ptr[lib.COO] ptr

    def __dealloc__(self):
        self.ptr.reset(NULL)

    @staticmethod
    cdef _RaftCOO from_ptr(unique_ptr[lib.COO]& ptr):
        cdef _RaftCOO self = _RaftCOO.__new__(_RaftCOO)
        self.ptr = move(ptr)
        return self

    def view_cupy_coo(self, shape):
        cdef lib.COO* coo = self.ptr.get()

        def view_as_cupy(uintptr_t ptr, dtype):
            dtype = np.dtype(dtype)
            mem = cp.cuda.UnownedMemory(
                ptr,
                coo.nnz * dtype.itemsize,
                owner=self,
            )
            memptr = cp.cuda.MemoryPointer(mem, 0)
            return cp.ndarray(coo.nnz, dtype=dtype, memptr=memptr)

        vals = view_as_cupy(<uintptr_t> coo.vals(), np.float32)
        rows = view_as_cupy(<uintptr_t> coo.rows(), np.int32)
        cols = view_as_cupy(<uintptr_t> coo.cols(), np.int32)
        return cupyx.scipy.sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=shape,
        )


def fuzzy_simplicial_set_nn_descent(
    X,
    n_neighbors,
    random_state=None,
    metric="euclidean",
    metric_kwds=None,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    verbose=False,
    build_kwds=None,
):
    """Build only the cuML fuzzy graph using NN-Descent.

    Seed handling intentionally follows ``cuml.manifold.UMAP`` rather than
    the public ``fuzzy_simplicial_set`` helper: unseeded calls draw an unsigned
    32-bit seed and set ``deterministic=False``.
    """

    cdef lib.UMAPParams umap_params
    cdef handle_t* handle_
    cdef unique_ptr[lib.COO] fss_graph_ptr
    cdef _RaftCOO fss_graph
    cdef uintptr_t X_ptr
    cdef int n_rows
    cdef int n_dims

    metric_kwds = {} if metric_kwds is None else dict(metric_kwds)
    build_kwds = {} if build_kwds is None else dict(build_kwds)

    deterministic = random_state is not None
    if isinstance(random_state, np.uint64):
        seed = random_state
    else:
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(random_state)
        seed = rs.randint(
            low=0,
            high=np.iinfo(np.uint32).max,
            dtype=np.uint32,
        )

    graph_degree = max(
        int(build_kwds.get("nnd_graph_degree", 64)),
        int(n_neighbors),
    )
    intermediate_graph_degree = max(
        int(build_kwds.get("nnd_intermediate_graph_degree", 128)),
        graph_degree,
    )
    n_clusters = max(
        int(
            build_kwds.get(
                "knn_n_clusters",
                build_kwds.get("nnd_n_clusters", 1),
            )
        ),
        1,
    )

    umap_params.n_neighbors = <int> n_neighbors
    umap_params.set_op_mix_ratio = <float> set_op_mix_ratio
    umap_params.local_connectivity = <float> local_connectivity
    umap_params.verbosity = (
        level_enum.info if verbose else level_enum.warn
    )
    umap_params.build_algo = lib.graph_build_algo.NN_DESCENT
    umap_params.build_params.n_clusters = <size_t> n_clusters
    umap_params.build_params.overlap_factor = <size_t> max(
        int(build_kwds.get("knn_overlap_factor", 2)),
        1,
    )
    umap_params.build_params.nnd.graph_degree = <size_t> graph_degree
    umap_params.build_params.nnd.intermediate_graph_degree = (
        <size_t> intermediate_graph_degree
    )
    umap_params.build_params.nnd.max_iterations = <size_t> build_kwds.get(
        "nnd_max_iterations", 20
    )
    umap_params.build_params.nnd.termination_threshold = (
        <float> build_kwds.get("nnd_termination_threshold", 0.0001)
    )
    umap_params.random_state = <uint64_t> seed
    umap_params.deterministic = <bool> deterministic

    metric_name = str(metric).lower()
    if metric_name in {"euclidean", "l2"}:
        umap_params.metric = DistanceType.L2SqrtExpanded
    elif metric_name == "sqeuclidean":
        umap_params.metric = DistanceType.L2Expanded
    elif metric_name == "cosine":
        umap_params.metric = DistanceType.CosineExpanded
    else:
        raise NotImplementedError(
            f"Metric {metric!r} is not supported by cuML NN-Descent"
        )
    umap_params.p = <float> metric_kwds.get("p", 2.0)

    # cuML 26.06 requires host memory for the dense NN-Descent build path.
    X_host = check_array(
        X,
        mem_type="host",
        dtype="float32",
        convert_dtype=True,
        order="C",
        accept_sparse=False,
        ensure_min_samples=2,
        input_name="X",
    )
    X_m = CumlArray(data=X_host)
    X_ptr = <uintptr_t> X_m.ptr
    n_rows = <int> X_m.shape[0]
    n_dims = <int> X_m.shape[1]

    handle = Handle()
    handle_ = <handle_t*><size_t> handle.getHandle()
    fss_graph_ptr = lib.get_graph(
        handle_[0],
        <float*> X_ptr,
        <float*> 0,
        n_rows,
        n_dims,
        <int64_t*> 0,
        <float*> 0,
        &umap_params,
    )
    handle.sync()
    fss_graph = _RaftCOO.from_ptr(fss_graph_ptr)
    return fss_graph.view_cupy_coo((n_rows, n_rows))
