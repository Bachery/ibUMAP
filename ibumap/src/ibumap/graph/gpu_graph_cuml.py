from __future__ import annotations

import os

from sklearn.utils import check_random_state


_UNSEEDED_OVERFLOW_RETRIES = 16
_UNSEEDED_OVERFLOW_MESSAGE = "can't convert negative value to uint64_t"
_CUML_AUTO_NN_DESCENT_THRESHOLD = 50_000


def _optional_graph_ext():
    if os.environ.get("UMAP_FFT_DISABLE_CUML_GRAPH_EXT") == "1":
        if os.environ.get("UMAP_FFT_REQUIRE_CUML_GRAPH_EXT") == "1":
            raise RuntimeError(
                "Both UMAP_FFT_DISABLE_CUML_GRAPH_EXT=1 and "
                "UMAP_FFT_REQUIRE_CUML_GRAPH_EXT=1 are set"
            )
        return None
    try:
        from umap_fft.graph import _cuml_graph_ext
    except (ImportError, OSError) as exc:
        if os.environ.get("UMAP_FFT_REQUIRE_CUML_GRAPH_EXT") == "1":
            raise RuntimeError(
                "UMAP_FFT_REQUIRE_CUML_GRAPH_EXT=1 but the optional cuML "
                "graph-only extension could not be imported"
            ) from exc
        return None
    return _cuml_graph_ext


def _should_use_nn_descent_graph(X, random_state) -> bool:
    if random_state is not None:
        return False
    n_samples = int(getattr(X, "shape", (0,))[0])
    return n_samples > _CUML_AUTO_NN_DESCENT_THRESHOLD


def _as_cupy_float32(X):
    import cupy as cp

    return cp.asarray(X, dtype=cp.float32)


def build_gpu_graph_cuml(
    X,
    n_neighbors: int,
    metric: str,
    random_state,
    verbose: bool,
    metric_kwds=None,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
):
    from cuml.manifold.umap import fuzzy_simplicial_set as cuml_fuzzy_simplicial_set

    metric_kwds = dict(metric_kwds or {})

    if _should_use_nn_descent_graph(X, random_state):
        graph_ext = _optional_graph_ext()
        if graph_ext is not None:
            return graph_ext.fuzzy_simplicial_set_nn_descent(
                X,
                n_neighbors=n_neighbors,
                random_state=None,
                metric=metric,
                metric_kwds=metric_kwds,
                set_op_mix_ratio=set_op_mix_ratio,
                local_connectivity=local_connectivity,
                verbose=verbose,
            )

    # The public cuML helper uses the brute-force build path. Keep its input on
    # the device, while leaving the large unseeded NN-Descent path above on the
    # host as required by cuML 26.06.
    X_device = _as_cupy_float32(X)
    kwargs = {
        "n_neighbors": n_neighbors,
        "metric": metric,
        "metric_kwds": metric_kwds,
        "set_op_mix_ratio": set_op_mix_ratio,
        "local_connectivity": local_connectivity,
        "verbose": verbose,
    }
    if random_state is not None:
        rs = check_random_state(random_state)
        kwargs["random_state"] = rs.randint(1, 100000)
        return cuml_fuzzy_simplicial_set(X_device, **kwargs)

    for attempt in range(_UNSEEDED_OVERFLOW_RETRIES):
        try:
            return cuml_fuzzy_simplicial_set(X_device, **kwargs)
        except OverflowError as exc:
            # Some cuML builds draw an invalid negative uint64 seed in the
            # unseeded default path. Retrying keeps the no-keyword fast path.
            if _UNSEEDED_OVERFLOW_MESSAGE not in str(exc):
                raise
            if attempt + 1 == _UNSEEDED_OVERFLOW_RETRIES:
                raise

    raise RuntimeError("unreachable")
