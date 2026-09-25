from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree
from time import perf_counter
import warnings

from ._spectral_defaults import (
    spectral_auto_defaults_enabled,
)
from ._spectral_cpu_impl import spectral_layout_cpu
from ._postprocess import (
    canonicalize_spectral_axes,
    compact_spectral_embedding,
    duplicate_row_stats,
    embedding_geometry,
)


_FINAL_JITTER_SCALE = 1e-4
_DUPLICATE_WARNING_RATIO = 0.01


def _add_timing(timing, key, value):
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0) + value)


def _set_diagnostic(diagnostics, key, value):
    if diagnostics is not None:
        diagnostics[key] = value


def _require_finite(name, values):
    values = np.asarray(values)
    nonfinite_count = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite_count:
        raise FloatingPointError(
            f"{name} contains {nonfinite_count} non-finite value(s)"
        )


def _duplicate_row_stats(values):
    return duplicate_row_stats(np, np.asarray(values))


def _record_duplicate_stage(diagnostics, stage, stats):
    for key, value in stats.items():
        _set_diagnostic(diagnostics, f"init_{stage}_{key}", value)


def _record_duplicate_diagnostics(diagnostics, stats):
    for key, value in stats.items():
        _set_diagnostic(diagnostics, f"init_{key}", value)
    if stats["duplicate_ratio"] > _DUPLICATE_WARNING_RATIO:
        warnings.warn(
            "Initialization contains "
            f"{stats['duplicate_rows']} duplicate row(s) "
            f"({stats['duplicate_ratio']:.2%}; "
            f"max group {stats['max_duplicate_group']}).",
            RuntimeWarning,
            stacklevel=3,
        )


def _record_spectral_geometry(diagnostics, stage, values):
    geometry = embedding_geometry(np, values)
    _set_diagnostic(
        diagnostics,
        f"init_spectral_{stage}_axis_spans",
        geometry["axis_spans"],
    )
    _set_diagnostic(
        diagnostics,
        f"init_spectral_{stage}_global_span",
        geometry["global_span"],
    )


def initialize_embedding_cpu(
    data,
    graph,
    n_components: int,
    init,
    random_state,
    metric,
    metric_kwds,
    dtype=np.float32,
    spectral_method=None,
    spectral_tol=None,
    spectral_maxiter=None,
    spectral_ncv=None,
    spectral_auto_defaults=None,
    spectral_scale_policy="legacy_box10",
    spectral_max_span=0.1,
    spectral_jitter_relative=1e-4,
    spectral_jitter_max=1e-4,
    report_duplicate_ratio=False,
    init_timing=None,
    init_diagnostics=None,
):
    dtype = np.dtype(dtype)
    spectral_scale_policy = str(spectral_scale_policy)
    if spectral_scale_policy == "auto":
        # The estimator resolves "auto" with its algorithm context. Preserve
        # the historical direct-function behavior when that context is absent.
        spectral_scale_policy = "legacy_box10"
    if spectral_scale_policy not in (
        "fft_compact", "raw", "legacy_box10", "legacy_box10_unoriented"
    ):
        raise ValueError(
            "spectral_scale_policy must be one of: fft_compact, raw, "
            "legacy_box10, legacy_box10_unoriented"
        )
    # "legacy_box10_unoriented" is legacy_box10 without the deterministic axis
    # reflection: the eigensolver's signs are kept. It reproduces spectral
    # initializations saved by versions that did not orient the axes.
    requested_scale_policy = spectral_scale_policy
    orient_axes = spectral_scale_policy != "legacy_box10_unoriented"
    if not orient_axes:
        spectral_scale_policy = "legacy_box10"
    is_spectral = isinstance(init, str) and init == "spectral"
    if isinstance(init, str) and init == "random":
        _set_diagnostic(init_diagnostics, "init_method", "random")
        t0 = perf_counter()
        init_embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(dtype)
        _add_timing(init_timing, "init_random_time", perf_counter() - t0)
    elif is_spectral:
        _set_diagnostic(init_diagnostics, "init_method", "spectral")
        _set_diagnostic(
            init_diagnostics,
            "init_spectral_scale_policy",
            requested_scale_policy,
        )
        auto_defaults = spectral_auto_defaults_enabled(
            init=init,
            spectral_auto_defaults=spectral_auto_defaults,
            method=spectral_method,
            tol=spectral_tol,
            maxiter=spectral_maxiter,
            ncv=spectral_ncv,
        )
        _set_diagnostic(init_diagnostics, "spectral_auto_defaults_enabled", auto_defaults)
        initialisation = spectral_layout_cpu(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
            method=spectral_method,
            tol=spectral_tol,
            maxiter=spectral_maxiter,
            ncv=spectral_ncv,
            auto_defaults=auto_defaults,
            timing=init_timing,
            diagnostics=init_diagnostics,
        )
        t0 = perf_counter()
        _require_finite("spectral initialization before global scaling", initialisation)
        _record_spectral_geometry(
            init_diagnostics,
            "raw",
            initialisation,
        )
        if spectral_scale_policy == "legacy_box10":
            scale_denominator = float(np.abs(initialisation).max())
            if not np.isfinite(scale_denominator) or scale_denominator <= 0.0:
                raise FloatingPointError(
                    "spectral initialization has an invalid global scaling denominator"
                )
            expansion = 10.0 / scale_denominator
            init_embedding = (initialisation * expansion).astype(dtype)
            _require_finite(
                "spectral initialization after global scaling",
                init_embedding,
            )
        else:
            init_embedding = np.array(
                initialisation,
                dtype=dtype,
                order="C",
                copy=True,
            )
        _add_timing(
            init_timing,
            "init_spectral_cast_scale_time",
            perf_counter() - t0,
        )
    else:
        _set_diagnostic(init_diagnostics, "init_method", "custom")
        t0 = perf_counter()
        init_data = np.array(init, dtype=dtype, order="C", copy=True)
        _add_timing(init_timing, "init_custom_copy_time", perf_counter() - t0)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                t0 = perf_counter()
                tree = KDTree(init_data)
                dist, _ = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                init_embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(dtype)
                _add_timing(init_timing, "init_custom_jitter_time", perf_counter() - t0)
            else:
                init_embedding = init_data
        else:
            raise ValueError("Unsupported init format")

    _require_finite("embedding before global normalization", init_embedding)
    if not is_spectral or spectral_scale_policy == "legacy_box10":
        t0 = perf_counter()
        min_v = np.min(init_embedding, 0)
        max_v = np.max(init_embedding, 0)
        _require_finite("embedding normalization bounds", np.stack([min_v, max_v]))
        init_embedding = (
            10.0
            * (init_embedding - min_v)
            / (max_v - min_v + 1e-12)
        ).astype(dtype, order="C")
        _require_finite("embedding after global normalization", init_embedding)
        _add_timing(init_timing, "init_normalize_time", perf_counter() - t0)

    if is_spectral and spectral_scale_policy == "legacy_box10":
        t0 = perf_counter()
        if orient_axes:
            init_embedding, orientation = canonicalize_spectral_axes(
                np,
                init_embedding,
            )
        else:
            orientation = {"reflected": [False] * int(init_embedding.shape[1])}
        for key, value in orientation.items():
            _set_diagnostic(
                init_diagnostics,
                f"init_axis_orientation_{key}",
                value,
            )
        _require_finite(
            "embedding after spectral axis canonicalization",
            init_embedding,
        )
        _add_timing(
            init_timing,
            "init_axis_canonicalization_time",
            perf_counter() - t0,
        )

        # Preserve the previous policy exactly as an explicit regression
        # comparison and rollback path.
        t0 = perf_counter()
        init_embedding = init_embedding + random_state.normal(
            scale=_FINAL_JITTER_SCALE,
            size=init_embedding.shape,
        ).astype(dtype)
        _require_finite("embedding after final jitter", init_embedding)
        _add_timing(init_timing, "init_jitter_time", perf_counter() - t0)
        _set_diagnostic(init_diagnostics, "init_final_jitter_scale", _FINAL_JITTER_SCALE)
    elif is_spectral and spectral_scale_policy == "fft_compact":
        t0 = perf_counter()
        init_embedding, compact_diagnostics = compact_spectral_embedding(
            np,
            init_embedding,
            random_state,
            max_span=float(spectral_max_span),
            jitter_relative=float(spectral_jitter_relative),
            jitter_max=float(spectral_jitter_max),
        )
        orientation = compact_diagnostics.pop("axis_orientation")
        for key, value in orientation.items():
            _set_diagnostic(
                init_diagnostics,
                f"init_axis_orientation_{key}",
                value,
            )
        for key, value in compact_diagnostics.items():
            _set_diagnostic(
                init_diagnostics,
                f"init_spectral_{key}",
                value,
            )
        _set_diagnostic(
            init_diagnostics,
            "init_final_jitter_scale",
            compact_diagnostics["final_jitter_scale"],
        )
        _require_finite("embedding after compact spectral postprocessing", init_embedding)
        _add_timing(
            init_timing,
            "init_spectral_compact_time",
            perf_counter() - t0,
        )
    elif is_spectral:
        _set_diagnostic(init_diagnostics, "init_final_jitter_scale", 0.0)

    if is_spectral:
        _record_spectral_geometry(
            init_diagnostics,
            "final",
            init_embedding,
        )

    if report_duplicate_ratio:
        t0 = perf_counter()
        final_stats = _duplicate_row_stats(init_embedding)
        _add_timing(
            init_timing,
            "init_duplicate_ratio_time",
            perf_counter() - t0,
        )
        _record_duplicate_stage(
            init_diagnostics,
            "post_jitter",
            final_stats,
        )
        _record_duplicate_diagnostics(
            init_diagnostics,
            final_stats,
        )

    return np.asarray(init_embedding, dtype=dtype, order="C")
