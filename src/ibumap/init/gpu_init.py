from __future__ import annotations

import numpy as np
from time import perf_counter
import warnings

from ._custom_init import custom_init_jitter_scale
from ._spectral_defaults import (
    spectral_auto_defaults_enabled,
)
from ._spectral_cupy_impl import spectral_layout_cupy
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


def _require_finite(cp, name, values):
    nonfinite_count = int(cp.count_nonzero(~cp.isfinite(values)).item())
    if nonfinite_count:
        raise FloatingPointError(
            f"{name} contains {nonfinite_count} non-finite value(s)"
        )


def _duplicate_row_stats(cp, values):
    return duplicate_row_stats(cp, values)


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


def _record_spectral_geometry(cp, diagnostics, stage, values):
    geometry = embedding_geometry(cp, values)
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


def initialize_embedding_gpu(
    data,
    graph_gpu,
    n_components: int,
    init,
    random_state,
    metric,
    metric_kwds,
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
    deterministic=False,
):
    import cupy as cp

    spectral_scale_policy = str(spectral_scale_policy)
    if spectral_scale_policy == "auto":
        spectral_scale_policy = "legacy_box10"
    if spectral_scale_policy not in ("fft_compact", "raw", "legacy_box10"):
        raise ValueError(
            "spectral_scale_policy must be one of: fft_compact, raw, "
            "legacy_box10"
        )
    is_spectral = isinstance(init, str) and init == "spectral"
    if isinstance(random_state, np.random.RandomState):
        random_seed = int(random_state.randint(0, 2**31 - 1))
    elif isinstance(random_state, int):
        random_seed = int(random_state)
    elif random_state is None:
        random_seed = None
    else:
        random_seed = int(random_state.randint(0, 2**31 - 1))
    rs = cp.random.RandomState(random_seed) if random_seed is not None else cp.random

    if isinstance(init, str) and init == "random":
        _set_diagnostic(init_diagnostics, "init_method", "random")
        t0 = perf_counter()
        init_embedding = rs.uniform(
            low=-10.0,
            high=10.0,
            size=(graph_gpu.shape[0], n_components),
        ).astype(cp.float32)
        _add_timing(init_timing, "init_random_time", perf_counter() - t0)
    elif is_spectral:
        _set_diagnostic(init_diagnostics, "init_method", "spectral")
        _set_diagnostic(
            init_diagnostics,
            "init_spectral_scale_policy",
            spectral_scale_policy,
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
        initialisation = spectral_layout_cupy(
            data,
            graph_gpu,
            n_components,
            rs,
            metric=metric,
            metric_kwds=metric_kwds or {},
            method=spectral_method,
            tol=spectral_tol,
            maxiter=spectral_maxiter,
            ncv=spectral_ncv,
            auto_defaults=auto_defaults,
            deterministic=deterministic,
            timing=init_timing,
            diagnostics=init_diagnostics,
        )
        t0 = perf_counter()
        _require_finite(cp, "spectral initialization before global scaling", initialisation)
        _record_spectral_geometry(
            cp,
            init_diagnostics,
            "raw",
            initialisation,
        )
        if spectral_scale_policy == "legacy_box10":
            scale_denominator = float(cp.max(cp.abs(initialisation)).item())
            if not np.isfinite(scale_denominator) or scale_denominator <= 0.0:
                raise FloatingPointError(
                    "spectral initialization has an invalid global scaling denominator"
                )
            expansion = 10.0 / scale_denominator
            init_embedding = (initialisation * expansion).astype(cp.float32)
            _require_finite(
                cp,
                "spectral initialization after global scaling",
                init_embedding,
            )
        else:
            init_embedding = cp.array(
                initialisation,
                dtype=cp.float32,
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
        init_embedding = cp.asarray(init, dtype=cp.float32)
        _add_timing(init_timing, "init_custom_to_gpu_time", perf_counter() - t0)

        t0 = perf_counter()
        jitter_scale = custom_init_jitter_scale(cp, init_embedding)
        _set_diagnostic(
            init_diagnostics,
            "init_custom_duplicates_detected",
            jitter_scale is not None,
        )
        _add_timing(init_timing, "init_custom_prepare_time", perf_counter() - t0)
        if jitter_scale is not None:
            t0 = perf_counter()
            init_embedding = init_embedding + (
                rs.normal(size=init_embedding.shape).astype(cp.float32)
                * jitter_scale.astype(cp.float32)
            )
            _add_timing(
                init_timing,
                "init_custom_jitter_time",
                perf_counter() - t0,
            )

    _require_finite(cp, "embedding before global normalization", init_embedding)
    if not is_spectral or spectral_scale_policy == "legacy_box10":
        t0 = perf_counter()
        min_v = cp.min(init_embedding, axis=0)
        max_v = cp.max(init_embedding, axis=0)
        _require_finite(cp, "embedding normalization bounds", cp.stack([min_v, max_v]))
        init_embedding = 10.0 * (init_embedding - min_v) / (max_v - min_v + 1e-12)
        init_embedding = init_embedding.astype(cp.float32)
        _require_finite(cp, "embedding after global normalization", init_embedding)
        _add_timing(init_timing, "init_normalize_time", perf_counter() - t0)

    if is_spectral and spectral_scale_policy == "legacy_box10":
        t0 = perf_counter()
        init_embedding, orientation = canonicalize_spectral_axes(
            cp,
            init_embedding,
        )
        for key, value in orientation.items():
            _set_diagnostic(
                init_diagnostics,
                f"init_axis_orientation_{key}",
                value,
            )
        _require_finite(
            cp,
            "embedding after spectral axis canonicalization",
            init_embedding,
        )
        _add_timing(
            init_timing,
            "init_axis_canonicalization_time",
            perf_counter() - t0,
        )

        t0 = perf_counter()
        init_embedding = init_embedding + rs.normal(
            scale=_FINAL_JITTER_SCALE,
            size=init_embedding.shape,
        ).astype(cp.float32)
        _require_finite(cp, "embedding after final jitter", init_embedding)
        _add_timing(init_timing, "init_jitter_time", perf_counter() - t0)
        _set_diagnostic(init_diagnostics, "init_final_jitter_scale", _FINAL_JITTER_SCALE)
    elif is_spectral and spectral_scale_policy == "fft_compact":
        t0 = perf_counter()
        init_embedding, compact_diagnostics = compact_spectral_embedding(
            cp,
            init_embedding,
            rs,
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
        _require_finite(
            cp,
            "embedding after compact spectral postprocessing",
            init_embedding,
        )
        _add_timing(
            init_timing,
            "init_spectral_compact_time",
            perf_counter() - t0,
        )
    elif is_spectral:
        _set_diagnostic(init_diagnostics, "init_final_jitter_scale", 0.0)

    if is_spectral:
        _record_spectral_geometry(
            cp,
            init_diagnostics,
            "final",
            init_embedding,
        )

    if report_duplicate_ratio:
        t0 = perf_counter()
        final_stats = _duplicate_row_stats(cp, init_embedding)
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
    return cp.ascontiguousarray(init_embedding, dtype=cp.float32)
