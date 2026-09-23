from __future__ import annotations

from typing import Any


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _float_list(values: Any) -> list[float]:
    return [float(_scalar(values[index])) for index in range(int(values.shape[0]))]


def embedding_geometry(
    array_module: Any,
    values: Any,
) -> dict[str, Any]:
    if values.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if values.shape[0] == 0:
        return {
            "axis_min": [],
            "axis_max": [],
            "axis_spans": [],
            "global_min": 0.0,
            "global_max": 0.0,
            "global_span": 0.0,
        }

    axis_min = array_module.min(values, axis=0)
    axis_max = array_module.max(values, axis=0)
    global_min = float(_scalar(array_module.min(values)))
    global_max = float(_scalar(array_module.max(values)))
    return {
        "axis_min": _float_list(axis_min),
        "axis_max": _float_list(axis_max),
        "axis_spans": _float_list(axis_max - axis_min),
        "global_min": global_min,
        "global_max": global_max,
        # This is the exact scalar range used by the ibFFT grid.
        "global_span": global_max - global_min,
    }


def canonicalize_spectral_axes(
    array_module: Any,
    values: Any,
) -> tuple[Any, dict[str, Any]]:
    """Choose a deterministic reflection for each normalized spectral axis.

    Eigensolver vectors have arbitrary signs.  After min-max normalization,
    changing one sign maps an axis ``x`` to ``low + high - x``.  Orienting the
    denser side toward the lower endpoint prevents an arbitrary sign choice
    from moving millions of values to the lower-precision float32 region near
    10 while preserving every pairwise distance.
    """

    if values.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if values.shape[0] == 0:
        return values, {
            "means_before": [],
            "midpoints": [],
            "reflected": [],
            "means_after": [],
        }

    low = array_module.min(values, axis=0)
    high = array_module.max(values, axis=0)
    means_before = array_module.mean(values, axis=0, dtype=array_module.float64)
    midpoints = (
        low.astype(array_module.float64) + high.astype(array_module.float64)
    ) / 2.0
    reflected = [
        bool(_scalar(means_before[axis] > midpoints[axis]))
        for axis in range(int(values.shape[1]))
    ]

    for axis, should_reflect in enumerate(reflected):
        if should_reflect:
            values[:, axis] = (
                low[axis] + high[axis] - values[:, axis]
            ).astype(values.dtype)

    means_after = array_module.mean(values, axis=0, dtype=array_module.float64)
    return values, {
        "means_before": _float_list(means_before),
        "midpoints": _float_list(midpoints),
        "reflected": reflected,
        "means_after": _float_list(means_after),
    }


def compact_spectral_embedding(
    array_module: Any,
    values: Any,
    random_state: Any,
    *,
    max_span: float,
    jitter_relative: float,
    jitter_max: float,
) -> tuple[Any, dict[str, Any]]:
    """Finalize a spectral layout without expanding its FFT grid range.

    The raw eigensolver geometry is preserved up to translation, reflection,
    and a uniform shrink.  In particular, layouts smaller than ``max_span`` are
    never expanded.  Exact duplicate rows are allowed; the small relative
    jitter merely avoids completely collapsed inputs in the common case.
    """

    if values.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    raw_geometry = embedding_geometry(array_module, values)
    if values.shape[0] == 0:
        return values, {
            "raw_axis_spans": raw_geometry["axis_spans"],
            "raw_global_span": raw_geometry["global_span"],
            "bbox_center": [],
            "scale_factor": 1.0,
            "degenerate_fallback": False,
            "final_jitter_scale": 0.0,
            "pre_jitter_global_span": 0.0,
            "final_axis_spans": [],
            "final_global_span": 0.0,
            "axis_orientation": {
                "means_before": [],
                "midpoints": [],
                "reflected": [],
                "means_after": [],
            },
        }

    axis_min = array_module.min(values, axis=0)
    axis_max = array_module.max(values, axis=0)
    bbox_center = (
        axis_min.astype(array_module.float64)
        + axis_max.astype(array_module.float64)
    ) / 2.0
    values = (
        values - bbox_center.astype(values.dtype)
    ).astype(values.dtype)
    values, orientation = canonicalize_spectral_axes(array_module, values)

    centered_geometry = embedding_geometry(array_module, values)
    centered_span = float(centered_geometry["global_span"])
    degenerate_fallback = centered_span <= 0.0
    if degenerate_fallback:
        half_span = 0.5 * float(max_span)
        values = random_state.uniform(
            low=-half_span,
            high=half_span,
            size=values.shape,
        ).astype(values.dtype)
        centered_geometry = embedding_geometry(array_module, values)
        centered_span = float(centered_geometry["global_span"])

    scale_factor = (
        min(1.0, float(max_span) / centered_span)
        if centered_span > 0.0
        else 1.0
    )
    if scale_factor < 1.0:
        values = (values * scale_factor).astype(values.dtype)

    pre_jitter_geometry = embedding_geometry(array_module, values)
    pre_jitter_span = float(pre_jitter_geometry["global_span"])
    jitter_scale = min(
        float(jitter_max),
        float(jitter_relative) * pre_jitter_span,
    )
    if jitter_scale > 0.0:
        values = values + random_state.normal(
            scale=jitter_scale,
            size=values.shape,
        ).astype(values.dtype)

    final_geometry = embedding_geometry(array_module, values)
    return values, {
        "raw_axis_spans": raw_geometry["axis_spans"],
        "raw_global_span": raw_geometry["global_span"],
        "bbox_center": _float_list(bbox_center),
        "scale_factor": scale_factor,
        "degenerate_fallback": degenerate_fallback,
        "final_jitter_scale": jitter_scale,
        "pre_jitter_global_span": pre_jitter_span,
        "final_axis_spans": final_geometry["axis_spans"],
        "final_global_span": final_geometry["global_span"],
        "axis_orientation": orientation,
    }


def _duplicate_info(
    array_module: Any,
    values: Any,
) -> dict[str, Any]:
    if values.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    n_rows = int(values.shape[0])
    if n_rows < 2:
        return {
            "unique_rows": n_rows,
            "duplicate_rows": 0,
            "duplicate_ratio": 0.0,
            "max_duplicate_group": n_rows,
        }

    # Both NumPy and CuPy treat the final lexsort key as primary.  Reverse the
    # coordinate rows so column zero remains the primary key.
    order = array_module.lexsort(values.T[::-1])
    ordered = values[order]
    duplicate_pairs = array_module.all(
        ordered[1:] == ordered[:-1],
        axis=1,
    )
    duplicate_rows = int(_scalar(array_module.count_nonzero(duplicate_pairs)))
    starts = array_module.concatenate(
        (array_module.asarray([True]), ~duplicate_pairs)
    )
    boundaries = array_module.concatenate(
        (
            array_module.flatnonzero(starts),
            array_module.asarray([n_rows]),
        )
    )
    group_sizes = boundaries[1:] - boundaries[:-1]
    return {
        "unique_rows": n_rows - duplicate_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": float(duplicate_rows / n_rows),
        "max_duplicate_group": int(_scalar(array_module.max(group_sizes))),
    }


def duplicate_row_stats(array_module: Any, values: Any) -> dict[str, Any]:
    return _duplicate_info(array_module, values)
