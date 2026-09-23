"""MLX/Metal attraction and degree-damping operators for ibUMAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .runtime import mlx_core


@dataclass(frozen=True, slots=True)
class MetalAttractionResult:
    force: Any
    epoch_of_next_sample: Any


_SAMPLING_ATTRACTION_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    float a = scalars[0];
    float b = scalars[1];
    float alpha = scalars[2];
    float epoch = scalars[3];
    float temp0 = 0.0f;
    float temp1 = 0.0f;
    for (int edge = edgesrc[i]; edge < edgesrc[i + 1]; ++edge) {
        float next_value = epoch_of_next_sample[edge];
        if (next_value <= epoch) {
            next_value += epochs_per_sample[edge];
            int target = edgetgt[edge];
            float delta0 = embedding[2 * i] - embedding[2 * target];
            float delta1 = embedding[2 * i + 1] - embedding[2 * target + 1];
            float dist2 = delta0 * delta0 + delta1 * delta1;
            float grad = 0.0f;
            if (dist2 > 0.0f) {
                float distance_power = metal::pow(dist2, b);
                grad = -2.0f * a * b * (distance_power / dist2);
                grad /= a * distance_power + 1.0f;
            }
            temp0 += alpha * metal::max(
                -4.0f, metal::min(4.0f, grad * delta0));
            temp1 += alpha * metal::max(
                -4.0f, metal::min(4.0f, grad * delta1));
        }
        next_out[edge] = next_value;
    }
    force[2 * i] = temp0;
    force[2 * i + 1] = temp1;
"""


def build_sampling_attraction_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_sampling_attraction",
        input_names=[
            "embedding",
            "edgesrc",
            "edgetgt",
            "epochs_per_sample",
            "epoch_of_next_sample",
            "scalars",
        ],
        output_names=["force", "next_out"],
        source=_SAMPLING_ATTRACTION_SOURCE,
    )


def sampling_attraction(
    embedding: Any,
    edgesrc: Any,
    edgetgt: Any,
    epochs_per_sample: Any,
    epoch_of_next_sample: Any,
    *,
    a: float,
    b: float,
    alpha: Any,
    epoch: int,
    mx: Any = None,
    kernel: Any = None,
) -> MetalAttractionResult:
    """Accumulate active CSR positive edges into their source rows."""

    mx = mlx_core() if mx is None else mx
    kernel = build_sampling_attraction_kernel(mx=mx) if kernel is None else kernel
    n_vertices = int(embedding.shape[0])
    edge_count = int(edgetgt.shape[0])
    scalars = mx.stack(
        (
            mx.array(float(a), dtype=mx.float32),
            mx.array(float(b), dtype=mx.float32),
            mx.array(alpha, dtype=mx.float32),
            mx.array(float(epoch), dtype=mx.float32),
        )
    )
    outputs = kernel(
        inputs=[
            embedding,
            edgesrc,
            edgetgt,
            epochs_per_sample,
            epoch_of_next_sample,
            scalars,
        ],
        output_shapes=[tuple(embedding.shape), (edge_count,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_vertices, 1, 1),
        threadgroup=(min(256, n_vertices), 1, 1),
        init_value=0,
    )
    return MetalAttractionResult(force=outputs[0], epoch_of_next_sample=outputs[1])


def weighted_degrees(edgesrc: Any, weights: Any, *, mx: Any = None) -> Any:
    """Compute CSR row sums without host materialization."""

    mx = mlx_core() if mx is None else mx
    prefix = mx.concatenate(
        (mx.zeros((1,), dtype=mx.float32), mx.cumsum(weights))
    )
    return prefix[edgesrc[1:]] - prefix[edgesrc[:-1]]


def degree_damping_scale(
    degree_values: Any,
    *,
    degree_ref: float,
    power: float,
    min_scale: float | None,
    mx: Any = None,
) -> Any:
    """Apply the CPU ibUMAP degree-damping formula on the device."""

    if not np.isfinite(degree_ref) or float(degree_ref) <= 0.0:
        raise ValueError("degree_ref must be positive and finite")
    if not np.isfinite(power) or float(power) <= 0.0:
        raise ValueError("power must be positive and finite")
    if min_scale is not None and not 0.0 < float(min_scale) <= 1.0:
        raise ValueError("min_scale must be in (0, 1]")
    mx = mlx_core() if mx is None else mx
    positive = degree_values > 0.0
    ratio = mx.where(
        positive,
        mx.array(float(degree_ref), dtype=mx.float32)
        / mx.maximum(degree_values, mx.array(1e-12, dtype=mx.float32)),
        mx.ones_like(degree_values),
    )
    scale = mx.minimum(
        mx.ones_like(ratio),
        mx.power(ratio, mx.array(float(power), dtype=mx.float32)),
    )
    scale = mx.where(positive, scale, mx.ones_like(scale))
    if min_scale is not None:
        scale = mx.maximum(
            scale, mx.array(float(min_scale), dtype=mx.float32)
        )
    return scale


__all__ = [
    "MetalAttractionResult",
    "build_sampling_attraction_kernel",
    "degree_damping_scale",
    "sampling_attraction",
    "weighted_degrees",
]
