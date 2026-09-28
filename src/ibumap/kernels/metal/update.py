"""Unfused Metal optimizer update helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .runtime import finite_count, mlx_core


_FUSED_UPDATE_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    float update_x = attraction[2 * i] + repulsion[2 * i];
    float update_y = attraction[2 * i + 1] + repulsion[2 * i + 1];
    if (params[0] > 0.5f) {
        float norm = metal::sqrt(update_x * update_x + update_y * update_y);
        float scale = metal::min(1.0f, params[1] / (norm + 1.0e-12f));
        update_x *= scale;
        update_y *= scale;
    }
    if (known_mask[i]) {
        output[2 * i] = embedding[2 * i];
        output[2 * i + 1] = embedding[2 * i + 1];
    } else {
        output[2 * i] = embedding[2 * i] + update_x;
        output[2 * i + 1] = embedding[2 * i + 1] + update_y;
    }
"""


def alpha_for_epoch(initial_alpha: float, epoch: int, n_epochs: int) -> float:
    """Return the alpha used by the existing CPU ibUMAP epoch loop.

    The CPU loop updates alpha at the end of an epoch using that epoch's index,
    so epochs zero and one both use the initial value.
    """

    if int(n_epochs) <= 0:
        raise ValueError("n_epochs must be positive")
    if int(epoch) < 0 or int(epoch) >= int(n_epochs):
        raise ValueError("epoch must be in [0, n_epochs)")
    previous_epoch = max(0, int(epoch) - 1)
    return float(initial_alpha) * (
        1.0 - float(previous_epoch) / float(n_epochs)
    )


def apply_optimizer_update(
    embedding: Any,
    attraction_force: Any,
    repulsive_force: Any,
    *,
    total_clip_norm: float | None = None,
    known_mask: Any = None,
    mx: Any = None,
) -> Any:
    """Apply the deliberately unfused ``embedding += attr + repl`` update."""

    if tuple(attraction_force.shape) != tuple(embedding.shape):
        raise ValueError("attraction_force shape must match embedding")
    if tuple(repulsive_force.shape) != tuple(embedding.shape):
        raise ValueError("repulsive_force shape must match embedding")
    mx = mlx_core() if mx is None else mx
    update = attraction_force + repulsive_force
    if total_clip_norm is not None:
        if not np.isfinite(total_clip_norm) or float(total_clip_norm) <= 0.0:
            raise ValueError("total_clip_norm must be positive and finite")
        norms = mx.sqrt(mx.sum(update * update, axis=1))
        scale = mx.minimum(
            mx.ones_like(norms),
            mx.array(float(total_clip_norm), dtype=mx.float32)
            / (norms + 1e-12),
        )
        update = update * scale[:, None]
    result = embedding + update
    if known_mask is not None:
        if tuple(known_mask.shape) != (int(embedding.shape[0]),):
            raise ValueError("known_mask must contain one value per vertex")
        result = mx.where(known_mask[:, None], embedding, result)
    return result


def require_finite(value: Any, *, label: str, mx: Any = None) -> None:
    """Raise at an intentional synchronization/validation boundary."""

    count = finite_count(value, mx=mx)
    if count:
        raise FloatingPointError(f"{label} contains {count} non-finite value(s)")


def build_fused_update_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_fused_total_update",
        input_names=[
            "embedding", "attraction", "repulsion", "known_mask", "params"
        ],
        output_names=["output"],
        source=_FUSED_UPDATE_SOURCE,
    )


def apply_optimizer_update_fused(
    embedding: Any,
    attraction_force: Any,
    repulsive_force: Any,
    *,
    total_clip_norm: float | None,
    known_mask: Any,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Fuse force addition, optional norm clip, hard mask, and embedding add."""

    if tuple(attraction_force.shape) != tuple(embedding.shape):
        raise ValueError("attraction_force shape must match embedding")
    if tuple(repulsive_force.shape) != tuple(embedding.shape):
        raise ValueError("repulsive_force shape must match embedding")
    mx = mlx_core() if mx is None else mx
    kernel = build_fused_update_kernel(mx=mx) if kernel is None else kernel
    params = mx.array(
        np.asarray(
            [
                0.0 if total_clip_norm is None else 1.0,
                0.0 if total_clip_norm is None else float(total_clip_norm),
            ],
            dtype=np.float32,
        )
    )
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[
            embedding, attraction_force, repulsive_force, known_mask, params
        ],
        output_shapes=[tuple(embedding.shape)],
        output_dtypes=[mx.float32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
    )[0]


__all__ = [
    "alpha_for_epoch",
    "apply_optimizer_update",
    "apply_optimizer_update_fused",
    "build_fused_update_kernel",
    "require_finite",
]
