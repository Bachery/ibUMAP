"""MLX ibUMAP optimizer assembled from validated Stage-3 operators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..fft_schedule import (
    fft_stage_index_for_epoch,
    resolve_fft_schedule,
    validate_fft_schedule_execution,
)
from ..kernels.metal.attraction import sampling_attraction
from ..kernels.metal.ibfft import ibfft_repulsive_force, scale_repulsive_force
from ..kernels.metal.runtime import evaluate, mlx_core, to_numpy
from ..kernels.metal.update import (
    alpha_for_epoch,
    apply_optimizer_update,
    apply_optimizer_update_fused,
    build_fused_update_kernel,
    require_finite,
)


class MetalOptimizerNotImplementedError(NotImplementedError):
    """Retained for compatibility with Stage-2 callers and error imports."""


@dataclass(frozen=True, slots=True)
class MetalOptimizerParameters:
    a: float
    b: float
    gamma: float = 1.0
    epsilon: float = 1e-3
    kernel_clip: float = 1.0
    intervals_per_integer: float = 1.0
    min_num_intervals: int = 100
    n_boxes_per_dim: float = 1.0
    p2m_mode: str = "atomic"
    n_interpolation_points: int = 1
    combine_stages: bool = False
    interpolation_schedule: Any = None
    initial_alpha: float = 1.0
    negative_sample_rate: float = 5.0
    repulsion_clip_norm: float | None = None
    repulsion_clip_with_alpha: bool = False
    repulsion_clip_epoch_range: tuple[int, int] | None = None
    total_update_clip_norm: float | None = None
    total_update_clip_with_alpha: bool = False
    total_update_clip_epoch_range: tuple[int, int] | None = None
    known_points_indices: Any = None
    noise_mode: str = "none"
    noise_scale: float = 0.0
    noise_decay: str = "linear"
    noise_until_epoch: int | None = None
    noise_seed: int | None = None
    hybrid_mode: str = "none"
    hybrid_switch_epoch: int | None = None
    fused_update: bool = True
    degree_damping_enabled: bool = True
    degree_damping_mode: str = "weighted_degree"
    degree_damping_ref: str = "p99"
    degree_damping_ref_value: float | None = None
    degree_damping_power: float = 0.5
    degree_damping_min_scale: float | None = None


@dataclass(frozen=True, slots=True)
class MetalOptimizerCoreResult:
    embedding: np.ndarray
    epoch_of_next_sample: np.ndarray
    checkpoints: dict[int, np.ndarray]
    timings: dict[str, float]
    degree_ref: float


def _host_degree_damping(
    edgesrc: np.ndarray,
    weights: np.ndarray,
    n_vertices: int,
    params: MetalOptimizerParameters,
) -> tuple[np.ndarray, float, np.ndarray]:
    if params.degree_damping_mode == "weighted_degree":
        prefix = np.concatenate(
            (
                np.zeros(1, dtype=np.float32),
                np.cumsum(weights, dtype=np.float32),
            )
        )
        values = prefix[edgesrc[1:]] - prefix[edgesrc[:-1]]
    elif params.degree_damping_mode == "degree":
        values = (edgesrc[1:] - edgesrc[:-1]).astype(np.float32)
    else:
        raise ValueError("degree_damping_mode must be 'weighted_degree' or 'degree'")
    if len(values) != n_vertices:
        raise ValueError("degree values must contain one value per vertex")
    if params.degree_damping_ref == "manual":
        if params.degree_damping_ref_value is None:
            raise ValueError("manual degree damping requires degree_damping_ref_value")
        degree_ref = float(params.degree_damping_ref_value)
    elif params.degree_damping_ref == "mean":
        degree_ref = float(np.mean(values))
    elif params.degree_damping_ref == "median":
        degree_ref = float(np.median(values))
    elif params.degree_damping_ref in {"p90", "p95", "p99"}:
        percentile = {"p90": 90.0, "p95": 95.0, "p99": 99.0}[
            params.degree_damping_ref
        ]
        degree_ref = float(np.percentile(values, percentile))
    else:
        raise ValueError("unsupported degree_damping_ref")
    if params.degree_damping_enabled and (
        not np.isfinite(degree_ref) or degree_ref <= 0.0
    ):
        raise ValueError("degree damping requires a positive degree reference")
    if not params.degree_damping_enabled:
        scale = np.ones(n_vertices, dtype=np.float32)
    else:
        positive = values > 0.0
        ratio = np.where(
            positive,
            np.float32(degree_ref) / np.maximum(values, np.float32(1e-12)),
            np.float32(1.0),
        )
        scale = np.minimum(
            np.float32(1.0),
            np.power(ratio, np.float32(params.degree_damping_power)),
        ).astype(np.float32)
        scale = np.where(positive, scale, np.float32(1.0))
        if params.degree_damping_min_scale is not None:
            scale = np.maximum(
                scale, np.float32(params.degree_damping_min_scale)
            )
    return values.astype(np.float32), degree_ref, scale.astype(np.float32)


def _validate_core_inputs(
    init_embedding: Any,
    edgesrc: Any,
    edgetgt: Any,
    weights: Any,
    degrees: Any,
    epochs_per_sample: Any,
    n_epochs: int,
) -> tuple[np.ndarray, ...]:
    embedding = np.asarray(init_embedding, dtype=np.float32, order="C")
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("Metal ibUMAP requires embedding shape (n_vertices, 2)")
    if len(embedding) < 2:
        raise ValueError("Metal ibUMAP requires at least two vertices")
    if not np.isfinite(embedding).all():
        raise FloatingPointError("Metal ibUMAP input embedding contains non-finite values")
    indptr = np.asarray(edgesrc, dtype=np.int32, order="C")
    indices = np.asarray(edgetgt, dtype=np.int32, order="C")
    edge_weights = np.asarray(weights, dtype=np.float32, order="C")
    degree_values = np.asarray(degrees, dtype=np.float32, order="C")
    epochs = np.asarray(epochs_per_sample, dtype=np.float32, order="C")
    if indptr.shape != (len(embedding) + 1,):
        raise ValueError("edgesrc must contain n_vertices + 1 CSR offsets")
    if not (len(indices) == len(edge_weights) == len(epochs) == int(indptr[-1])):
        raise ValueError("CSR edge arrays and epochs_per_sample must have equal length")
    if degree_values.shape != (len(embedding),):
        raise ValueError("degrees must contain one value per vertex")
    if int(n_epochs) <= 0:
        raise ValueError("n_epochs must be positive")
    return embedding, indptr, indices, edge_weights, degree_values, epochs


def _noise_scale_for_epoch(
    params: MetalOptimizerParameters, epoch: int, n_epochs: int
) -> float:
    if params.noise_mode == "none" or float(params.noise_scale) <= 0.0:
        return 0.0
    if params.noise_mode not in ("force", "embedding"):
        raise ValueError("noise_mode must be 'none', 'force', or 'embedding'")
    active_until = (
        int(n_epochs)
        if params.noise_until_epoch is None
        else int(params.noise_until_epoch)
    )
    if params.hybrid_mode == "early_noisy_late_deterministic":
        switch = (
            int(n_epochs) // 2
            if params.hybrid_switch_epoch is None
            else int(params.hybrid_switch_epoch)
        )
        active_until = min(active_until, switch)
    elif params.hybrid_mode != "none":
        raise ValueError("unsupported hybrid_mode")
    if active_until <= 0 or int(epoch) >= active_until:
        return 0.0
    if params.noise_decay == "constant":
        return float(params.noise_scale)
    progress = float(epoch) / float(max(active_until, 1))
    if params.noise_decay == "linear":
        return float(params.noise_scale) * max(0.0, 1.0 - progress)
    if params.noise_decay == "exponential":
        return float(params.noise_scale) * float(np.exp(-5.0 * progress))
    raise ValueError("unsupported noise_decay")


def _active_clip_norm(
    norm: float | None,
    with_alpha: bool,
    epoch_range: tuple[int, int] | None,
    *,
    alpha: float,
    epoch: int,
) -> float | None:
    if norm is None:
        return None
    if epoch_range is not None and not (
        int(epoch_range[0]) <= int(epoch) < int(epoch_range[1])
    ):
        return None
    return float(norm) * alpha if with_alpha else float(norm)


def run_metal_optimizer_core(
    init_embedding: Any,
    *,
    edgesrc: Any,
    edgetgt: Any,
    weights: Any,
    degrees: Any,
    epochs_per_sample: Any,
    n_epochs: int,
    params: MetalOptimizerParameters,
    checkpoint_epochs: Iterable[int] = (),
    workspace: Any = None,
    mx: Any = None,
) -> MetalOptimizerCoreResult:
    """Run the MLX optimizer and materialize only requested host boundaries."""

    optimizer_started = perf_counter()
    (
        embedding_np,
        edgesrc_np,
        edgetgt_np,
        weights_np,
        degrees_np,
        epochs_np,
    ) = _validate_core_inputs(
        init_embedding,
        edgesrc,
        edgetgt,
        weights,
        degrees,
        epochs_per_sample,
        n_epochs,
    )
    n_vertices = len(embedding_np)
    _, degree_ref, damping_np = _host_degree_damping(
        edgesrc_np, weights_np, n_vertices, params
    )
    neg_effects_np = (
        degrees_np
        * np.float32(0.5 * params.negative_sample_rate / n_vertices)
    )
    known_mask_np = np.zeros(n_vertices, dtype=np.bool_)
    if params.known_points_indices is not None:
        known_indices = np.asarray(params.known_points_indices, dtype=np.int64)
        if known_indices.ndim != 1 or np.any(known_indices < 0) or np.any(
            known_indices >= n_vertices
        ):
            raise ValueError("known_points_indices are out of bounds")
        known_mask_np[known_indices] = True
    noise_rng = np.random.default_rng(params.noise_seed)

    checkpoint_set = {int(value) for value in checkpoint_epochs}
    if any(value < 1 or value > int(n_epochs) for value in checkpoint_set):
        raise ValueError("checkpoint epochs must be in [1, n_epochs]")
    checkpoints: dict[int, np.ndarray] = {}
    stages = resolve_fft_schedule(
        n_epochs=int(n_epochs),
        n_interpolation_points=params.n_interpolation_points,
        combine_stages=params.combine_stages,
        interpolation_schedule=params.interpolation_schedule,
    )
    validate_fft_schedule_execution(
        stages, device="metal", p2m_mode=params.p2m_mode
    )
    host_prepare_time = perf_counter() - optimizer_started

    transfer_started = perf_counter()
    mx = mlx_core() if mx is None else mx
    embedding = mx.array(embedding_np, dtype=mx.float32)
    metal_edgesrc = mx.array(edgesrc_np, dtype=mx.int32)
    metal_edgetgt = mx.array(edgetgt_np, dtype=mx.int32)
    metal_epochs = mx.array(epochs_np, dtype=mx.float32)
    epoch_of_next = mx.array(epochs_np, dtype=mx.float32)
    # Percentile/reference resolution intentionally happens once on the host;
    # the resulting per-row scale remains an MLX array for every epoch.
    damping = mx.array(damping_np, dtype=mx.float32)
    neg_effects = mx.array(neg_effects_np, dtype=mx.float32)
    known_mask = mx.array(known_mask_np)
    evaluate(
        embedding,
        metal_edgesrc,
        metal_edgetgt,
        metal_epochs,
        epoch_of_next,
        damping,
        neg_effects,
        known_mask,
        mx=mx,
    )
    require_finite(embedding, label="Metal optimizer input embedding", mx=mx)
    transfer_time = perf_counter() - transfer_started

    epoch_loop_started = perf_counter()
    for epoch in range(int(n_epochs)):
        stage = stages[fft_stage_index_for_epoch(stages, epoch)]
        alpha = alpha_for_epoch(params.initial_alpha, epoch, int(n_epochs))
        clip_norm = _active_clip_norm(
            params.repulsion_clip_norm,
            params.repulsion_clip_with_alpha,
            params.repulsion_clip_epoch_range,
            alpha=alpha,
            epoch=epoch,
        )
        total_clip_norm = _active_clip_norm(
            params.total_update_clip_norm,
            params.total_update_clip_with_alpha,
            params.total_update_clip_epoch_range,
            alpha=alpha,
            epoch=epoch,
        )
        noise_scale = _noise_scale_for_epoch(params, epoch, int(n_epochs))
        attraction = sampling_attraction(
            embedding,
            metal_edgesrc,
            metal_edgetgt,
            metal_epochs,
            epoch_of_next,
            a=params.a,
            b=params.b,
            alpha=mx.array(alpha, dtype=mx.float32),
            epoch=epoch,
            mx=mx,
        )
        epoch_of_next = attraction.epoch_of_next_sample
        attraction_force = attraction.force * damping[:, None]
        if params.noise_mode == "force" and noise_scale > 0.0:
            noise_np = noise_rng.standard_normal(embedding_np.shape).astype(np.float32)
            noise_np *= np.float32(noise_scale)
            noise_np[known_mask_np] = 0.0
            attraction_force = attraction_force + mx.array(
                noise_np, dtype=mx.float32
            )
        ibfft = ibfft_repulsive_force(
            embedding,
            intervals_per_integer=params.intervals_per_integer,
            min_num_intervals=params.min_num_intervals,
            n_boxes_per_dim=params.n_boxes_per_dim,
            a=params.a,
            b=params.b,
            gamma=params.gamma,
            epsilon=params.epsilon,
            clip_value=params.kernel_clip,
            p2m_mode=params.p2m_mode,
            n_interpolation_points=stage.n_interpolation_points,
            workspace=workspace,
            mx=mx,
        )
        repulsion_force = scale_repulsive_force(
            ibfft.force,
            alpha=mx.array(alpha, dtype=mx.float32),
            neg_effects=neg_effects,
            clip_norm=clip_norm,
            mx=mx,
        )
        if params.fused_update:
            update_kernel = (
                build_fused_update_kernel(mx=mx)
                if workspace is None
                else workspace.get_kernel(
                    "fused_update", lambda: build_fused_update_kernel(mx=mx)
                )
            )
            embedding = apply_optimizer_update_fused(
                embedding,
                attraction_force,
                repulsion_force,
                total_clip_norm=total_clip_norm,
                known_mask=known_mask,
                mx=mx,
                kernel=update_kernel,
            )
        else:
            embedding = apply_optimizer_update(
                embedding,
                attraction_force,
                repulsion_force,
                total_clip_norm=total_clip_norm,
                known_mask=known_mask,
                mx=mx,
            )
        if params.noise_mode == "embedding" and noise_scale > 0.0:
            noise_np = noise_rng.standard_normal(embedding_np.shape).astype(np.float32)
            noise_np *= np.float32(noise_scale)
            noise_np[known_mask_np] = 0.0
            embedding = embedding + mx.array(noise_np, dtype=mx.float32)
        completed_epoch = epoch + 1
        if completed_epoch in checkpoint_set:
            checkpoints[completed_epoch] = to_numpy(embedding, mx=mx)
    require_finite(embedding, label="Metal optimizer output embedding", mx=mx)
    epoch_loop_time = perf_counter() - epoch_loop_started

    materialize_started = perf_counter()
    result_embedding = to_numpy(embedding, mx=mx)
    result_next = to_numpy(epoch_of_next, mx=mx)
    materialize_time = perf_counter() - materialize_started
    total_time = perf_counter() - optimizer_started
    return MetalOptimizerCoreResult(
        embedding=result_embedding,
        epoch_of_next_sample=result_next,
        checkpoints=checkpoints,
        timings={
            "metal_optimizer_time_s": float(total_time),
            "metal_host_prepare_time_s": float(host_prepare_time),
            "metal_transfer_time_s": float(transfer_time),
            "metal_epoch_loop_time_s": float(epoch_loop_time),
            "metal_materialize_time_s": float(materialize_time),
            "metal_optimizer_epochs": float(n_epochs),
        },
        degree_ref=float(degree_ref),
    )


def optimize_ibumap_metal(
    init_embedding: Any,
    *,
    edgesrc: Any,
    edgetgt: Any,
    weights: Any,
    degrees: Any,
    epochs_per_sample: Any,
    n_epochs: int,
    model: Any,
    workspace: Any = None,
) -> tuple[Any, dict[str, float]]:
    """Run the validated p=1/p2/p3 optimizer from the hybrid Metal pipeline."""

    cfg = model.runtime
    params = MetalOptimizerParameters(
        a=float(model._a),
        b=float(model._b),
        gamma=float(model.repulsion_strength),
        epsilon=float(cfg.numerics.epsilon),
        kernel_clip=float(model.ibfft_kernel_clip),
        intervals_per_integer=float(cfg.fft.intervals_per_integer),
        min_num_intervals=int(cfg.fft.min_num_intervals),
        n_boxes_per_dim=float(cfg.fft.n_boxes_per_dim),
        p2m_mode=(
            "segmented"
            if cfg.fft.p2m_mode == "auto" and model.runtime.deterministic
            else "atomic"
            if cfg.fft.p2m_mode == "auto"
            else str(cfg.fft.p2m_mode)
        ),
        n_interpolation_points=int(cfg.fft.n_interpolation_points),
        combine_stages=bool(cfg.fft.combine_stages),
        interpolation_schedule=cfg.fft.interpolation_schedule,
        initial_alpha=float(model.learning_rate),
        negative_sample_rate=float(model.negative_sample_rate),
        repulsion_clip_norm=model.repulsion_clip_norm,
        repulsion_clip_with_alpha=bool(model.repulsion_clip_with_alpha),
        repulsion_clip_epoch_range=(
            None
            if model.repulsion_clip_epoch_range is None
            else tuple(int(value) for value in model.repulsion_clip_epoch_range)
        ),
        total_update_clip_norm=model.total_update_clip_norm,
        total_update_clip_with_alpha=bool(model.total_update_clip_with_alpha),
        total_update_clip_epoch_range=(
            None
            if model.total_update_clip_epoch_range is None
            else tuple(int(value) for value in model.total_update_clip_epoch_range)
        ),
        known_points_indices=cfg.constraint.known_points_indices,
        noise_mode=str(cfg.noise.mode),
        noise_scale=float(cfg.noise.scale),
        noise_decay=str(cfg.noise.decay),
        noise_until_epoch=cfg.noise.until_epoch,
        noise_seed=cfg.noise.seed,
        hybrid_mode=str(cfg.noise.hybrid_mode),
        hybrid_switch_epoch=cfg.noise.hybrid_switch_epoch,
        degree_damping_enabled=bool(model.attraction_degree_damping),
        degree_damping_mode=str(model.attraction_degree_damping_mode),
        degree_damping_ref=str(model.attraction_degree_damping_ref),
        degree_damping_ref_value=model.attraction_degree_damping_ref_value,
        degree_damping_power=float(model.attraction_degree_damping_power),
        degree_damping_min_scale=model.attraction_degree_damping_min_scale,
    )
    workspace_before = workspace.snapshot() if workspace is not None else None
    result = run_metal_optimizer_core(
        init_embedding,
        edgesrc=edgesrc,
        edgetgt=edgetgt,
        weights=weights,
        degrees=degrees,
        epochs_per_sample=epochs_per_sample,
        n_epochs=n_epochs,
        params=params,
        workspace=workspace,
    )
    if workspace is not None:
        workspace_after = workspace.snapshot()
        result.timings.update(
            {
                f"metal_workspace_{key}": float(value)
                for key, value in workspace_after.items()
            }
        )
        result.timings.update(
            {
                f"metal_workspace_delta_{key}": float(
                    value - int(workspace_before.get(key, 0))
                )
                for key, value in workspace_after.items()
            }
        )
    return result.embedding, result.timings


__all__ = [
    "MetalOptimizerCoreResult",
    "MetalOptimizerNotImplementedError",
    "MetalOptimizerParameters",
    "optimize_ibumap_metal",
    "run_metal_optimizer_core",
]
