"""MLX/Metal ibFFT operators.

The functions in this module intentionally expose the intermediate arrays used
by ibFFT.  Stage-3 experiments compare those arrays with the CPU reference one
operator at a time before the optimizer composes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fft_grid import quantize_fft_box_width
from .runtime import mlx_core, to_numpy


ALLOWED_N_BOXES_PER_DIM = np.asarray(
    [
        50, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243,
        256, 288, 324, 384, 432, 576, 648, 768, 864, 972, 1152, 1296,
        1536, 1728, 1944, 2304, 2592, 3072, 3456, 3888, 4608, 5184,
        6144, 6912,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True, slots=True)
class MetalGridSpec:
    """Host-side shape metadata resolved from two synchronized bounds."""

    min_coord: float
    max_coord: float
    span: float
    n_boxes_per_dim: int
    raw_box_width: float
    box_width: float
    quantization_level: int
    n_interpolation_points: int
    interpolation_dim: int
    fft_dim: int


@dataclass(frozen=True, slots=True)
class MetalFFTResult:
    """Inspectable outputs of the batched convolution stage."""

    kernel_spectrum: Any
    charge_spectrum: Any
    frequency_product: Any
    potential: Any


@dataclass(frozen=True, slots=True)
class MetalIbFFTResult:
    """Inspectable result of one complete ibFFT repulsion evaluation."""

    force: Any
    bounds: Any
    grid: MetalGridSpec
    box_idx: Any
    circulant_kernel: Any
    mat_w: Any
    fft: MetalFFTResult


_BOX_INDEX_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    float min_coord = params[0];
    float box_width = params[1];
    int n_boxes = dims[0];
    int gx = int(metal::floor((embedding[2 * i] - min_coord) / box_width));
    int gy = int(metal::floor((embedding[2 * i + 1] - min_coord) / box_width));
    box_idx[2 * i] = metal::max(0, metal::min(n_boxes - 1, gx));
    box_idx[2 * i + 1] = metal::max(0, metal::min(n_boxes - 1, gy));
"""


_UMAP_KERNEL_SOURCE = r"""
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    int size = dims[0];
    int fft_dim = dims[1];
    if (x >= size || y >= size) return;
    float box_width = params[0];
    float a = params[1];
    float b = params[2];
    float gamma = params[3];
    float epsilon = params[4];
    float clip_value = params[5];
    float spacing = box_width / float(dims[2]);
    float dist2 = float(x * x + y * y) * spacing * spacing;
    float value = (2.0f * gamma * b) /
        ((epsilon + dist2) * (a * metal::pow(dist2, b) + 1.0f));
    value = metal::max(-clip_value, metal::min(clip_value, value));
    circulant[(size + x) * fft_dim + (size + y)] = value;
    circulant[(size - x) * fft_dim + (size + y)] = value;
    circulant[(size + x) * fft_dim + (size - y)] = value;
    circulant[(size - x) * fft_dim + (size - y)] = value;
"""


_P2M_P1_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    int gx = box_idx[2 * i];
    int gy = box_idx[2 * i + 1];
    int fft_dim = dims[0];
    uint plane = uint(fft_dim * fft_dim);
    uint offset = uint(gx * fft_dim + gy);
    atomic_fetch_add_explicit(
        &mat_w[offset], embedding[2 * i], memory_order_relaxed);
    atomic_fetch_add_explicit(
        &mat_w[plane + offset], embedding[2 * i + 1], memory_order_relaxed);
    atomic_fetch_add_explicit(
        &mat_w[2 * plane + offset], 1.0f, memory_order_relaxed);
"""


_M2P_P1_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    int size = dims[0];
    int fft_dim = dims[1];
    int gx = box_idx[2 * i] + size;
    int gy = box_idx[2 * i + 1] + size;
    uint plane = uint(fft_dim * fft_dim);
    uint offset = uint(gx * fft_dim + gy);
    float common = potential[2 * plane + offset];
    force[2 * i] = common * embedding[2 * i] - potential[offset];
    force[2 * i + 1] =
        common * embedding[2 * i + 1] - potential[plane + offset];
"""


_P2M_P1_SEGMENTED_SOURCE = r"""
    uint segment = thread_position_in_grid.x;
    if (segment >= segment_cells_shape[0]) return;
    int cell = segment_cells[segment];
    int n_boxes = dims[0];
    int fft_dim = dims[1];
    int gx = cell / n_boxes;
    int gy = cell - gx * n_boxes;
    float x_total = 0.0f;
    float y_total = 0.0f;
    float count = 0.0f;
    for (int pos = segment_starts[segment];
         pos < segment_ends[segment]; ++pos) {
        int point = ordered_points[pos];
        x_total += embedding[2 * point];
        y_total += embedding[2 * point + 1];
        count += 1.0f;
    }
    uint plane = uint(fft_dim * fft_dim);
    uint offset = uint(gx * fft_dim + gy);
    mat_w[offset] = x_total;
    mat_w[plane + offset] = y_total;
    mat_w[2 * plane + offset] = count;
"""


_P2M_INTERPOLATED_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    int gx = box_idx[2 * i];
    int gy = box_idx[2 * i + 1];
    float box_width = params[0];
    float min_coord = params[1];
    int n_boxes = dims[0];
    int fft_dim = dims[1];
    int p = dims[2];
    float h = box_width / float(p);
    float local_x = embedding[2 * i] - min_coord - float(gx) * box_width;
    float local_y = embedding[2 * i + 1] - min_coord - float(gy) * box_width;
    uint plane = uint(fft_dim * fft_dim);
    for (int j = 0; j < p; ++j) {
        float node_j = (float(j) + 0.5f) * h;
        float wx = 1.0f;
        float wy = 1.0f;
        for (int k = 0; k < p; ++k) {
            if (k == j) continue;
            float node_k = (float(k) + 0.5f) * h;
            wx *= (local_x - node_k) / (node_j - node_k);
            wy *= (local_y - node_k) / (node_j - node_k);
        }
        for (int k = 0; k < p; ++k) {
            float node_k = (float(k) + 0.5f) * h;
            float wy_k = 1.0f;
            for (int q = 0; q < p; ++q) {
                if (q == k) continue;
                float node_q = (float(q) + 0.5f) * h;
                wy_k *= (local_y - node_q) / (node_k - node_q);
            }
            float weight = wx * wy_k;
            int ix = gx * p + j;
            int iy = gy * p + k;
            uint offset = uint(ix * fft_dim + iy);
            atomic_fetch_add_explicit(
                &mat_w[offset], weight * embedding[2 * i], memory_order_relaxed);
            atomic_fetch_add_explicit(
                &mat_w[plane + offset],
                weight * embedding[2 * i + 1], memory_order_relaxed);
            atomic_fetch_add_explicit(
                &mat_w[2 * plane + offset], weight, memory_order_relaxed);
        }
    }
"""


_M2P_INTERPOLATED_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= embedding_shape[0]) return;
    int gx = box_idx[2 * i];
    int gy = box_idx[2 * i + 1];
    float box_width = params[0];
    float min_coord = params[1];
    int interp_dim = dims[0];
    int fft_dim = dims[1];
    int p = dims[2];
    float h = box_width / float(p);
    float local_x = embedding[2 * i] - min_coord - float(gx) * box_width;
    float local_y = embedding[2 * i + 1] - min_coord - float(gy) * box_width;
    uint plane = uint(fft_dim * fft_dim);
    float potential_x = 0.0f;
    float potential_y = 0.0f;
    float potential_common = 0.0f;
    for (int j = 0; j < p; ++j) {
        float node_j = (float(j) + 0.5f) * h;
        float wx = 1.0f;
        for (int q = 0; q < p; ++q) {
            if (q == j) continue;
            float node_q = (float(q) + 0.5f) * h;
            wx *= (local_x - node_q) / (node_j - node_q);
        }
        for (int k = 0; k < p; ++k) {
            float node_k = (float(k) + 0.5f) * h;
            float wy = 1.0f;
            for (int q = 0; q < p; ++q) {
                if (q == k) continue;
                float node_q = (float(q) + 0.5f) * h;
                wy *= (local_y - node_q) / (node_k - node_q);
            }
            float weight = wx * wy;
            int ix = gx * p + j + interp_dim;
            int iy = gy * p + k + interp_dim;
            uint offset = uint(ix * fft_dim + iy);
            potential_x += weight * potential[offset];
            potential_y += weight * potential[plane + offset];
            potential_common += weight * potential[2 * plane + offset];
        }
    }
    force[2 * i] = potential_common * embedding[2 * i] - potential_x;
    force[2 * i + 1] =
        potential_common * embedding[2 * i + 1] - potential_y;
"""


def embedding_bounds(embedding: Any, *, mx: Any = None) -> Any:
    """Return ``[min, max, nonfinite_count]`` on the Metal device."""

    mx = mlx_core() if mx is None else mx
    return mx.stack(
        (
            mx.min(embedding),
            mx.max(embedding),
            mx.sum(~mx.isfinite(embedding)).astype(mx.float32),
        )
    )


def resolve_grid_spec(
    bounds: Any,
    n_points: int,
    *,
    n_interpolation_points: int = 1,
    intervals_per_integer: float = 1.0,
    min_num_intervals: int = 50,
    n_boxes_per_dim: float = 1.0,
    mx: Any = None,
) -> MetalGridSpec:
    """Resolve the same grid policy used by the CPU ibFFT implementation."""

    if int(n_points) < 2:
        raise ValueError("ibFFT requires at least two points")
    if int(n_interpolation_points) < 1:
        raise ValueError("n_interpolation_points must be positive")
    if not np.isfinite(intervals_per_integer) or intervals_per_integer <= 0:
        raise ValueError("intervals_per_integer must be positive and finite")
    bounds_np = np.asarray(
        to_numpy(bounds, mx=mx, copy=False) if mx is not None else bounds,
        dtype=np.float32,
    )
    if bounds_np.shape != (3,):
        raise ValueError("bounds must have shape (3,)")
    if int(bounds_np[2]) != 0:
        raise FloatingPointError(
            f"ibFFT input embedding contains {int(bounds_np[2])} non-finite value(s)"
        )
    min_coord = float(bounds_np[0])
    max_coord = float(bounds_np[1])
    span = max_coord - min_coord
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("ibFFT embedding span must be positive and finite")

    n_points_float = float(n_points)
    boxes_1 = np.sqrt(16.0 * n_points_float)
    boxes_2 = np.sqrt(4.0 * n_points_float / np.log(n_points_float))
    boxes_3 = span / float(intervals_per_integer)
    estimate = int(
        min(
            boxes_1,
            max(boxes_2, max(float(min_num_intervals), boxes_3)),
        )
    )
    estimate = int(float(n_boxes_per_dim) * estimate)
    if estimate >= int(ALLOWED_N_BOXES_PER_DIM[-1]):
        n_boxes = int(ALLOWED_N_BOXES_PER_DIM[-1])
    else:
        n_boxes = int(ALLOWED_N_BOXES_PER_DIM[ALLOWED_N_BOXES_PER_DIM > estimate][0])
    raw_width, box_width, level = quantize_fft_box_width(
        span, n_boxes, np.float32
    )
    interpolation_dim = int(n_interpolation_points) * n_boxes
    return MetalGridSpec(
        min_coord=min_coord,
        max_coord=max_coord,
        span=span,
        n_boxes_per_dim=n_boxes,
        raw_box_width=float(raw_width),
        box_width=float(box_width),
        quantization_level=int(level),
        n_interpolation_points=int(n_interpolation_points),
        interpolation_dim=interpolation_dim,
        fft_dim=2 * interpolation_dim,
    )


def build_box_index_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_box_index",
        input_names=["embedding", "params", "dims"],
        output_names=["box_idx"],
        source=_BOX_INDEX_SOURCE,
    )


def box_indices(
    embedding: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    mx = mlx_core() if mx is None else mx
    kernel = build_box_index_kernel(mx=mx) if kernel is None else kernel
    params = mx.array(
        np.asarray([grid.min_coord, grid.box_width], dtype=np.float32)
    )
    dims = mx.array(np.asarray([grid.n_boxes_per_dim], dtype=np.int32))
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[embedding, params, dims],
        output_shapes=[(n_points, 2)],
        output_dtypes=[mx.int32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
    )[0]


def build_umap_kernel_builder(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_umap_circulant",
        input_names=["params", "dims"],
        output_names=["circulant"],
        source=_UMAP_KERNEL_SOURCE,
    )


def umap_circulant_kernel(
    grid: MetalGridSpec,
    *,
    a: float,
    b: float,
    gamma: float,
    epsilon: float,
    clip_value: float,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Build the real padded circulant kernel on Metal."""

    mx = mlx_core() if mx is None else mx
    kernel = build_umap_kernel_builder(mx=mx) if kernel is None else kernel
    params = mx.array(
        np.asarray(
            [grid.box_width, a, b, gamma, epsilon, clip_value],
            dtype=np.float32,
        )
    )
    dims = mx.array(
        np.asarray(
            [
                grid.interpolation_dim,
                grid.fft_dim,
                grid.n_interpolation_points,
            ],
            dtype=np.int32,
        )
    )
    size = grid.interpolation_dim
    return kernel(
        inputs=[params, dims],
        output_shapes=[(grid.fft_dim, grid.fft_dim)],
        output_dtypes=[mx.float32],
        grid=(size, size, 1),
        threadgroup=(min(16, size), min(16, size), 1),
        init_value=0,
    )[0]


def build_p2m_p1_atomic_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_p2m_p1_atomic",
        input_names=["embedding", "box_idx", "dims"],
        output_names=["mat_w"],
        source=_P2M_P1_SOURCE,
        atomic_outputs=True,
    )


def p2m_p1_atomic(
    embedding: Any,
    box_idx: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Scatter ``[x, y, 1]`` point charges into the padded FFT grid."""

    if grid.n_interpolation_points != 1:
        raise ValueError("p2m_p1_atomic requires n_interpolation_points=1")
    mx = mlx_core() if mx is None else mx
    kernel = build_p2m_p1_atomic_kernel(mx=mx) if kernel is None else kernel
    dims = mx.array(np.asarray([grid.fft_dim], dtype=np.int32))
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[embedding, box_idx, dims],
        output_shapes=[(3, grid.fft_dim, grid.fft_dim)],
        output_dtypes=[mx.float32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
        init_value=0,
    )[0]


def build_p2m_p1_segmented_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_p2m_p1_segmented",
        input_names=[
            "embedding",
            "ordered_points",
            "segment_cells",
            "segment_starts",
            "segment_ends",
            "dims",
        ],
        output_names=["mat_w"],
        source=_P2M_P1_SEGMENTED_SOURCE,
    )


def p2m_p1_segmented(
    embedding: Any,
    box_idx: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Deterministically reduce each occupied cell in stable point order."""

    if grid.n_interpolation_points != 1:
        raise ValueError("p2m_p1_segmented requires n_interpolation_points=1")
    mx = mlx_core() if mx is None else mx
    indices = np.asarray(to_numpy(box_idx, mx=mx), dtype=np.int32)
    cell_ids = (
        indices[:, 0].astype(np.int64) * grid.n_boxes_per_dim
        + indices[:, 1].astype(np.int64)
    )
    order = np.argsort(cell_ids, kind="stable").astype(np.int32)
    sorted_cells = cell_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
    ).astype(np.int32)
    ends = np.r_[starts[1:], len(order)].astype(np.int32)
    segment_cells = sorted_cells[starts].astype(np.int32)
    kernel = build_p2m_p1_segmented_kernel(mx=mx) if kernel is None else kernel
    dims = mx.array(
        np.asarray([grid.n_boxes_per_dim, grid.fft_dim], dtype=np.int32)
    )
    n_segments = len(segment_cells)
    return kernel(
        inputs=[
            embedding,
            mx.array(order, dtype=mx.int32),
            mx.array(segment_cells, dtype=mx.int32),
            mx.array(starts, dtype=mx.int32),
            mx.array(ends, dtype=mx.int32),
            dims,
        ],
        output_shapes=[(3, grid.fft_dim, grid.fft_dim)],
        output_dtypes=[mx.float32],
        grid=(n_segments, 1, 1),
        threadgroup=(min(256, max(1, n_segments)), 1, 1),
        init_value=0,
    )[0]


def build_p2m_interpolated_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_p2m_interpolated_atomic",
        input_names=["embedding", "box_idx", "params", "dims"],
        output_names=["mat_w"],
        source=_P2M_INTERPOLATED_SOURCE,
        atomic_outputs=True,
    )


def p2m_interpolated_atomic(
    embedding: Any,
    box_idx: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Atomic P2M for fixed interpolation orders greater than one."""

    if grid.n_interpolation_points <= 1:
        raise ValueError("p2m_interpolated_atomic requires p>1")
    mx = mlx_core() if mx is None else mx
    kernel = build_p2m_interpolated_kernel(mx=mx) if kernel is None else kernel
    params = mx.array(
        np.asarray([grid.box_width, grid.min_coord], dtype=np.float32)
    )
    dims = mx.array(
        np.asarray(
            [grid.n_boxes_per_dim, grid.fft_dim, grid.n_interpolation_points],
            dtype=np.int32,
        )
    )
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[embedding, box_idx, params, dims],
        output_shapes=[(3, grid.fft_dim, grid.fft_dim)],
        output_dtypes=[mx.float32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
        init_value=0,
    )[0]


def fft_convolve(
    mat_w: Any,
    circulant_kernel: Any,
    grid: MetalGridSpec,
    *,
    kernel_spectrum: Any = None,
    mx: Any = None,
) -> MetalFFTResult:
    """Convolve all charge channels with the real circulant kernel."""

    mx = mlx_core() if mx is None else mx
    expected_grid_shape = (3, grid.fft_dim, grid.fft_dim)
    if tuple(mat_w.shape) != expected_grid_shape:
        raise ValueError(
            f"mat_w must have shape {expected_grid_shape}, got {tuple(mat_w.shape)}"
        )
    if kernel_spectrum is None:
        expected_kernel_shape = (grid.fft_dim, grid.fft_dim)
        if circulant_kernel is None or tuple(circulant_kernel.shape) != expected_kernel_shape:
            actual_shape = None if circulant_kernel is None else tuple(circulant_kernel.shape)
            raise ValueError(
                "circulant_kernel must have shape "
                f"{expected_kernel_shape}, got {actual_shape}"
            )
        kernel_spectrum = mx.fft.rfft2(circulant_kernel)
    charge_spectrum = mx.fft.rfft2(mat_w, axes=(-2, -1))
    frequency_product = charge_spectrum * kernel_spectrum
    potential = mx.fft.irfft2(
        frequency_product,
        s=(grid.fft_dim, grid.fft_dim),
        axes=(-2, -1),
    )
    return MetalFFTResult(
        kernel_spectrum=kernel_spectrum,
        charge_spectrum=charge_spectrum,
        frequency_product=frequency_product,
        potential=potential,
    )


def build_m2p_p1_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_m2p_p1",
        input_names=["embedding", "box_idx", "potential", "dims"],
        output_names=["force"],
        source=_M2P_P1_SOURCE,
    )


def m2p_p1(
    embedding: Any,
    box_idx: Any,
    potential: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Interpolate p=1 potentials and form the raw ibFFT force."""

    if grid.n_interpolation_points != 1:
        raise ValueError("m2p_p1 requires n_interpolation_points=1")
    mx = mlx_core() if mx is None else mx
    kernel = build_m2p_p1_kernel(mx=mx) if kernel is None else kernel
    dims = mx.array(
        np.asarray([grid.interpolation_dim, grid.fft_dim], dtype=np.int32)
    )
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[embedding, box_idx, potential, dims],
        output_shapes=[tuple(embedding.shape)],
        output_dtypes=[mx.float32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
    )[0]


def build_m2p_interpolated_kernel(*, mx: Any = None) -> Any:
    mx = mlx_core() if mx is None else mx
    return mx.fast.metal_kernel(
        name="ibumap_metal_m2p_interpolated",
        input_names=["embedding", "box_idx", "potential", "params", "dims"],
        output_names=["force"],
        source=_M2P_INTERPOLATED_SOURCE,
    )


def m2p_interpolated(
    embedding: Any,
    box_idx: Any,
    potential: Any,
    grid: MetalGridSpec,
    *,
    mx: Any = None,
    kernel: Any = None,
) -> Any:
    """Higher-order interpolation of mesh potentials back to points."""

    if grid.n_interpolation_points <= 1:
        raise ValueError("m2p_interpolated requires p>1")
    mx = mlx_core() if mx is None else mx
    kernel = build_m2p_interpolated_kernel(mx=mx) if kernel is None else kernel
    params = mx.array(
        np.asarray([grid.box_width, grid.min_coord], dtype=np.float32)
    )
    dims = mx.array(
        np.asarray(
            [grid.interpolation_dim, grid.fft_dim, grid.n_interpolation_points],
            dtype=np.int32,
        )
    )
    n_points = int(embedding.shape[0])
    return kernel(
        inputs=[embedding, box_idx, potential, params, dims],
        output_shapes=[tuple(embedding.shape)],
        output_dtypes=[mx.float32],
        grid=(n_points, 1, 1),
        threadgroup=(min(256, n_points), 1, 1),
    )[0]


def scale_repulsive_force(
    raw_force: Any,
    *,
    alpha: Any,
    neg_effects: Any,
    clip_norm: float | None = None,
    mx: Any = None,
) -> Any:
    """Apply true-loss scaling in the CPU optimizer's operation order."""

    mx = mlx_core() if mx is None else mx
    force = raw_force * alpha
    force = force * neg_effects[:, None]
    if clip_norm is not None:
        norms = mx.sqrt(mx.sum(force * force, axis=1))
        scale = mx.minimum(
            mx.ones_like(norms),
            mx.array(float(clip_norm), dtype=mx.float32) / (norms + 1e-12),
        )
        force = force * scale[:, None]
    return force


def ibfft_repulsive_force(
    embedding: Any,
    *,
    intervals_per_integer: float,
    min_num_intervals: int,
    n_boxes_per_dim: float,
    a: float,
    b: float,
    gamma: float,
    epsilon: float,
    clip_value: float,
    p2m_mode: str = "atomic",
    n_interpolation_points: int = 1,
    workspace: Any = None,
    mx: Any = None,
) -> MetalIbFFTResult:
    """Run the validated p=1 atomic ibFFT operator chain."""

    mx = mlx_core() if mx is None else mx
    bounds = embedding_bounds(embedding, mx=mx)
    grid = resolve_grid_spec(
        bounds,
        int(embedding.shape[0]),
        n_interpolation_points=n_interpolation_points,
        intervals_per_integer=intervals_per_integer,
        min_num_intervals=min_num_intervals,
        n_boxes_per_dim=n_boxes_per_dim,
        mx=mx,
    )
    def cached_kernel(key: str, builder: Any) -> Any:
        if workspace is None:
            return builder()
        return workspace.get_kernel(key, builder)

    box_kernel = cached_kernel("box_index", lambda: build_box_index_kernel(mx=mx))
    box_idx = box_indices(embedding, grid, mx=mx, kernel=box_kernel)
    spectrum_key = (
        "umap",
        int(grid.quantization_level),
        int(grid.n_boxes_per_dim),
        int(grid.n_interpolation_points),
        float(a),
        float(b),
        float(gamma),
        float(epsilon),
        float(clip_value),
    )
    cached_spectrum = (
        None if workspace is None else workspace.get_fft_kernel(spectrum_key)
    )
    circulant = None
    if cached_spectrum is None:
        kernel_builder = cached_kernel(
            "umap_circulant", lambda: build_umap_kernel_builder(mx=mx)
        )
        circulant = umap_circulant_kernel(
            grid,
            a=a,
            b=b,
            gamma=gamma,
            epsilon=epsilon,
            clip_value=clip_value,
            mx=mx,
            kernel=kernel_builder,
        )
    if grid.n_interpolation_points == 1 and p2m_mode == "atomic":
        p2m_kernel = cached_kernel(
            "p2m_p1_atomic", lambda: build_p2m_p1_atomic_kernel(mx=mx)
        )
        mat_w = p2m_p1_atomic(
            embedding, box_idx, grid, mx=mx, kernel=p2m_kernel
        )
    elif grid.n_interpolation_points == 1 and p2m_mode == "segmented":
        p2m_kernel = cached_kernel(
            "p2m_p1_segmented", lambda: build_p2m_p1_segmented_kernel(mx=mx)
        )
        mat_w = p2m_p1_segmented(
            embedding, box_idx, grid, mx=mx, kernel=p2m_kernel
        )
    elif grid.n_interpolation_points > 1 and p2m_mode == "atomic":
        p2m_kernel = cached_kernel(
            "p2m_interpolated", lambda: build_p2m_interpolated_kernel(mx=mx)
        )
        mat_w = p2m_interpolated_atomic(
            embedding, box_idx, grid, mx=mx, kernel=p2m_kernel
        )
    else:
        raise ValueError("segmented P2M is currently supported for p=1 only")
    fft = fft_convolve(
        mat_w,
        circulant,
        grid,
        kernel_spectrum=cached_spectrum,
        mx=mx,
    )
    if workspace is not None and cached_spectrum is None:
        workspace.store_fft_kernel(spectrum_key, fft.kernel_spectrum)
    if grid.n_interpolation_points == 1:
        m2p_kernel = cached_kernel("m2p_p1", lambda: build_m2p_p1_kernel(mx=mx))
        force = m2p_p1(
            embedding, box_idx, fft.potential, grid, mx=mx, kernel=m2p_kernel
        )
    else:
        m2p_kernel = cached_kernel(
            "m2p_interpolated", lambda: build_m2p_interpolated_kernel(mx=mx)
        )
        force = m2p_interpolated(
            embedding, box_idx, fft.potential, grid, mx=mx, kernel=m2p_kernel
        )
    return MetalIbFFTResult(
        force=force,
        bounds=bounds,
        grid=grid,
        box_idx=box_idx,
        circulant_kernel=circulant,
        mat_w=mat_w,
        fft=fft,
    )


__all__ = [
    "ALLOWED_N_BOXES_PER_DIM",
    "MetalGridSpec",
    "MetalFFTResult",
    "MetalIbFFTResult",
    "box_indices",
    "build_box_index_kernel",
    "build_m2p_p1_kernel",
    "build_m2p_interpolated_kernel",
    "build_p2m_p1_atomic_kernel",
    "build_p2m_p1_segmented_kernel",
    "build_p2m_interpolated_kernel",
    "build_umap_kernel_builder",
    "embedding_bounds",
    "fft_convolve",
    "ibfft_repulsive_force",
    "m2p_p1",
    "m2p_interpolated",
    "resolve_grid_spec",
    "p2m_p1_atomic",
    "p2m_p1_segmented",
    "p2m_interpolated_atomic",
    "scale_repulsive_force",
    "umap_circulant_kernel",
]
