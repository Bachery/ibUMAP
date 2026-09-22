"""MLX Metal kernels for ibUMAP.

The package is intentionally empty after Stage 2. Stage 3 adds validated
operators here without making MLX a mandatory import for CPU/CUDA users.
"""

from .runtime import (
    MetalTiming,
    as_metal_array,
    evaluate,
    finite_count,
    memory_snapshot,
    mlx_core,
    timed_call,
    to_numpy,
)
from .ibfft import (
    ALLOWED_N_BOXES_PER_DIM,
    MetalFFTResult,
    MetalIbFFTResult,
    MetalGridSpec,
    box_indices,
    embedding_bounds,
    fft_convolve,
    ibfft_repulsive_force,
    m2p_p1,
    m2p_interpolated,
    p2m_p1_atomic,
    p2m_p1_segmented,
    p2m_interpolated_atomic,
    resolve_grid_spec,
    scale_repulsive_force,
    umap_circulant_kernel,
)
from .attraction import (
    MetalAttractionResult,
    degree_damping_scale,
    sampling_attraction,
    weighted_degrees,
)
from .update import (
    alpha_for_epoch,
    apply_optimizer_update,
    apply_optimizer_update_fused,
    require_finite,
)

__all__ = [
    "MetalTiming",
    "as_metal_array",
    "evaluate",
    "finite_count",
    "memory_snapshot",
    "mlx_core",
    "timed_call",
    "to_numpy",
    "ALLOWED_N_BOXES_PER_DIM",
    "MetalFFTResult",
    "MetalIbFFTResult",
    "MetalGridSpec",
    "box_indices",
    "embedding_bounds",
    "fft_convolve",
    "ibfft_repulsive_force",
    "m2p_p1",
    "m2p_interpolated",
    "p2m_p1_atomic",
    "p2m_p1_segmented",
    "p2m_interpolated_atomic",
    "resolve_grid_spec",
    "scale_repulsive_force",
    "umap_circulant_kernel",
    "MetalAttractionResult",
    "degree_damping_scale",
    "sampling_attraction",
    "weighted_degrees",
    "alpha_for_epoch",
    "apply_optimizer_update",
    "apply_optimizer_update_fused",
    "require_finite",
]
