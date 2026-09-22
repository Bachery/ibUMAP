"""Path-aware UMAPFFT parameter usage registry and reporting helpers."""

from __future__ import annotations

from dataclasses import MISSING, asdict, fields, is_dataclass
from inspect import Parameter, signature
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from .config import (
    AttractionScheduleConfig,
    ConstraintConfig,
    FFTConfig,
    HybridOptimizerConfig,
    InitializationConfig,
    NoiseConfig,
    NumericsConfig,
    TFDPConfig,
    UMAPFFTConfigBundle,
)


PATH_LABELS = (
    "ibumap_cpu",
    "ibumap_cuda",
    "ibumap_metal",
    "hybrid_cpu",
    "cpu_umap_async",
    "cpu_umap_sync",
    "umap_learn_baseline",
    "cuml_umap_baseline",
    "tfdp_cpu",
    "tfdp_cuda",
    "fixed_knn",
    "optimization_only",
    "update_embedding",
)
REGISTRY_VERSION = "2026-07-21"

IBUMAP = {"ibumap_cpu", "ibumap_cuda", "ibumap_metal"}
HYBRID = {"hybrid_cpu"}
CPU_UMAP = {"cpu_umap_async", "cpu_umap_sync"}
TFDP = {"tfdp_cpu", "tfdp_cuda"}
CUMl = {"cuml_umap_baseline"}
IBUMAP_FAMILY = IBUMAP | HYBRID
INTERNAL = IBUMAP_FAMILY | CPU_UMAP | TFDP
FULL_PIPELINE = INTERNAL | CUMl


def _names(text: str) -> set[str]:
    return set(text.split())


# Entries override the conservative default of "unused". Usage describes
# actual consumption observed in api.py, pipeline/* and optimizers/*.
GROUPS: list[tuple[set[str], str, set[str], str, str]] = [
    (_names("runtime graph umap initialization fft ibumap cpu_umap hybrid tfdp constraints diagnostics"), "canonical_config", set(PATH_LABELS), "supported", "Canonical grouped constructor configuration."),
    (_names("algorithm device"), "public_core", FULL_PIPELINE, "supported", "Selects the canonical implementation path."),
    (_names("attraction_mode repulsion_mode"), "ibumap_core", IBUMAP_FAMILY | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "ibUMAP force formulation selector."),
    (_names("graph_backend"), "graph", FULL_PIPELINE, "supported", "Resolved to cpu on CPU/Metal and cuml on CUDA full-pipeline runs; bypassed by fixed-input entrypoints."),
    (_names("workspace_policy workspace_limit_bytes"), "public_core", IBUMAP_FAMILY | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Controls reusable ibUMAP workspace retention and the budget used by automatic P2M mode selection."),
    (_names("p2m_mode"), "ibumap_core", IBUMAP_FAMILY | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Selects automatic, atomic, block-atomic, segmented, or serial point-to-mesh accumulation."),
    (_names("umap_sgd_update_mode"), "cpu_umap_auxiliary", CPU_UMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "auxiliary", "Only selects CPU UMAP asynchronous versus synchronous SGD."),
    (_names("rng_lifecycle"), "cpu_umap_auxiliary", CPU_UMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Selects independent per-stage streams or one UMAP-compatible shared RandomState for CPU graph, initialization, and sampling. An explicit optimization-only rng_state overrides sampling-state derivation."),
    (_names("n_components n_epochs"), "public_core", FULL_PIPELINE | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Shared embedding shape and optimizer schedule."),
    (_names("n_neighbors"), "graph", FULL_PIPELINE | {"fixed_knn"}, "supported", "KNN width; fixed-KNN also validates the supplied array width."),
    (_names("min_dist repulsion_strength"), "public_core", IBUMAP_FAMILY | CPU_UMAP | CUMl | {"optimization_only", "update_embedding"}, "supported", "UMAP low-dimensional objective parameters; unused by tFDP."),
    (_names("spread"), "public_core", IBUMAP_FAMILY | CPU_UMAP | {"optimization_only", "update_embedding"}, "supported", "Used to derive UMAP a/b; not forwarded to the cuML baseline."),
    (_names("negative_sample_rate learning_rate"), "ibumap_core", IBUMAP_FAMILY | CPU_UMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "UMAP/ibUMAP optimization controls; not forwarded to cuML and not consumed by the selected tFDP exact path."),
    (_names("refinement_fraction refinement_learning_rate refinement_negative_sample_rate"), "hybrid", HYBRID | {"optimization_only"}, "supported", "Independent CPU UMAP refinement-stage schedule and sampling controls."),
    (_names("random_state"), "initialization", FULL_PIPELINE | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Root random state. CPU UMAP uses independent graph/init/sampling streams by default or one call-scoped RandomState when rng_lifecycle='shared_rng'. An explicit NoiseConfig.seed/noise_seed overrides only the derived noise stream."),
    (_names("init spectral_method spectral_tol spectral_maxiter spectral_ncv spectral_auto_defaults spectral_scale_policy spectral_max_span spectral_jitter_relative spectral_jitter_max report_duplicate_ratio"), "initialization", INTERNAL | {"fixed_knn", "update_embedding"}, "supported", "Initialization method, spectral eigensolver controls, FFT-aware spectral range policy, and optional exact duplicate-ratio reporting."),
    (_names("metric"), "graph", INTERNAL | {"fixed_knn", "update_embedding"}, "supported", "Used by internal graph/initialization paths; current cuML baseline adapter does not forward it."),
    (_names("metric_kwds"), "graph", INTERNAL | {"fixed_knn", "update_embedding"}, "supported", "Used by CPU graph, CUDA cuML graph construction, and CPU/GPU initialization."),
    (_names("n_jobs low_memory"), "graph", {"ibumap_cpu", "ibumap_metal", "cpu_umap_async", "cpu_umap_sync", "tfdp_cpu"}, "supported", "CPU nearest-neighbor search, including the Metal hybrid pipeline."),
    (_names("angular_rp_forest"), "graph", {"ibumap_cpu", "ibumap_metal", "cpu_umap_async", "cpu_umap_sync", "tfdp_cpu", "fixed_knn"}, "supported", "CPU nearest-neighbor and fuzzy-graph construction; fixed-KNN CUDA and the Metal hybrid path build the fuzzy graph on CPU."),
    (_names("set_op_mix_ratio local_connectivity"), "graph", {"ibumap_cpu", "ibumap_cuda", "ibumap_metal", "cpu_umap_async", "cpu_umap_sync", "tfdp_cpu", "tfdp_cuda", "fixed_knn"}, "supported", "Fuzzy-graph construction controls for CPU, CUDA, and Metal hybrid paths; fixed-KNN accelerator paths build the fuzzy graph on CPU."),
    (_names("verbose"), "public_core", FULL_PIPELINE | {"fixed_knn", "update_embedding"}, "supported", "Controls available graph/optimizer logging; not consistently forwarded to the cuML estimator."),
    (_names("tqdm_kwds"), "cpu_umap_auxiliary", CPU_UMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "auxiliary", "CPU UMAP progress bar only."),
    (_names("deterministic"), "public_core", FULL_PIPELINE | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Requests same-device repeatability by supplying a fixed root seed when random_state is None; CPU graph construction and CPU UMAP use deterministic paths, while CUDA ibUMAP uses fixed-order CSR Laplacian construction plus deterministic CSR SpMV for spectral eigsh, auto-selects segmented P2M with a serial budget fallback, and reuses workspace-owned charge FFT plans/buffers outside minimal-workspace mode. CUDA lobpcg remains backend-limited."),
    (_names("fft_config"), "ibumap_core", IBUMAP_FAMILY | TFDP | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "Container for interpolation/FFT controls."),
    (_names("tfdp_config"), "legacy", TFDP | {"fixed_knn", "optimization_only", "update_embedding"}, "legacy", "Legacy tFDP optimizer configuration."),
    (_names("constraint_config"), "experimental", INTERNAL | {"fixed_knn", "optimization_only", "update_embedding"}, "partial", "Hard constraints work broadly; soft pull forces are implemented on CPU UMAP/ibUMAP but not GPU kernels or the selected tFDP exact path."),
    (_names("numerics_config"), "public_core", IBUMAP_FAMILY | CPU_UMAP | TFDP | {"fixed_knn", "optimization_only", "update_embedding"}, "partial", "epsilon and CPU dtype are consumed; CPU UMAP and hybrid require float32, while CPU ibUMAP/tFDP also accept float64. NumericsConfig.deterministic is unused. CUDA kernels are float32-only."),
    (_names("noise_config"), "experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "duplicated", "Alternative container for the seven flat noise/hybrid parameters."),
    (_names("ibfft_kernel_clip umap_epsilon"), "ibumap_stability", IBUMAP_FAMILY | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "ibFFT UMAP kernel stability controls. epsilon is also used by CPU UMAP and duplicates NumericsConfig.epsilon."),
    (_names("ibfft_kernel_subsample_mode ibfft_kernel_subsample_radius_cells ibfft_kernel_subsample_points"), "experimental", {"ibumap_cpu", "fixed_knn", "optimization_only", "update_embedding"}, "experimental", "Near-origin kernel discretization experiment; CPU ibUMAP only."),
    (_names("noise_mode noise_scale noise_decay noise_until_epoch noise_seed hybrid_mode hybrid_switch_epoch"), "experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "experimental", "Flat aliases for NoiseConfig; default-off and duplicated."),
    (_names("attraction_kernel_mode attraction_schedule_mode attraction_calendar_memory_limit_bytes attraction_warp_min_degree"), "experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "experimental", "ibUMAP attraction kernel/scheduler controls; kernel_mode=auto keeps CPU on row-thread and enables the CUDA warp-per-row fast path above warp_min_degree. CPU sampling experiments include active_edge_calendar and active_edge_periodic."),
    (_names("attraction_degree_damping attraction_degree_damping_mode attraction_degree_damping_ref attraction_degree_damping_ref_value attraction_degree_damping_power attraction_degree_damping_min_scale"), "ibumap_stability", IBUMAP_FAMILY | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "ibUMAP degree damping; enabled by default for ibUMAP and hybrid."),
    (_names("repulsion_clip_norm repulsion_clip_with_alpha repulsion_clip_epoch_range"), "ibumap_stability", IBUMAP_FAMILY | {"cpu_umap_sync", "fixed_knn", "optimization_only", "update_embedding"}, "supported", "Enabled by default for ibUMAP/hybrid; CPU UMAP synchronous also consumes it."),
    (_names("total_update_clip_norm total_update_clip_with_alpha total_update_clip_epoch_range"), "experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "experimental", "Default-off combined-update clipping."),
    (_names("diagnostics_path diagnostics_timing_path diagnostics_thresholds diagnostics_topk_path diagnostics_topk diagnostics_labels diagnostics_point_ids"), "diagnostics", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "diagnostics", "Default-off force/update CSV diagnostics. Timing diagnostics require CPU local-exact mode."),
    (_names("diagnostics_memory_path diagnostics_memory_epoch_stride diagnostics_memory_sample_interval_ms"), "diagnostics", FULL_PIPELINE | {"fixed_knn", "optimization_only", "update_embedding"}, "diagnostics", "Default-off stage and optional periodic memory JSONL diagnostics. Records snapshots only when diagnostics_memory_path is set."),
    (_names("local_exact_repulsion local_exact_k local_exact_radius_factor local_exact_weight local_exact_every local_exact_clip local_exact_symmetric local_exact_start_epoch local_exact_end_epoch local_exact_start_frac local_exact_end_frac local_exact_density_filter local_exact_min_cell_count local_exact_min_cell_count_quantile local_exact_density_include_neighbor_cells local_exact_timing_sample_size"), "experimental", {"ibumap_cpu", "fixed_knn", "optimization_only", "update_embedding"}, "experimental", "Default-off CPU-only exact near-field correction."),
    (_names("local_density_pressure local_density_pressure_weight local_density_pressure_every local_density_pressure_min_count local_density_pressure_clip local_density_pressure_power"), "experimental", {"ibumap_cpu", "fixed_knn", "optimization_only", "update_embedding"}, "experimental", "Default-off CPU-only density pressure correction."),
]


FLAT_TO_CANONICAL = {
    "algorithm": "runtime.algorithm",
    "device": "runtime.device",
    "graph_backend": "runtime.graph_backend",
    "random_state": "runtime.random_state",
    "workspace_policy": "runtime.workspace_policy",
    "workspace_limit_bytes": "runtime.workspace_limit_bytes",
    "p2m_mode": "fft.p2m_mode",
    "verbose": "runtime.verbose",
    "n_neighbors": "graph.n_neighbors",
    "metric": "graph.metric",
    "metric_kwds": "graph.metric_kwds",
    "n_jobs": "graph.n_jobs",
    "angular_rp_forest": "graph.angular_rp_forest",
    "low_memory": "graph.low_memory",
    "set_op_mix_ratio": "graph.set_op_mix_ratio",
    "local_connectivity": "graph.local_connectivity",
    "n_components": "umap.n_components",
    "n_epochs": "umap.n_epochs",
    "init": "initialization.init",
    "spectral_method": "initialization.spectral_method",
    "spectral_tol": "initialization.spectral_tol",
    "spectral_maxiter": "initialization.spectral_maxiter",
    "spectral_ncv": "initialization.spectral_ncv",
    "spectral_auto_defaults": "initialization.spectral_auto_defaults",
    "spectral_scale_policy": "initialization.spectral_scale_policy",
    "spectral_max_span": "initialization.spectral_max_span",
    "spectral_jitter_relative": "initialization.spectral_jitter_relative",
    "spectral_jitter_max": "initialization.spectral_jitter_max",
    "report_duplicate_ratio": "initialization.report_duplicate_ratio",
    "min_dist": "umap.min_dist",
    "spread": "umap.spread",
    "repulsion_strength": "umap.repulsion_strength",
    "negative_sample_rate": "umap.negative_sample_rate",
    "learning_rate": "umap.learning_rate",
    "refinement_fraction": "hybrid.refinement_fraction",
    "refinement_learning_rate": "hybrid.refinement_learning_rate",
    "refinement_negative_sample_rate": "hybrid.refinement_negative_sample_rate",
    "umap_epsilon": "umap.epsilon",
    "numerics_config": "legacy_numerics",
    "repulsion_clip_norm": "umap.repulsion_clipping.norm",
    "repulsion_clip_with_alpha": "umap.repulsion_clipping.with_alpha",
    "repulsion_clip_epoch_range": "umap.repulsion_clipping.epoch_range",
    "attraction_degree_damping": "umap.degree_damping.enabled",
    "attraction_degree_damping_mode": "umap.degree_damping.mode",
    "attraction_degree_damping_ref": "umap.degree_damping.ref",
    "attraction_degree_damping_ref_value": "umap.degree_damping.ref_value",
    "attraction_degree_damping_power": "umap.degree_damping.power",
    "attraction_degree_damping_min_scale": "umap.degree_damping.min_scale",
    "fft_config": "fft",
    "ibfft_kernel_clip": "fft.kernel_clip",
    "fft_kernel_cache_policy": "fft.kernel_cache_policy",
    "fft_kernel_cache_limit_bytes": "fft.kernel_cache_limit_bytes",
    "fft_kernel_cache_max_entries": "fft.kernel_cache_max_entries",
    "attraction_mode": "ibumap.attraction_mode",
    "repulsion_mode": "ibumap.repulsion_mode",
    "noise_config": "ibumap.experimental.noise",
    "noise_mode": "ibumap.experimental.noise.mode",
    "noise_scale": "ibumap.experimental.noise.scale",
    "noise_decay": "ibumap.experimental.noise.decay",
    "noise_until_epoch": "ibumap.experimental.noise.until_epoch",
    "noise_seed": "ibumap.experimental.noise.seed",
    "hybrid_mode": "ibumap.experimental.noise.hybrid_mode",
    "hybrid_switch_epoch": "ibumap.experimental.noise.hybrid_switch_epoch",
    "attraction_kernel_mode": "ibumap.experimental.attraction_schedule.kernel_mode",
    "attraction_schedule_mode": "ibumap.experimental.attraction_schedule.schedule_mode",
    "attraction_calendar_memory_limit_bytes": "ibumap.experimental.attraction_schedule.calendar_memory_limit_bytes",
    "attraction_warp_min_degree": "ibumap.experimental.attraction_schedule.warp_min_degree",
    "ibfft_kernel_subsample_mode": "ibumap.experimental.kernel_subsampling.mode",
    "ibfft_kernel_subsample_radius_cells": "ibumap.experimental.kernel_subsampling.radius_cells",
    "ibfft_kernel_subsample_points": "ibumap.experimental.kernel_subsampling.points",
    "total_update_clip_norm": "ibumap.experimental.total_update_clipping.norm",
    "total_update_clip_with_alpha": "ibumap.experimental.total_update_clipping.with_alpha",
    "total_update_clip_epoch_range": "ibumap.experimental.total_update_clipping.epoch_range",
    "local_exact_repulsion": "ibumap.experimental.local_exact_repulsion.enabled",
    "local_exact_k": "ibumap.experimental.local_exact_repulsion.k",
    "local_exact_radius_factor": "ibumap.experimental.local_exact_repulsion.radius_factor",
    "local_exact_weight": "ibumap.experimental.local_exact_repulsion.weight",
    "local_exact_every": "ibumap.experimental.local_exact_repulsion.every",
    "local_exact_clip": "ibumap.experimental.local_exact_repulsion.clip",
    "local_exact_symmetric": "ibumap.experimental.local_exact_repulsion.symmetric",
    "local_exact_start_epoch": "ibumap.experimental.local_exact_repulsion.start_epoch",
    "local_exact_end_epoch": "ibumap.experimental.local_exact_repulsion.end_epoch",
    "local_exact_start_frac": "ibumap.experimental.local_exact_repulsion.start_frac",
    "local_exact_end_frac": "ibumap.experimental.local_exact_repulsion.end_frac",
    "local_exact_density_filter": "ibumap.experimental.local_exact_repulsion.density_filter",
    "local_exact_min_cell_count": "ibumap.experimental.local_exact_repulsion.min_cell_count",
    "local_exact_min_cell_count_quantile": "ibumap.experimental.local_exact_repulsion.min_cell_count_quantile",
    "local_exact_density_include_neighbor_cells": "ibumap.experimental.local_exact_repulsion.density_include_neighbor_cells",
    "local_exact_timing_sample_size": "ibumap.experimental.local_exact_repulsion.timing_sample_size",
    "local_density_pressure": "ibumap.experimental.local_density_pressure.enabled",
    "local_density_pressure_weight": "ibumap.experimental.local_density_pressure.weight",
    "local_density_pressure_every": "ibumap.experimental.local_density_pressure.every",
    "local_density_pressure_min_count": "ibumap.experimental.local_density_pressure.min_count",
    "local_density_pressure_clip": "ibumap.experimental.local_density_pressure.clip",
    "local_density_pressure_power": "ibumap.experimental.local_density_pressure.power",
    "umap_sgd_update_mode": "cpu_umap.update_mode",
    "rng_lifecycle": "cpu_umap.rng_lifecycle",
    "deterministic": "cpu_umap.deterministic",
    "tqdm_kwds": "cpu_umap.tqdm_kwds",
    "tfdp_config": "tfdp",
    "constraint_config": "constraints",
    "diagnostics_path": "diagnostics.summary_path",
    "diagnostics_timing_path": "diagnostics.timing_path",
    "diagnostics_thresholds": "diagnostics.thresholds",
    "diagnostics_topk_path": "diagnostics.topk_path",
    "diagnostics_topk": "diagnostics.topk",
    "diagnostics_labels": "diagnostics.labels",
    "diagnostics_point_ids": "diagnostics.point_ids",
    "diagnostics_memory_path": "diagnostics.memory_path",
    "diagnostics_memory_epoch_stride": "diagnostics.memory_epoch_stride",
    "diagnostics_memory_sample_interval_ms": "diagnostics.memory_sample_interval_ms",
}

CANONICAL_TO_FLAT: dict[str, list[str]] = {}
for _flat_name, _canonical_name in FLAT_TO_CANONICAL.items():
    CANONICAL_TO_FLAT.setdefault(_canonical_name, []).append(_flat_name)


NESTED_GROUPS: dict[str, tuple[str, set[str], str, str]] = {
    "FFTConfig": ("ibumap_core", IBUMAP_FAMILY | TFDP | {"fixed_knn", "optimization_only", "update_embedding"}, "supported", "FFT/interpolation leaf setting."),
    "InitializationConfig": ("initialization", INTERNAL | {"fixed_knn", "update_embedding"}, "supported", "Initialization and spectral eigensolver leaf setting."),
    "TFDPConfig": ("legacy", TFDP | {"fixed_knn", "optimization_only", "update_embedding"}, "legacy", "Legacy tFDP-only leaf setting."),
    "ConstraintConfig": ("experimental", INTERNAL | {"fixed_knn", "optimization_only", "update_embedding"}, "partial", "Constraint leaf setting; GPU soft constraints and tFDP soft pulls are not implemented."),
    "NumericsConfig": ("public_core", IBUMAP_FAMILY | CPU_UMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "partial", "Numerics leaf setting."),
    "NoiseConfig": ("experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "duplicated", "Duplicates the flat noise/hybrid constructor parameters."),
    "AttractionScheduleConfig": ("experimental", IBUMAP | {"fixed_knn", "optimization_only", "update_embedding"}, "experimental", "ibUMAP attraction kernel/scheduler experiment leaf setting."),
    "HybridOptimizerConfig": ("hybrid", HYBRID | {"optimization_only"}, "supported", "Independent CPU UMAP refinement-stage setting."),
}


DEFAULT_RESULT_NAMES = _names(
    "algorithm device n_components n_neighbors n_epochs min_dist spread "
    "repulsion_strength negative_sample_rate learning_rate refinement_fraction "
    "refinement_learning_rate refinement_negative_sample_rate random_state metric "
    "metric_kwds set_op_mix_ratio local_connectivity init spectral_method "
    "spectral_tol spectral_maxiter spectral_ncv spectral_auto_defaults "
    "spectral_scale_policy spectral_max_span spectral_jitter_relative "
    "spectral_jitter_max report_duplicate_ratio fft_config attraction_mode "
    "repulsion_mode ibfft_kernel_clip fft_kernel_cache_policy "
    "fft_kernel_cache_limit_bytes fft_kernel_cache_max_entries "
    "umap_epsilon attraction_degree_damping "
    "attraction_degree_damping_mode attraction_degree_damping_ref "
    "attraction_degree_damping_power repulsion_clip_norm repulsion_clip_with_alpha "
    "attraction_kernel_mode attraction_schedule_mode attraction_calendar_memory_limit_bytes "
    "attraction_warp_min_degree"
)
DEFAULT_RUNTIME_NAMES = DEFAULT_RESULT_NAMES | _names(
    "graph_backend n_jobs low_memory angular_rp_forest verbose deterministic workspace_policy"
)
DEFAULT_RESULT_QUALIFIED = {
    "initialization.init",
    "initialization.spectral_method",
    "initialization.spectral_tol",
    "initialization.spectral_maxiter",
    "initialization.spectral_ncv",
    "initialization.spectral_auto_defaults",
    "initialization.spectral_scale_policy",
    "initialization.spectral_max_span",
    "initialization.spectral_jitter_relative",
    "initialization.spectral_jitter_max",
    "initialization.report_duplicate_ratio",
    "umap.epsilon",
    "fft.n_interpolation_points",
    "fft.intervals_per_integer",
    "fft.min_num_intervals",
    "fft.n_boxes_per_dim",
    "fft.combine_stages",
    "fft.interpolation_schedule",
    "fft.kernel_clip",
    "fft.kernel_cache_policy",
    "fft.kernel_cache_limit_bytes",
    "fft.kernel_cache_max_entries",
    "tfdp.alpha",
    "tfdp.beta",
    "tfdp.gamma",
    "tfdp.para_factor",
    "FFTConfig.n_interpolation_points",
    "FFTConfig.intervals_per_integer",
    "FFTConfig.min_num_intervals",
    "FFTConfig.n_boxes_per_dim",
    "FFTConfig.combine_stages",
    "FFTConfig.interpolation_schedule",
    "FFTConfig.kernel_cache_policy",
    "FFTConfig.kernel_cache_limit_bytes",
    "FFTConfig.kernel_cache_max_entries",
    "TFDPConfig.alpha",
    "TFDPConfig.beta",
    "TFDPConfig.gamma",
    "TFDPConfig.para_factor",
    "NumericsConfig.epsilon",
    "AttractionScheduleConfig.kernel_mode",
    "AttractionScheduleConfig.schedule_mode",
    "AttractionScheduleConfig.calendar_memory_limit_bytes",
    "AttractionScheduleConfig.warp_min_degree",
}


def _json_safe(value: Any) -> Any:
    if value is Parameter.empty or value is MISSING:
        return None
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _metadata_for(name: str) -> tuple[str, set[str], str, str]:
    for names, category, used_by, status, notes in GROUPS:
        if name in names:
            return category, set(used_by), status, notes
    return "unused", set(), "unused", "Accepted by the public surface but no consumption was found."


def _canonical_metadata(name: str) -> tuple[str, set[str], str, str]:
    aliases = CANONICAL_TO_FLAT.get(name, [])
    if aliases:
        return _metadata_for(aliases[0])
    root = name.split(".", 1)[0]
    if root == "runtime":
        return "public_core", set(FULL_PIPELINE), "supported", "Canonical runtime selection field."
    if root == "graph":
        return "graph", set(FULL_PIPELINE), "supported", "Canonical graph construction field."
    if root == "umap":
        return "public_core", set(INTERNAL | CUMl), "supported", "Canonical shared UMAP field."
    if root == "fft":
        return "ibumap_core", set(IBUMAP_FAMILY | TFDP), "supported", "Canonical FFT/interpolation field."
    if root == "ibumap":
        return "ibumap_core", set(IBUMAP_FAMILY), "supported", "Canonical ibUMAP-specific field."
    if root == "cpu_umap":
        return "cpu_umap_auxiliary", set(CPU_UMAP), "auxiliary", "Canonical CPU UMAP field."
    if root == "hybrid":
        return "hybrid", set(HYBRID), "supported", "Canonical hybrid refinement field."
    if root == "tfdp":
        return "legacy", set(TFDP), "legacy", "Canonical tFDP field."
    if root == "constraints":
        return "experimental", set(INTERNAL), "partial", "Canonical constraint field."
    if root == "diagnostics":
        return "diagnostics", set(IBUMAP), "diagnostics", "Canonical diagnostics field."
    return "legacy", set(), "legacy", "Legacy compatibility field."


def _record(
    name: str,
    default: Any,
    category: str,
    used_by: Iterable[str],
    status: str,
    notes: str,
    source_locations: list[str],
    *,
    canonical_name: Optional[str] = None,
    legacy_names: Iterable[str] = (),
    is_legacy_alias: bool = False,
) -> dict[str, Any]:
    used = set(used_by)
    base_name = name.rsplit(".", 1)[-1]
    affects_result = name in DEFAULT_RESULT_QUALIFIED or base_name in DEFAULT_RESULT_NAMES
    affects_runtime = name in DEFAULT_RESULT_QUALIFIED or base_name in DEFAULT_RUNTIME_NAMES
    if status == "unused":
        affects_result = False
        affects_runtime = False
    if category in {"diagnostics", "experimental"} and default in (None, False, 0, 0.0, "none"):
        affects_result = False
        affects_runtime = False
    return {
        "name": name,
        "default": _json_safe(default),
        "category": category,
        "used_by": sorted(used),
        "ignored_by": sorted(set(PATH_LABELS) - used),
        "affects_result_by_default": bool(affects_result),
        "affects_runtime_by_default": bool(affects_runtime),
        "status": status,
        "notes": notes,
        "source_locations": source_locations,
        "canonical_name": canonical_name or name,
        "legacy_names": sorted(set(legacy_names)),
        "is_legacy_alias": is_legacy_alias,
    }


def _iter_config_leaves(value: Any, prefix: str = ""):
    for item in fields(value):
        if item.name == "legacy_numerics" or item.name.startswith("_"):
            continue
        child = getattr(value, item.name)
        name = f"{prefix}.{item.name}" if prefix else item.name
        if is_dataclass(child):
            yield from _iter_config_leaves(child, name)
        else:
            yield name, child


def build_parameter_usage_registry() -> list[dict[str, Any]]:
    """Build the registry from signatures plus code-inspection metadata."""
    from .api import UMAPFFT

    records: list[dict[str, Any]] = []
    for name, parameter in signature(UMAPFFT.__init__).parameters.items():
        if name == "self":
            continue
        category, used_by, status, notes = _metadata_for(name)
        records.append(
            _record(
                name,
                parameter.default,
                category,
                used_by,
                status,
                notes,
                ["src/umap_fft/api.py:65"],
                canonical_name=FLAT_TO_CANONICAL.get(name, name),
                legacy_names=(
                    (name,)
                    if name in FLAT_TO_CANONICAL
                    else CANONICAL_TO_FLAT.get(name, ())
                ),
                is_legacy_alias=name in FLAT_TO_CANONICAL,
            )
        )

    defaults = UMAPFFTConfigBundle()
    for name, default in _iter_config_leaves(defaults):
        category, used_by, status, notes = _canonical_metadata(name)
        records.append(
            _record(
                name,
                default,
                category,
                used_by,
                status,
                notes,
                ["src/umap_fft/config.py"],
                legacy_names=CANONICAL_TO_FLAT.get(name, ()),
            )
        )

    for config_type in (
        FFTConfig,
        InitializationConfig,
        TFDPConfig,
        ConstraintConfig,
        NumericsConfig,
        NoiseConfig,
        AttractionScheduleConfig,
        HybridOptimizerConfig,
    ):
        category, used_by, status, notes = NESTED_GROUPS[config_type.__name__]
        for field in fields(config_type):
            if field.default is not MISSING:
                default = field.default
            elif field.default_factory is not MISSING:
                default = field.default_factory()
            else:
                default = None
            leaf_status = status
            leaf_notes = notes
            if config_type is NumericsConfig and field.name == "deterministic":
                leaf_status = "unused"
                leaf_notes = "Duplicated by top-level deterministic and never read by a pipeline."
                leaf_used_by: set[str] = set()
            elif config_type is NumericsConfig and field.name == "dtype":
                leaf_status = "partial"
                leaf_notes = "CPU ibUMAP and tFDP accept float32 or float64; CPU UMAP and hybrid require float32. CUDA and Metal kernels currently support float32 only."
                leaf_used_by = IBUMAP_FAMILY | CPU_UMAP | TFDP | {
                    "fixed_knn",
                    "optimization_only",
                    "update_embedding",
                }
            else:
                leaf_used_by = set(used_by)
            records.append(
                _record(
                    f"{config_type.__name__}.{field.name}",
                    default,
                    category if leaf_status != "unused" else "unused",
                    leaf_used_by,
                    leaf_status,
                    leaf_notes,
                    [f"src/umap_fft/config.py:{config_type.__name__}"],
                )
            )

    records.extend(
        [
            _record(
                "fit.y",
                None,
                "unused",
                set(),
                "unused",
                "Accepted by fit/fit_transform but never read or forwarded.",
                ["src/umap_fft/api.py:541"],
            ),
            _record(
                "update_embedding.init",
                None,
                "initialization",
                INTERNAL | CUMl | {"update_embedding"},
                "supported",
                "None continues from the current embedding; string/array values reinitialize. GPU UMAP still reruns cuML full fit.",
                ["src/umap_fft/api.py:628"],
            ),
            _record(
                "algorithm alias: drfft_umap",
                None,
                "deprecated",
                IBUMAP,
                "deprecated",
                "Backward-compatible alias canonicalized to ibumap and accompanied by FutureWarning.",
                ["src/umap_fft/config.py:158"],
            ),
            _record(
                "device alias: gpu",
                None,
                "deprecated",
                {"ibumap_cuda", "cuml_umap_baseline", "tfdp_cuda"},
                "deprecated",
                "Backward-compatible alias canonicalized to cuda and accompanied by FutureWarning.",
                ["src/umap_fft/config.py:171"],
            ),
        ]
    )
    return records


def _execution_path(model: Any) -> str:
    algorithm = model.runtime.algorithm
    if algorithm == "ibumap":
        return f"ibumap_{model.runtime.device}"
    if algorithm == "hybrid":
        return "hybrid_cpu"
    if algorithm == "umap" and model.runtime.device == "cpu":
        update_label = {
            "asynchronous": "async",
            "synchronous": "sync",
        }[model.runtime.umap_sgd_update_mode]
        return f"cpu_umap_{update_label}"
    if algorithm == "umap":
        return "cuml_umap_baseline"
    return f"tfdp_{model.runtime.device}"


def _model_value(model: Any, name: str) -> Any:
    if name.split(".", 1)[0] in {
        "runtime", "graph", "umap", "fft", "ibumap", "cpu_umap", "hybrid",
        "tfdp", "constraints", "diagnostics",
    }:
        value = model.config
        for part in name.split("."):
            # Optional mechanism groups (for example repulsion_clipping) are
            # represented by ``None`` when explicitly disabled.  Flat aliases
            # beneath that group should report ``None`` instead of making the
            # effective-config report unusable.
            if value is None:
                return None
            value = getattr(value, part)
        return value
    runtime_fields = {
        "algorithm",
        "device",
        "attraction_mode",
        "repulsion_mode",
        "graph_backend",
        "workspace_policy",
        "workspace_limit_bytes",
        "deterministic",
        "umap_sgd_update_mode",
    }
    if name in runtime_fields:
        return getattr(model.runtime, name)
    config_map = {
        "fft_config": model.runtime.fft,
        "tfdp_config": model.runtime.tfdp,
        "constraint_config": model.runtime.constraint,
        "numerics_config": model.runtime.numerics,
        "noise_config": model.runtime.noise,
    }
    if name in config_map:
        return config_map[name]
    flat_noise = {
        "noise_mode": "mode",
        "noise_scale": "scale",
        "noise_decay": "decay",
        "noise_until_epoch": "until_epoch",
        "noise_seed": "seed",
        "hybrid_mode": "hybrid_mode",
        "hybrid_switch_epoch": "hybrid_switch_epoch",
    }
    if name in flat_noise:
        return getattr(model.runtime.noise, flat_noise[name])
    if "." in name:
        type_name, field_name = name.split(".", 1)
        nested = {
            "FFTConfig": model.runtime.fft,
            "TFDPConfig": model.runtime.tfdp,
            "ConstraintConfig": model.runtime.constraint,
            "NumericsConfig": model.runtime.numerics,
            "NoiseConfig": model.runtime.noise,
            "HybridOptimizerConfig": model.runtime.hybrid,
        }.get(type_name)
        if nested is not None:
            return getattr(nested, field_name)
    if name == "fit.y" or name.startswith(("algorithm alias:", "device alias:")):
        return None
    if name == "update_embedding.init":
        return "method argument"
    return getattr(model, name, None)


def explain_model_config(model: Any) -> dict[str, Any]:
    path = _execution_path(model)
    entrypoint = getattr(model, "_last_entrypoint", None)
    usage_path = {
        "fit_transform_from_knn": "fixed_knn",
        "optimize_from_graph": "optimization_only",
        "optimize_from_prepared_graph": "optimization_only",
        "update_embedding": "update_embedding",
    }.get(entrypoint, path)
    registry = build_parameter_usage_registry()
    used: dict[str, Any] = {}
    ignored: dict[str, Any] = {}
    for record in registry:
        target = used if usage_path in record["used_by"] else ignored
        target[record["name"]] = _json_safe(_model_value(model, record["name"]))

    experimental_enabled: list[str] = []
    checks = {
        "ibfft_kernel_subsample_mode": model.ibfft_kernel_subsample_mode is not None,
        "noise": model.runtime.noise.mode != "none" and model.runtime.noise.scale > 0.0,
        "hybrid": model.runtime.noise.hybrid_mode != "none",
        "total_update_clip": model.total_update_clip_norm is not None,
        "local_exact_repulsion": model.local_exact_repulsion,
        "local_density_pressure": model.local_density_pressure,
    }
    experimental_enabled.extend(name for name, enabled in checks.items() if enabled)

    diagnostics = {
        "enabled": bool(
            model.diagnostics_path
            or model.diagnostics_timing_path
            or (model.diagnostics_topk_path and model.diagnostics_topk > 0)
            or model.diagnostics_memory_path
        ),
        "summary_path": model.diagnostics_path,
        "timing_path": model.diagnostics_timing_path,
        "topk_path": model.diagnostics_topk_path,
        "topk": model.diagnostics_topk,
        "memory_path": model.diagnostics_memory_path,
        "memory_epoch_stride": model.diagnostics_memory_epoch_stride,
        "memory_sample_interval_ms": model.diagnostics_memory_sample_interval_ms,
    }
    warnings: list[str] = []
    if path == "cuml_umap_baseline":
        warnings.append(
            "GPU algorithm='umap' delegates to cuML full fit; the separately "
            "built graph and spectral initialization are not consumed by cuML."
        )
    if path.startswith("tfdp_"):
        warnings.append("tfdp is retained as a legacy/experimental path.")
    if path == "ibumap_metal":
        warnings.append(
            "Metal uses a hybrid boundary: graph construction and initialization "
            "run on CPU, while optimization is dispatched to MLX Metal. The "
            "numerical optimizer is implemented incrementally in Stage 3."
        )
    constraint = model.runtime.constraint
    if (
        constraint.mode == "soft"
        and constraint.known_points_indices is not None
        and path in {"ibumap_cuda", "ibumap_metal", "tfdp_cpu", "tfdp_cuda"}
    ):
        warnings.append("Soft constraint pull forces are not implemented by this path.")
    if model.runtime.numerics.deterministic != model.runtime.deterministic:
        warnings.append(
            "NumericsConfig.deterministic is unused; top-level deterministic is authoritative."
        )
    if path in IBUMAP_FAMILY | TFDP and model.n_components != 2:
        warnings.append("The selected optimizer kernels are primarily implemented for 2D embeddings.")

    algorithm_aliases = sorted(set(model._deprecated_algorithm_aliases))
    device_aliases = sorted(set(model._deprecated_device_aliases))
    grouped_config = _json_safe(model.config)
    grouped_config.pop("legacy_numerics", None)
    grouped_config["umap"]["degree_damping"]["resolved_enabled"] = bool(
        model.attraction_degree_damping
    )
    entrypoint_bypasses = {
        "fit_transform_from_knn": ["nearest_neighbor_search"],
        "optimize_from_graph": ["nearest_neighbor_search", "graph_construction", "initialization"],
        "optimize_from_prepared_graph": [
            "nearest_neighbor_search",
            "graph_construction",
            "graph_preprocessing",
            "initialization",
        ],
        "update_embedding": ["nearest_neighbor_search", "graph_construction"],
    }.get(entrypoint, [])
    if path == "ibumap_metal":
        graph_device = "cpu"
        init_device = "cpu"
        optimizer_device = "metal"
    else:
        graph_device = model.runtime.device
        init_device = model.runtime.device
        optimizer_device = model.runtime.device

    report = {
        "schema_version": 2,
        "config": grouped_config,
        "legacy_flat_parameters_used": list(model._legacy_flat_parameters_used),
        "entrypoint": entrypoint,
        "parameter_usage_path": usage_path,
        "entrypoint_bypasses": entrypoint_bypasses,
        "update_init_overrode_config": bool(
            entrypoint == "update_embedding"
            and getattr(model, "_last_update_init_override", False)
        ),
        "canonical_algorithm": model.runtime.algorithm,
        "canonical_device": model.runtime.device,
        "deprecated_aliases_used": sorted(set(algorithm_aliases + device_aliases)),
        "deprecated_algorithm_aliases_used": algorithm_aliases,
        "deprecated_device_aliases_used": device_aliases,
        "device": model.runtime.device,
        "execution_path": path,
        "graph_device": graph_device,
        "init_device": init_device,
        "optimizer_device": optimizer_device,
        "used_parameters": used,
        "ignored_parameters": ignored,
        "experimental_parameters_enabled": experimental_enabled,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "registry_version": REGISTRY_VERSION,
    }
    if path == "ibumap_metal":
        from .runtime.metal import metal_capability_matrix

        report["metal_capabilities"] = metal_capability_matrix()
    return report


def write_registry_docs(output_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Write the checked-in JSON and Markdown registry artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = build_parameter_usage_registry()
    version_slug = REGISTRY_VERSION.replace("-", "")
    json_path = output / f"{version_slug}-parameter_usage_registry.json"
    md_path = output / f"{version_slug}-parameter_usage_registry.md"
    json_path.write_text(
        json.dumps(
            {"registry_version": REGISTRY_VERSION, "path_labels": PATH_LABELS, "parameters": records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# UMAPFFT Parameter Usage Registry ({REGISTRY_VERSION})",
        "",
        "This file is generated from `umap_fft.parameter_usage`. `used_by` records actual code consumption; an accepted parameter that is absent from a path is intentionally listed under `ignored_by` in the JSON registry. Flat compatibility inputs are marked as aliases and point to their canonical dotted name.",
        "",
        f"Total records: **{len(records)}** (UMAPFFT constructor parameters, config leaf fields, method-only parameters, and deprecated aliases).",
        "",
        "| Name | Canonical name | Alias | Default | Category | Used by | Status | Default result/runtime effect | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        default = json.dumps(record["default"], ensure_ascii=False)
        used_by = ", ".join(record["used_by"]) or "—"
        effect = f"{record['affects_result_by_default']}/{record['affects_runtime_by_default']}"
        notes = record["notes"].replace("|", "\\|")
        alias = "yes" if record["is_legacy_alias"] else "no"
        lines.append(
            f"| `{record['name']}` | `{record['canonical_name']}` | {alias} | `{default}` | {record['category']} | {used_by} | {record['status']} | {effect} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## Duplicated and conflicting inputs",
            "",
            "- `umap_epsilon`, `NumericsConfig.epsilon`, and `umap.epsilon` are equal representations of one value; unequal explicit inputs raise `ValueError`.",
            "- `NumericsConfig.dtype` accepts `float32` or `float64` for CPU ibUMAP and tFDP; CPU UMAP, hybrid, CUDA, and Metal paths require `float32`.",
            "- top-level `deterministic` is authoritative for same-device repeatability across pipelines; `NumericsConfig.deterministic` is currently unused.",
            "- the seven flat noise/hybrid parameters and `NoiseConfig` are compatibility representations of the same settings; equal duplicates are accepted and unequal duplicates raise `ValueError`.",
            "- `algorithm='drfft_umap'` is a deprecated alias and is canonicalized to `algorithm='ibumap'`.",
            "- `device='gpu'` is a deprecated alias and is canonicalized to `device='cuda'`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs")
    args = parser.parse_args()
    markdown, json_file = write_registry_docs(args.output_dir)
    print(markdown)
    print(json_file)
