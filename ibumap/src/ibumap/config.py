from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union
import warnings

import numpy as np

from .fft_schedule import FFTStage, normalize_fft_schedule

Algorithm = Literal["umap", "ibumap", "drfft_umap", "hybrid", "tfdp"]
Device = Literal["cpu", "cuda", "metal"]
AttractionMode = Literal["sampling", "true_loss"]
RepulsionMode = Literal["sampling", "true_loss"]
UMAPSGDUpdateMode = Literal["asynchronous", "synchronous"]
RNGLifecycle = Literal["independent", "shared_rng"]
GraphBackend = Literal["auto", "cpu", "cuml"]
WorkspacePolicy = Literal[
    "auto", "performance", "minimal", "fast", "low_memory"
]
P2MMode = Literal["auto", "atomic", "block_atomic", "segmented", "serial"]
FFTKernelCachePolicy = Literal["auto", "legacy", "byte_lru", "disabled"]
AttractionKernelMode = Literal["thread_per_row", "warp_per_row", "auto"]
AttractionScheduleMode = Literal[
    "row_scan", "active_edge_calendar", "active_edge_periodic", "auto"
]
ConstraintMode = Literal["hard", "soft"]
NoiseMode = Literal["none", "force", "embedding"]
NoiseDecay = Literal["constant", "linear", "exponential"]
HybridMode = Literal["none", "early_noisy_late_deterministic"]
NumericDType = Literal["float32", "float64"]
SpectralScalePolicy = Literal[
    "auto",
    "fft_compact",
    "raw",
    "legacy_box10",
]


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"

    def __deepcopy__(self, memo):
        return self


_UNSET = _UnsetType()


def resolve_numeric_dtype(dtype: Any) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be either 'float32' or 'float64'")
    return resolved


def _normalize_init(init: Any) -> Any:
    if isinstance(init, str):
        if init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral', 'random', or a 2D array")
        return init
    init_array = np.asarray(init)
    if init_array.ndim != 2:
        raise ValueError("custom init must be a 2D array")
    if not np.issubdtype(init_array.dtype, np.number):
        raise ValueError("custom init must be numeric")
    return np.array(init_array, order="C", copy=True)


@dataclass(slots=True)
class FFTConfig:
    n_interpolation_points: int = 1
    intervals_per_integer: float = 1.0
    min_num_intervals: int = 100
    n_boxes_per_dim: float = 1.0
    combine_stages: bool = False
    kernel_clip: float = 4.0
    p2m_mode: P2MMode = "auto"
    kernel_cache_policy: FFTKernelCachePolicy = "auto"
    kernel_cache_limit_bytes: Optional[int] = None
    kernel_cache_max_entries: Optional[int] = None
    interpolation_schedule: Optional[Sequence[FFTStage]] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_interpolation_points, bool)
            or not isinstance(self.n_interpolation_points, (int, np.integer))
            or self.n_interpolation_points < 1
        ):
            raise ValueError("n_interpolation_points must be a positive integer")
        self.n_interpolation_points = int(self.n_interpolation_points)
        if self.intervals_per_integer <= 0:
            raise ValueError("intervals_per_integer must be > 0")
        if self.min_num_intervals < 1:
            raise ValueError("min_num_intervals must be >= 1")
        if self.n_boxes_per_dim <= 0:
            raise ValueError("n_boxes_per_dim must be > 0")
        if not isinstance(self.combine_stages, (bool, np.bool_)):
            raise ValueError("combine_stages must be a boolean")
        self.combine_stages = bool(self.combine_stages)
        self.interpolation_schedule = normalize_fft_schedule(
            self.interpolation_schedule
        )
        if self.combine_stages and self.interpolation_schedule is not None:
            raise ValueError(
                "Use interpolation_schedule or combine_stages=True, not both"
            )
        if self.combine_stages and self.n_interpolation_points != 1:
            raise ValueError(
                "combine_stages=True requires n_interpolation_points=1; use an "
                "explicit interpolation_schedule for other starting orders"
            )
        if self.combine_stages:
            warnings.warn(
                "FFTConfig.combine_stages=True is deprecated; use "
                "interpolation_schedule=(FFTStage(0.0, 1), "
                "FFTStage(0.90, 2), FFTStage(0.95, 3)) instead",
                FutureWarning,
                stacklevel=2,
            )
        self.kernel_clip = float(self.kernel_clip)
        if not np.isfinite(self.kernel_clip) or self.kernel_clip <= 0:
            raise ValueError("kernel_clip must be positive and finite")
        if self.p2m_mode not in (
            "auto",
            "atomic",
            "block_atomic",
            "segmented",
            "serial",
        ):
            raise ValueError(
                "p2m_mode must be one of: auto, atomic, block_atomic, "
                "segmented, serial"
            )
        if self.kernel_cache_policy not in (
            "auto",
            "legacy",
            "byte_lru",
            "disabled",
        ):
            raise ValueError(
                "kernel_cache_policy must be one of: auto, legacy, "
                "byte_lru, disabled"
            )
        if self.kernel_cache_limit_bytes is not None:
            if (
                isinstance(self.kernel_cache_limit_bytes, bool)
                or int(self.kernel_cache_limit_bytes) < 0
            ):
                raise ValueError(
                    "kernel_cache_limit_bytes must be a non-negative integer"
                )
            self.kernel_cache_limit_bytes = int(self.kernel_cache_limit_bytes)
        if self.kernel_cache_max_entries is not None:
            if (
                isinstance(self.kernel_cache_max_entries, bool)
                or int(self.kernel_cache_max_entries) < 1
            ):
                raise ValueError(
                    "kernel_cache_max_entries must be a positive integer or None"
                )
            self.kernel_cache_max_entries = int(self.kernel_cache_max_entries)


@dataclass(slots=True)
class RuntimeConfig:
    algorithm: Algorithm = "ibumap"
    device: Device = "cpu"
    graph_backend: GraphBackend = "auto"
    random_state: Any = None
    workspace_policy: WorkspacePolicy = "performance"
    workspace_limit_bytes: Optional[int] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.algorithm == "drfft_umap":
            warnings.warn(
                "algorithm='drfft_umap' is deprecated and will be replaced by "
                "algorithm='ibumap'; use 'ibumap' instead",
                FutureWarning,
                stacklevel=3,
            )
            self.algorithm = "ibumap"
        if self.algorithm not in ("umap", "ibumap", "hybrid", "tfdp"):
            raise ValueError(
                "algorithm must be one of: umap, ibumap, hybrid, tfdp "
                "(drfft_umap is a deprecated alias for ibumap)"
            )
        if self.device == "gpu":
            warnings.warn(
                "device='gpu' is deprecated and will be replaced by "
                "device='cuda'; use 'cuda' instead",
                FutureWarning,
                stacklevel=3,
            )
            self.device = "cuda"
        if self.device not in ("cpu", "cuda", "metal"):
            raise ValueError(
                "device must be one of: cpu, cuda, metal "
                "(gpu is a deprecated alias for cuda)"
            )
        if self.algorithm == "hybrid" and self.device != "cpu":
            raise NotImplementedError(
                "algorithm='hybrid' currently supports device='cpu' only"
            )
        if self.graph_backend not in ("auto", "cpu", "cuml"):
            raise ValueError("graph_backend must be one of: auto, cpu, cuml")
        aliases = {"fast": "performance", "low_memory": "minimal"}
        self.workspace_policy = aliases.get(self.workspace_policy, self.workspace_policy)
        if self.workspace_policy not in ("auto", "performance", "minimal"):
            raise ValueError(
                "workspace_policy must be one of: auto, performance, minimal "
                "(fast and low_memory are compatibility aliases)"
            )
        if self.workspace_limit_bytes is not None:
            if (
                isinstance(self.workspace_limit_bytes, bool)
                or int(self.workspace_limit_bytes) < 0
            ):
                raise ValueError("workspace_limit_bytes must be a non-negative integer")
            self.workspace_limit_bytes = int(self.workspace_limit_bytes)
        self.verbose = bool(self.verbose)
        if isinstance(self.random_state, np.random.RandomState):
            copied = np.random.RandomState()
            copied.set_state(self.random_state.get_state())
            self.random_state = copied


@dataclass(slots=True)
class GraphConfig:
    n_neighbors: int = 15
    metric: str = "euclidean"
    metric_kwds: Optional[Mapping[str, Any]] = None
    n_jobs: int = -1
    angular_rp_forest: bool = False
    low_memory: bool = True
    set_op_mix_ratio: float = 1.0
    local_connectivity: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.n_neighbors, bool) or int(self.n_neighbors) < 2:
            raise ValueError("n_neighbors must be an integer >= 2")
        self.n_neighbors = int(self.n_neighbors)
        self.metric = str(self.metric)
        self.metric_kwds = dict(self.metric_kwds or {})
        self.n_jobs = int(self.n_jobs)
        self.angular_rp_forest = bool(self.angular_rp_forest)
        self.low_memory = bool(self.low_memory)
        self.set_op_mix_ratio = float(self.set_op_mix_ratio)
        self.local_connectivity = float(self.local_connectivity)
        if not 0.0 <= self.set_op_mix_ratio <= 1.0:
            raise ValueError("set_op_mix_ratio must be in [0, 1]")
        if self.local_connectivity < 0:
            raise ValueError("local_connectivity must be >= 0")


@dataclass(slots=True)
class ForceClippingConfig:
    norm: float
    with_alpha: bool
    epoch_range: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        self.norm = float(self.norm)
        if not np.isfinite(self.norm) or self.norm <= 0:
            raise ValueError("norm must be positive and finite")
        self.with_alpha = bool(self.with_alpha)
        if self.epoch_range is not None:
            if isinstance(self.epoch_range, (str, bytes)):
                raise ValueError("epoch_range must be a length-two integer sequence")
            try:
                values = tuple(self.epoch_range)
            except TypeError as exc:
                raise ValueError(
                    "epoch_range must be a length-two integer sequence"
                ) from exc
            if len(values) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise ValueError("epoch_range must be a length-two integer sequence")
            start, end = int(values[0]), int(values[1])
            if start < 0 or start >= end:
                raise ValueError("epoch_range must satisfy 0 <= start < end")
            self.epoch_range = (start, end)


@dataclass(slots=True)
class DegreeDampingConfig:
    enabled: Optional[bool] = None
    mode: Literal["weighted_degree", "degree"] = "weighted_degree"
    ref: Literal["median", "mean", "p90", "p95", "p99", "manual"] = "p99"
    ref_value: Optional[float] = None
    power: float = 0.5
    min_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.enabled is not None:
            self.enabled = bool(self.enabled)
        if self.mode not in ("weighted_degree", "degree"):
            raise ValueError("mode must be 'weighted_degree' or 'degree'")
        if self.ref not in ("median", "mean", "p90", "p95", "p99", "manual"):
            raise ValueError(
                "ref must be one of: median, mean, p90, p95, p99, manual"
            )
        if self.ref_value is not None:
            self.ref_value = float(self.ref_value)
            if not np.isfinite(self.ref_value) or self.ref_value <= 0:
                raise ValueError("ref_value must be positive and finite")
        if self.ref == "manual" and self.ref_value is None:
            raise ValueError("ref_value is required when ref='manual'")
        self.power = float(self.power)
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be positive and finite")
        if self.min_scale is not None:
            self.min_scale = float(self.min_scale)
            if not np.isfinite(self.min_scale) or not 0 < self.min_scale <= 1:
                raise ValueError("min_scale must be in (0, 1]")


@dataclass(slots=True)
class InitializationConfig:
    init: Any = "spectral"
    spectral_method: Optional[Literal["eigsh", "lobpcg"]] = None
    spectral_tol: Optional[float] = None
    spectral_maxiter: Optional[int] = None
    spectral_ncv: Optional[int] = None
    spectral_auto_defaults: Optional[bool] = None
    spectral_scale_policy: SpectralScalePolicy = "auto"
    spectral_max_span: float = 0.1
    spectral_jitter_relative: float = 1e-4
    spectral_jitter_max: float = 1e-4
    report_duplicate_ratio: bool = False

    def __post_init__(self) -> None:
        self.init = _normalize_init(self.init)
        if self.spectral_method is not None:
            self.spectral_method = str(self.spectral_method)
            if self.spectral_method not in ("eigsh", "lobpcg"):
                raise ValueError("spectral_method must be one of: eigsh, lobpcg")
        if self.spectral_tol is not None:
            self.spectral_tol = float(self.spectral_tol)
            if not np.isfinite(self.spectral_tol) or self.spectral_tol <= 0:
                raise ValueError("spectral_tol must be positive and finite")
        if self.spectral_maxiter is not None:
            if (
                isinstance(self.spectral_maxiter, bool)
                or not isinstance(self.spectral_maxiter, (int, np.integer))
                or self.spectral_maxiter < 1
            ):
                raise ValueError("spectral_maxiter must be a positive integer")
            self.spectral_maxiter = int(self.spectral_maxiter)
        if self.spectral_ncv is not None:
            if (
                isinstance(self.spectral_ncv, bool)
                or not isinstance(self.spectral_ncv, (int, np.integer))
                or self.spectral_ncv < 1
            ):
                raise ValueError("spectral_ncv must be a positive integer")
            self.spectral_ncv = int(self.spectral_ncv)
        if self.spectral_auto_defaults is not None:
            if not isinstance(self.spectral_auto_defaults, (bool, np.bool_)):
                raise ValueError("spectral_auto_defaults must be a bool or None")
            self.spectral_auto_defaults = bool(self.spectral_auto_defaults)
        self.spectral_scale_policy = str(self.spectral_scale_policy)
        if self.spectral_scale_policy not in (
            "auto",
            "fft_compact",
            "raw",
            "legacy_box10",
        ):
            raise ValueError(
                "spectral_scale_policy must be one of: auto, fft_compact, "
                "raw, legacy_box10"
            )
        self.spectral_max_span = float(self.spectral_max_span)
        if (
            not np.isfinite(self.spectral_max_span)
            or self.spectral_max_span <= 0.0
        ):
            raise ValueError("spectral_max_span must be positive and finite")
        self.spectral_jitter_relative = float(self.spectral_jitter_relative)
        if (
            not np.isfinite(self.spectral_jitter_relative)
            or self.spectral_jitter_relative < 0.0
        ):
            raise ValueError(
                "spectral_jitter_relative must be non-negative and finite"
            )
        self.spectral_jitter_max = float(self.spectral_jitter_max)
        if (
            not np.isfinite(self.spectral_jitter_max)
            or self.spectral_jitter_max < 0.0
        ):
            raise ValueError(
                "spectral_jitter_max must be non-negative and finite"
            )
        if not isinstance(self.report_duplicate_ratio, (bool, np.bool_)):
            raise ValueError("report_duplicate_ratio must be a bool")
        self.report_duplicate_ratio = bool(self.report_duplicate_ratio)

    def resolve_spectral_scale_policy(self, algorithm: str) -> str:
        """Resolve the algorithm-aware default without changing explicit choices."""

        if self.spectral_scale_policy != "auto":
            return self.spectral_scale_policy
        if algorithm in ("ibumap", "drfft_umap", "hybrid"):
            return "fft_compact"
        return "legacy_box10"


@dataclass(slots=True)
class UMAPConfig:
    n_components: int = 2
    n_epochs: Optional[int] = None
    min_dist: float = 0.2
    spread: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: float = 5.0
    learning_rate: float = 1.0
    epsilon: float = 1e-3
    repulsion_clipping: Optional[ForceClippingConfig] = field(
        default_factory=lambda: ForceClippingConfig(norm=4.0, with_alpha=True)
    )
    degree_damping: DegreeDampingConfig = field(default_factory=DegreeDampingConfig)
    init: InitVar[Any] = _UNSET
    _legacy_init: Any = field(default=_UNSET, init=False, repr=False, compare=False)

    def __post_init__(self, init: Any = _UNSET) -> None:
        if isinstance(self.n_components, bool) or int(self.n_components) < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = int(self.n_components)
        if self.n_epochs is not None:
            if isinstance(self.n_epochs, bool) or int(self.n_epochs) < 1:
                raise ValueError("n_epochs must be a positive integer when set")
            self.n_epochs = int(self.n_epochs)
        if init is not _UNSET:
            self._legacy_init = _normalize_init(init)
        self.min_dist = float(self.min_dist)
        self.spread = float(self.spread)
        self.repulsion_strength = float(self.repulsion_strength)
        self.negative_sample_rate = float(self.negative_sample_rate)
        self.learning_rate = float(self.learning_rate)
        self.epsilon = float(self.epsilon)
        if self.min_dist < 0:
            raise ValueError("min_dist must be >= 0")
        if self.spread <= 0:
            raise ValueError("spread must be > 0")
        if self.repulsion_strength <= 0:
            raise ValueError("repulsion_strength must be > 0")
        if self.negative_sample_rate <= 0:
            raise ValueError("negative_sample_rate must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if self.repulsion_clipping is not None:
            self.repulsion_clipping = deepcopy(self.repulsion_clipping)
            self.repulsion_clipping.__post_init__()
        self.degree_damping = deepcopy(self.degree_damping)
        self.degree_damping.__post_init__()


@dataclass(slots=True)
class TFDPConfig:
    alpha: float = 0.1
    beta: float = 8.0
    gamma: float = 2.0
    para_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.para_factor <= 0:
            raise ValueError("para_factor must be > 0")


@dataclass(slots=True)
class ConstraintConfig:
    mode: ConstraintMode = "hard"
    weight: float = 0.1
    known_points_indices: Optional[Sequence[int]] = None
    known_points_positions: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")

        if self.known_points_indices is None and self.known_points_positions is None:
            return
        if self.known_points_indices is None or self.known_points_positions is None:
            raise ValueError(
                "known_points_indices and known_points_positions must both be set"
            )

        idx = np.asarray(self.known_points_indices)
        pos = np.asarray(self.known_points_positions)
        if idx.ndim != 1:
            raise ValueError("known_points_indices must be 1D")
        if pos.ndim != 2:
            raise ValueError("known_points_positions must be 2D")
        if not np.issubdtype(pos.dtype, np.number):
            raise ValueError("known_points_positions must be numeric")
        if pos.shape[0] != idx.shape[0]:
            raise ValueError(
                "known_points_positions first dim must match known_points_indices"
            )

        self.known_points_indices = idx.astype(np.int64, copy=False)
        self.known_points_positions = np.array(pos, order="C", copy=True)


@dataclass(slots=True)
class NumericsConfig:
    dtype: NumericDType = "float32"
    deterministic: bool = False
    epsilon: float = 1e-3   # umap_epsilon

    def __post_init__(self) -> None:
        self.dtype = resolve_numeric_dtype(self.dtype).name
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")


@dataclass(slots=True)
class NoiseConfig:
    mode: NoiseMode = "none"
    scale: float = 0.0
    decay: NoiseDecay = "linear"
    until_epoch: Optional[int] = None
    seed: Optional[int] = None
    hybrid_mode: HybridMode = "none"
    hybrid_switch_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("none", "force", "embedding"):
            raise ValueError("noise_mode must be one of: none, force, embedding")
        self.scale = float(self.scale)
        if self.scale < 0:
            raise ValueError("noise_scale must be >= 0")
        if self.decay not in ("constant", "linear", "exponential"):
            raise ValueError(
                "noise_decay must be one of: constant, linear, exponential"
            )
        if self.until_epoch is not None:
            self.until_epoch = int(self.until_epoch)
            if self.until_epoch < 0:
                raise ValueError("noise_until_epoch must be >= 0")
        if self.seed is not None:
            self.seed = int(self.seed)
        if self.hybrid_mode not in ("none", "early_noisy_late_deterministic"):
            raise ValueError(
                "hybrid_mode must be one of: none, early_noisy_late_deterministic"
            )
        if self.hybrid_switch_epoch is not None:
            self.hybrid_switch_epoch = int(self.hybrid_switch_epoch)
            if self.hybrid_switch_epoch < 0:
                raise ValueError("hybrid_switch_epoch must be >= 0")


@dataclass(slots=True)
class LocalExactRepulsionConfig:
    enabled: bool = False
    k: int = 8
    radius_factor: float = 0.5
    weight: float = 0.1
    every: int = 1
    clip: float = 4.0
    symmetric: bool = True
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    start_frac: Optional[float] = None
    end_frac: Optional[float] = None
    density_filter: bool = False
    min_cell_count: Optional[int] = None
    min_cell_count_quantile: Optional[float] = None
    density_include_neighbor_cells: bool = False
    timing_sample_size: int = 2048

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if (
            isinstance(self.k, bool)
            or not isinstance(self.k, (int, np.integer))
            or self.k < 1
        ):
            raise ValueError("k must be a positive integer")
        self.k = int(self.k)
        self.radius_factor = float(self.radius_factor)
        self.weight = float(self.weight)
        self.clip = float(self.clip)
        if self.radius_factor <= 0:
            raise ValueError("radius_factor must be positive")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if (
            isinstance(self.every, bool)
            or not isinstance(self.every, (int, np.integer))
            or self.every < 1
        ):
            raise ValueError("every must be a positive integer")
        self.every = int(self.every)
        if self.clip <= 0:
            raise ValueError("clip must be positive")
        for name in ("start_epoch", "end_epoch"):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")
                setattr(self, name, int(value))
        if self.end_epoch == 0:
            raise ValueError("end_epoch must be positive when set")
        if (
            self.start_epoch is not None
            and self.end_epoch is not None
            and self.start_epoch >= self.end_epoch
        ):
            raise ValueError("epoch bounds must satisfy start < end")
        if any(value is not None for value in (self.start_epoch, self.end_epoch)) and any(
            value is not None for value in (self.start_frac, self.end_frac)
        ):
            raise ValueError("Use epoch bounds or fraction bounds, not both")
        if self.start_frac is not None:
            self.start_frac = float(self.start_frac)
            if not 0 <= self.start_frac < 1:
                raise ValueError("start_frac must be in [0, 1)")
        if self.end_frac is not None:
            self.end_frac = float(self.end_frac)
            if not 0 < self.end_frac <= 1:
                raise ValueError("end_frac must be in (0, 1]")
        if (
            self.start_frac is not None
            and self.end_frac is not None
            and self.start_frac >= self.end_frac
        ):
            raise ValueError("fraction bounds must satisfy start < end")
        if self.min_cell_count is not None:
            if (
                isinstance(self.min_cell_count, bool)
                or not isinstance(self.min_cell_count, (int, np.integer))
                or self.min_cell_count < 1
            ):
                raise ValueError("min_cell_count must be a positive integer")
            self.min_cell_count = int(self.min_cell_count)
        if self.min_cell_count_quantile is not None:
            self.min_cell_count_quantile = float(self.min_cell_count_quantile)
            if not 0 <= self.min_cell_count_quantile <= 1:
                raise ValueError("min_cell_count_quantile must be in [0, 1]")
        if self.min_cell_count is not None and self.min_cell_count_quantile is not None:
            raise ValueError("Use min_cell_count or min_cell_count_quantile, not both")
        self.density_filter = bool(
            self.density_filter
            or self.min_cell_count is not None
            or self.min_cell_count_quantile is not None
        )
        self.density_include_neighbor_cells = bool(
            self.density_include_neighbor_cells
        )
        if (
            isinstance(self.timing_sample_size, bool)
            or not isinstance(self.timing_sample_size, (int, np.integer))
            or self.timing_sample_size < 1
        ):
            raise ValueError("timing_sample_size must be a positive integer")
        self.timing_sample_size = int(self.timing_sample_size)


@dataclass(slots=True)
class LocalDensityPressureConfig:
    enabled: bool = False
    weight: float = 0.05
    every: int = 1
    min_count: int = 4
    clip: float = 4.0
    power: float = 1.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.weight = float(self.weight)
        self.clip = float(self.clip)
        self.power = float(self.power)
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if (
            isinstance(self.every, bool)
            or not isinstance(self.every, (int, np.integer))
            or self.every < 1
        ):
            raise ValueError("every must be a positive integer")
        if (
            isinstance(self.min_count, bool)
            or not isinstance(self.min_count, (int, np.integer))
            or self.min_count < 1
        ):
            raise ValueError("min_count must be a positive integer")
        self.every = int(self.every)
        self.min_count = int(self.min_count)
        if self.clip <= 0:
            raise ValueError("clip must be positive")
        if self.power <= 0:
            raise ValueError("power must be positive")


@dataclass(slots=True)
class KernelSubsamplingConfig:
    mode: Literal["near_origin"] = "near_origin"
    radius_cells: int = 2
    points: int = 4

    def __post_init__(self) -> None:
        if self.mode != "near_origin":
            raise ValueError("mode must be 'near_origin'")
        if (
            isinstance(self.radius_cells, bool)
            or not isinstance(self.radius_cells, (int, np.integer))
            or self.radius_cells < 0
        ):
            raise ValueError("radius_cells must be a non-negative integer")
        if (
            isinstance(self.points, bool)
            or not isinstance(self.points, (int, np.integer))
            or self.points < 1
        ):
            raise ValueError("points must be a positive integer")
        self.radius_cells = int(self.radius_cells)
        self.points = int(self.points)


@dataclass(slots=True)
class AttractionScheduleConfig:
    """Experimental ibUMAP attraction execution controls.

    The default kernel mode is device-adaptive: CPU resolves to the established
    source-row scan kernel, while CUDA may use the warp-per-row fast path when
    the graph degree is high enough.
    """

    kernel_mode: AttractionKernelMode = "auto"
    schedule_mode: AttractionScheduleMode = "row_scan"
    calendar_memory_limit_bytes: Optional[int] = None
    warp_min_degree: int = 8

    def __post_init__(self) -> None:
        if self.kernel_mode not in ("thread_per_row", "warp_per_row", "auto"):
            raise ValueError(
                "kernel_mode must be one of: thread_per_row, warp_per_row, auto"
            )
        if self.schedule_mode not in (
            "row_scan",
            "active_edge_calendar",
            "active_edge_periodic",
            "auto",
        ):
            raise ValueError(
                "schedule_mode must be one of: row_scan, active_edge_calendar, "
                "active_edge_periodic, auto"
            )
        if self.calendar_memory_limit_bytes is not None:
            if (
                isinstance(self.calendar_memory_limit_bytes, bool)
                or int(self.calendar_memory_limit_bytes) < 0
            ):
                raise ValueError(
                    "calendar_memory_limit_bytes must be a non-negative integer"
                )
            self.calendar_memory_limit_bytes = int(
                self.calendar_memory_limit_bytes
            )
        if (
            isinstance(self.warp_min_degree, bool)
            or not isinstance(self.warp_min_degree, (int, np.integer))
            or self.warp_min_degree < 1
        ):
            raise ValueError("warp_min_degree must be a positive integer")
        self.warp_min_degree = int(self.warp_min_degree)


@dataclass(slots=True)
class IBUMAPExperimentalConfig:
    total_update_clipping: Optional[ForceClippingConfig] = None
    local_exact_repulsion: LocalExactRepulsionConfig = field(
        default_factory=LocalExactRepulsionConfig
    )
    local_density_pressure: LocalDensityPressureConfig = field(
        default_factory=LocalDensityPressureConfig
    )
    kernel_subsampling: Optional[KernelSubsamplingConfig] = None
    attraction_schedule: AttractionScheduleConfig = field(
        default_factory=AttractionScheduleConfig
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def __post_init__(self) -> None:
        self.total_update_clipping = deepcopy(self.total_update_clipping)
        if self.total_update_clipping is not None:
            self.total_update_clipping.__post_init__()
        self.local_exact_repulsion = deepcopy(self.local_exact_repulsion)
        self.local_exact_repulsion.__post_init__()
        self.local_density_pressure = deepcopy(self.local_density_pressure)
        self.local_density_pressure.__post_init__()
        self.kernel_subsampling = deepcopy(self.kernel_subsampling)
        if self.kernel_subsampling is not None:
            self.kernel_subsampling.__post_init__()
        self.attraction_schedule = deepcopy(self.attraction_schedule)
        self.attraction_schedule.__post_init__()
        self.noise = deepcopy(self.noise)
        self.noise.__post_init__()


@dataclass(slots=True)
class IBUMAPConfig:
    attraction_mode: AttractionMode = "sampling"
    repulsion_mode: RepulsionMode = "true_loss"
    experimental: IBUMAPExperimentalConfig = field(
        default_factory=IBUMAPExperimentalConfig
    )

    def __post_init__(self) -> None:
        if self.attraction_mode not in ("sampling", "true_loss"):
            raise ValueError("attraction_mode must be one of: sampling, true_loss")
        if self.repulsion_mode not in ("sampling", "true_loss"):
            raise ValueError("repulsion_mode must be one of: sampling, true_loss")
        self.experimental = deepcopy(self.experimental)
        self.experimental.__post_init__()


@dataclass(slots=True)
class CPUUMAPConfig:
    update_mode: UMAPSGDUpdateMode = "asynchronous"
    rng_lifecycle: RNGLifecycle = "independent"
    deterministic: bool = False
    tqdm_kwds: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.update_mode not in ("asynchronous", "synchronous"):
            raise ValueError(
                "umap_sgd_update_mode/update_mode must be one of: "
                "asynchronous, synchronous"
            )
        if self.rng_lifecycle not in ("independent", "shared_rng"):
            raise ValueError(
                "rng_lifecycle must be one of: independent, shared_rng"
            )
        self.deterministic = bool(self.deterministic)
        self.tqdm_kwds = None if self.tqdm_kwds is None else dict(self.tqdm_kwds)


@dataclass(slots=True)
class HybridOptimizerConfig:
    """Controls the independent UMAP refinement stage of the CPU hybrid path."""

    refinement_fraction: float = 0.10
    refinement_learning_rate: float = 1.0
    refinement_negative_sample_rate: float = 5.0

    def __post_init__(self) -> None:
        self.refinement_fraction = float(self.refinement_fraction)
        self.refinement_learning_rate = float(self.refinement_learning_rate)
        self.refinement_negative_sample_rate = float(
            self.refinement_negative_sample_rate
        )
        if (
            not np.isfinite(self.refinement_fraction)
            or not 0.0 < self.refinement_fraction < 1.0
        ):
            raise ValueError("refinement_fraction must be finite and in (0, 1)")
        if (
            not np.isfinite(self.refinement_learning_rate)
            or self.refinement_learning_rate <= 0.0
        ):
            raise ValueError(
                "refinement_learning_rate must be positive and finite"
            )
        if (
            not np.isfinite(self.refinement_negative_sample_rate)
            or self.refinement_negative_sample_rate <= 0.0
        ):
            raise ValueError(
                "refinement_negative_sample_rate must be positive and finite"
            )

    def split_epochs(self, total_epochs: int) -> tuple[int, int]:
        if (
            isinstance(total_epochs, bool)
            or not isinstance(total_epochs, (int, np.integer))
            or int(total_epochs) < 2
        ):
            raise ValueError("hybrid optimization requires n_epochs >= 2")
        total_epochs = int(total_epochs)
        refinement_epochs = int(ceil(total_epochs * self.refinement_fraction))
        refinement_epochs = min(total_epochs - 1, max(1, refinement_epochs))
        return total_epochs - refinement_epochs, refinement_epochs


@dataclass(slots=True)
class DiagnosticsConfig:
    summary_path: Optional[Union[str, Path]] = None
    timing_path: Optional[Union[str, Path]] = None
    thresholds: Optional[Sequence[float]] = None
    topk_path: Optional[Union[str, Path]] = None
    topk: int = 0
    labels: Optional[Sequence[Any]] = None
    point_ids: Optional[Sequence[Any]] = None
    memory_path: Optional[Union[str, Path]] = None
    memory_epoch_stride: int = 1
    memory_sample_interval_ms: Optional[float] = None

    def __post_init__(self) -> None:
        self.summary_path = None if self.summary_path is None else str(self.summary_path)
        self.timing_path = None if self.timing_path is None else str(self.timing_path)
        self.topk_path = None if self.topk_path is None else str(self.topk_path)
        self.memory_path = None if self.memory_path is None else str(self.memory_path)
        default_thresholds = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
        values = default_thresholds if self.thresholds is None else self.thresholds
        self.thresholds = tuple(float(value) for value in values)
        self.topk = max(0, int(self.topk))
        if (
            isinstance(self.memory_epoch_stride, bool)
            or not isinstance(self.memory_epoch_stride, (int, np.integer))
            or self.memory_epoch_stride < 1
        ):
            raise ValueError("memory_epoch_stride must be a positive integer")
        self.memory_epoch_stride = int(self.memory_epoch_stride)
        if self.memory_sample_interval_ms is not None:
            if isinstance(self.memory_sample_interval_ms, bool):
                raise ValueError("memory_sample_interval_ms must be positive and finite")
            self.memory_sample_interval_ms = float(self.memory_sample_interval_ms)
            if not np.isfinite(self.memory_sample_interval_ms) or self.memory_sample_interval_ms <= 0:
                raise ValueError("memory_sample_interval_ms must be positive and finite")
        self.labels = None if self.labels is None else tuple(str(value) for value in self.labels)
        self.point_ids = (
            None if self.point_ids is None else tuple(str(value) for value in self.point_ids)
        )


@dataclass(slots=True)
class UMAPFFTConfigBundle:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    fft: FFTConfig = field(default_factory=FFTConfig)
    ibumap: IBUMAPConfig = field(default_factory=IBUMAPConfig)
    cpu_umap: CPUUMAPConfig = field(default_factory=CPUUMAPConfig)
    hybrid: HybridOptimizerConfig = field(default_factory=HybridOptimizerConfig)
    tfdp: TFDPConfig = field(default_factory=TFDPConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    legacy_numerics: NumericsConfig = field(default_factory=NumericsConfig)

    def __post_init__(self) -> None:
        for name in (
            "runtime",
            "graph",
            "umap",
            "initialization",
            "fft",
            "ibumap",
            "cpu_umap",
            "hybrid",
            "tfdp",
            "constraints",
            "diagnostics",
            "legacy_numerics",
        ):
            setattr(self, name, deepcopy(getattr(self, name)))


@dataclass(slots=True)
class UMAPFFTConfig:
    algorithm: Algorithm = "ibumap"
    device: Device = "cpu"
    attraction_mode: AttractionMode = "sampling"
    repulsion_mode: RepulsionMode = "true_loss"
    graph_backend: GraphBackend = "auto"
    workspace_policy: WorkspacePolicy = "performance"
    workspace_limit_bytes: Optional[int] = None
    deterministic: bool = False
    fft: FFTConfig = field(default_factory=FFTConfig)
    tfdp: TFDPConfig = field(default_factory=TFDPConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    hybrid: HybridOptimizerConfig = field(default_factory=HybridOptimizerConfig)
    umap_sgd_update_mode: UMAPSGDUpdateMode = "asynchronous"
    rng_lifecycle: RNGLifecycle = "independent"

    def __post_init__(self) -> None:
        if self.algorithm == "drfft_umap":
            warnings.warn(
                "algorithm='drfft_umap' is deprecated and will be replaced by "
                "algorithm='ibumap'; use 'ibumap' instead",
                FutureWarning,
                stacklevel=3,
            )
            self.algorithm = "ibumap"
        if self.algorithm not in ("umap", "ibumap", "hybrid", "tfdp"):
            raise ValueError(
                "algorithm must be one of: umap, ibumap, hybrid, tfdp "
                "(drfft_umap is a deprecated alias for ibumap)"
            )
        if self.device == "gpu":
            warnings.warn(
                "device='gpu' is deprecated and will be replaced by "
                "device='cuda'; use 'cuda' instead",
                FutureWarning,
                stacklevel=3,
            )
            self.device = "cuda"
        if self.device not in ("cpu", "cuda", "metal"):
            raise ValueError(
                "device must be one of: cpu, cuda, metal "
                "(gpu is a deprecated alias for cuda)"
            )
        if self.algorithm == "hybrid" and self.device != "cpu":
            raise NotImplementedError(
                "algorithm='hybrid' currently supports device='cpu' only"
            )
        if self.attraction_mode not in ("sampling", "true_loss"):
            raise ValueError("attraction_mode must be one of: sampling, true_loss")
        if self.repulsion_mode not in ("sampling", "true_loss"):
            raise ValueError("repulsion_mode must be one of: sampling, true_loss")
        if self.algorithm == "hybrid" and self.attraction_mode != "sampling":
            raise ValueError(
                "algorithm='hybrid' currently requires attraction_mode='sampling'"
            )
        if self.algorithm == "hybrid" and self.repulsion_mode != "true_loss":
            raise ValueError(
                "algorithm='hybrid' currently requires repulsion_mode='true_loss'"
            )
        if self.umap_sgd_update_mode not in ("asynchronous", "synchronous"):
            raise ValueError(
                "umap_sgd_update_mode must be one of: asynchronous, synchronous"
            )
        if self.rng_lifecycle not in ("independent", "shared_rng"):
            raise ValueError(
                "rng_lifecycle must be one of: independent, shared_rng"
            )
        if self.graph_backend not in ("auto", "cpu", "cuml"):
            raise ValueError("graph_backend must be one of: auto, cpu, cuml")
        aliases = {"fast": "performance", "low_memory": "minimal"}
        self.workspace_policy = aliases.get(self.workspace_policy, self.workspace_policy)
        if self.workspace_policy not in ("auto", "performance", "minimal"):
            raise ValueError(
                "workspace_policy must be one of: auto, performance, minimal "
                "(fast and low_memory are compatibility aliases)"
            )
        if self.workspace_limit_bytes is not None:
            if (
                isinstance(self.workspace_limit_bytes, bool)
                or int(self.workspace_limit_bytes) < 0
            ):
                raise ValueError("workspace_limit_bytes must be a non-negative integer")
            self.workspace_limit_bytes = int(self.workspace_limit_bytes)
        self.hybrid = deepcopy(self.hybrid)
        self.hybrid.__post_init__()
        if self.umap_sgd_update_mode == "synchronous":
            if self.algorithm != "umap":
                raise ValueError(
                    "synchronous UMAP SGD mode only applies to algorithm='umap'"
                )
            if self.device != "cpu":
                raise NotImplementedError(
                    "synchronous UMAP SGD mode is only implemented for device='cpu'"
                )
        if self.rng_lifecycle == "shared_rng":
            if self.algorithm != "umap":
                raise ValueError(
                    "rng_lifecycle='shared_rng' only applies to algorithm='umap'"
                )
            if self.device != "cpu":
                raise NotImplementedError(
                    "rng_lifecycle='shared_rng' is only implemented for device='cpu'"
                )
        if self.algorithm not in ("ibumap", "hybrid") and self.attraction_mode != "sampling":
            raise ValueError(
                "attraction_mode is only configurable for algorithm='ibumap' or 'hybrid'"
            )
        if self.algorithm not in ("ibumap", "hybrid") and self.repulsion_mode != "true_loss":
            raise ValueError(
                "repulsion_mode is only configurable for algorithm='ibumap' or 'hybrid'"
            )
        if self.device == "cuda" and self.constraint.mode == "soft":
            raise NotImplementedError("soft constraints currently support CPU only")
        if self.device == "cuda" and resolve_numeric_dtype(self.numerics.dtype) == np.dtype(np.float64):
            raise NotImplementedError(
                "dtype='float64' is currently supported on CPU only; "
                "CUDA kernels use float32"
            )
        if self.algorithm != "ibumap":
            noise_is_active = (
                self.noise.mode != "none"
                or self.noise.scale != 0.0
                or self.noise.decay != "linear"
                or self.noise.until_epoch is not None
                or self.noise.seed is not None
                or self.noise.hybrid_mode != "none"
                or self.noise.hybrid_switch_epoch is not None
            )
            if noise_is_active:
                raise ValueError(
                    "noise and hybrid modes are only configurable for "
                    "algorithm='ibumap'"
                )
