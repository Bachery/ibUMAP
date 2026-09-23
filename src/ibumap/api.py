from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from functools import wraps
from inspect import signature
from numbers import Integral
from time import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence
import warnings

import numpy as np
from scipy.optimize import curve_fit
from sklearn.utils import check_array, check_random_state

from .config import (
    AttractionScheduleConfig,
    AttractionMode,
    CPUUMAPConfig,
    ConstraintConfig,
    DegreeDampingConfig,
    Device,
    DiagnosticsConfig,
    FFTConfig,
    ForceClippingConfig,
    GraphConfig,
    GraphBackend,
    HybridOptimizerConfig,
    IBUMAPConfig,
    IBUMAPExperimentalConfig,
    InitializationConfig,
    KernelSubsamplingConfig,
    LocalDensityPressureConfig,
    LocalExactRepulsionConfig,
    NoiseConfig,
    NumericsConfig,
    RNGLifecycle,
    RuntimeConfig,
    TFDPConfig,
    UMAPConfig,
    UMAPSGDUpdateMode,
    EffectiveConfig,
    ConfigBundle,
    _UnsetType as _ConfigUnsetType,
    resolve_numeric_dtype,
)
from .runtime import (
    ensure_cupy_runtime,
    ensure_gpu_runtime,
    ensure_metal_runtime,
    resolve_graph_backend,
    validate_metal_config,
)
from .runtime.memory import StageMemoryRecorder
from .utils import normalize_rng_state

if TYPE_CHECKING:
    from .prepared import PreparedInputs


def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    def _curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape[0], dtype=np.float64)
    mask = xv < min_dist
    yv[mask] = 1.0
    yv[~mask] = np.exp(-(xv[~mask] - min_dist) / spread)

    params, _ = curve_fit(_curve, xv, yv)
    return float(params[0]), float(params[1])


def _validate_clip_epoch_range(
    value: Optional[Sequence[int]],
    name: str,
    n_epochs: Optional[int],
) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a length-two sequence of integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a length-two sequence of integers") from exc
    if len(values) != 2 or any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise ValueError(f"{name} must be a length-two sequence of integers")
    start, end = (int(values[0]), int(values[1]))
    if start < 0 or start >= end:
        raise ValueError(f"{name} must satisfy 0 <= start < end")
    if n_epochs is not None and end > int(n_epochs):
        raise ValueError(f"{name} end must not exceed n_epochs={int(n_epochs)}")
    return start, end


def _validate_init_shape(init: Any, n_samples: int, n_components: int) -> None:
    if isinstance(init, str):
        if init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral', 'random', or a 2D array")
        return
    array = np.asarray(init)
    expected = (int(n_samples), int(n_components))
    if array.shape != expected:
        raise ValueError(f"init shape must be {expected}, got {array.shape}")


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _UnsetType()


def _track_constructor_arguments(func):
    """Record which arguments the caller supplied without changing the signature."""
    func_signature = signature(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = func_signature.bind_partial(self, *args, **kwargs)
        self._provided_constructor_parameters = set(bound.arguments) - {"self"}
        return func(self, *args, **kwargs)

    return wrapper


def _values_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) or is_dataclass(right):
        if type(left) is not type(right):
            return False
        return all(
            _values_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except Exception:
            return False
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def _merge_flat_field(
    config: Any,
    field_name: str,
    flat_name: str,
    flat_values: Dict[str, Any],
    provided: set[str],
    *,
    config_was_supplied: bool,
    canonical_name: str,
    normalize=lambda value: value,
) -> None:
    flat_value = flat_values[flat_name] if flat_name in provided else _UNSET
    if flat_value is _UNSET:
        return
    normalized = normalize(flat_value)
    current = getattr(config, field_name)
    if config_was_supplied and not _values_equal(current, normalized):
        raise ValueError(
            f"Conflicting values for flat '{flat_name}' ({flat_value!r}) and "
            f"'{canonical_name}' ({current!r})"
        )
    setattr(config, field_name, deepcopy(normalized))


def _merge_legacy_container(
    canonical: Any,
    legacy: Any,
    *,
    canonical_was_supplied: bool,
    legacy_was_supplied: bool,
    canonical_name: str,
    legacy_name: str,
) -> Any:
    if not legacy_was_supplied or legacy is None:
        return deepcopy(canonical)
    if canonical_was_supplied and not _values_equal(canonical, legacy):
        raise ValueError(
            f"Conflicting values for legacy '{legacy_name}' and '{canonical_name}'"
        )
    return deepcopy(legacy)


def _resolve_degree_enabled(config: ConfigBundle) -> bool:
    enabled = config.umap.degree_damping.enabled
    return (
        config.runtime.algorithm in ("ibumap", "hybrid")
        if enabled is None
        else bool(enabled)
    )


def _validate_effective_config(config: ConfigBundle) -> None:
    algorithm = config.runtime.algorithm
    device = config.runtime.device
    numeric_dtype = resolve_numeric_dtype(config.legacy_numerics.dtype)
    if algorithm in ("ibumap", "hybrid") and config.umap.n_components != 2:
        raise ValueError(
            f"algorithm={algorithm!r} currently requires n_components=2"
        )
    if algorithm == "hybrid":
        if device != "cpu":
            raise NotImplementedError(
                "algorithm='hybrid' currently supports device='cpu' only"
            )
        if config.umap.n_epochs is not None and config.umap.n_epochs < 2:
            raise ValueError("hybrid optimization requires n_epochs >= 2")
    if (
        device == "cpu"
        and algorithm in ("umap", "hybrid")
        and numeric_dtype != np.dtype(np.float32)
    ):
        raise NotImplementedError(
            "algorithm='umap' and the UMAP refinement in algorithm='hybrid' "
            "currently require dtype='float32'; their shared distance kernel "
            "has a float32 signature"
        )
    if device == "cuda" and numeric_dtype == np.dtype(np.float64):
        raise NotImplementedError(
            "dtype='float64' is currently supported on CPU only; CUDA kernels use float32"
        )
    if device == "metal":
        validate_metal_config(config)
    if config.cpu_umap.update_mode == "synchronous":
        if algorithm != "umap":
            raise ValueError("synchronous UMAP SGD mode only applies to algorithm='umap'")
        if device != "cpu":
            raise NotImplementedError(
                "synchronous UMAP SGD mode is only implemented for device='cpu'"
            )
    if config.cpu_umap.rng_lifecycle == "shared_rng":
        if algorithm != "umap":
            raise ValueError(
                "rng_lifecycle='shared_rng' only applies to algorithm='umap'"
            )
        if device != "cpu":
            raise NotImplementedError(
                "rng_lifecycle='shared_rng' is only implemented for device='cpu'"
            )
    if config.cpu_umap.deterministic and config.fft.p2m_mode in (
        "atomic",
        "block_atomic",
    ):
        raise ValueError(
            "deterministic=True is incompatible with p2m_mode="
            f"{config.fft.p2m_mode!r}; use 'auto', 'segmented', or 'serial'"
        )
    if algorithm not in ("ibumap", "hybrid") and config.ibumap.attraction_mode != "sampling":
        raise ValueError(
            "attraction_mode is only configurable for algorithm='ibumap' or 'hybrid'"
        )
    if algorithm not in ("ibumap", "hybrid") and config.ibumap.repulsion_mode != "true_loss":
        raise ValueError(
            "repulsion_mode is only configurable for algorithm='ibumap' or 'hybrid'"
        )
    if algorithm == "hybrid":
        if config.ibumap.attraction_mode != "sampling":
            raise ValueError(
                "algorithm='hybrid' currently requires attraction_mode='sampling'"
            )
        if config.ibumap.repulsion_mode != "true_loss":
            raise ValueError(
                "algorithm='hybrid' currently requires repulsion_mode='true_loss'"
            )

    noise = config.ibumap.experimental.noise
    noise_is_active = (
        noise.mode != "none"
        or noise.scale != 0.0
        or noise.decay != "linear"
        or noise.until_epoch is not None
        or noise.seed is not None
        or noise.hybrid_mode != "none"
        or noise.hybrid_switch_epoch is not None
    )
    if algorithm != "ibumap" and noise_is_active:
        raise ValueError(
            "noise and hybrid modes are only configurable for algorithm='ibumap'"
        )

    if algorithm == "hybrid":
        default_experimental = IBUMAPExperimentalConfig()
        if not _values_equal(config.ibumap.experimental, default_experimental):
            raise ValueError(
                "algorithm='hybrid' does not yet support non-default ibUMAP "
                "experimental mechanisms"
            )
        default_degree = DegreeDampingConfig()
        degree = config.umap.degree_damping
        degree_matches_default = (
            degree.enabled in (None, True)
            and degree.mode == default_degree.mode
            and degree.ref == default_degree.ref
            and degree.ref_value == default_degree.ref_value
            and degree.power == default_degree.power
            and degree.min_scale == default_degree.min_scale
        )
        if not degree_matches_default:
            raise ValueError(
                "algorithm='hybrid' currently requires default attraction degree damping"
            )
        if not _values_equal(
            config.umap.repulsion_clipping,
            UMAPConfig().repulsion_clipping,
        ):
            raise ValueError(
                "algorithm='hybrid' currently requires default repulsion clipping"
            )
        if not _values_equal(config.constraints, ConstraintConfig()):
            raise ValueError(
                "algorithm='hybrid' does not yet support embedding constraints"
            )
        if config.fft.combine_stages or config.fft.interpolation_schedule is not None:
            raise ValueError(
                "algorithm='hybrid' does not yet support FFT stage schedules"
            )

    damping_requested = config.umap.degree_damping.enabled
    if damping_requested is True and algorithm not in ("ibumap", "hybrid"):
        if algorithm == "umap" and config.cpu_umap.update_mode == "synchronous":
            raise NotImplementedError(
                "degree damping is not yet implemented for synchronous CPU UMAP"
            )
        raise ValueError("attraction degree damping requires algorithm='ibumap'")

    experimental = config.ibumap.experimental
    local_enabled = (
        experimental.local_exact_repulsion.enabled
        or experimental.local_density_pressure.enabled
    )
    if local_enabled and algorithm != "ibumap":
        raise ValueError("local repulsion mechanisms require algorithm='ibumap'")
    if local_enabled and device == "cuda":
        raise NotImplementedError(
            "local_exact_repulsion and local_density_pressure currently support CPU only"
        )
    if experimental.kernel_subsampling is not None and device == "cuda":
        raise NotImplementedError("ibFFT kernel sub-sampling currently supports CPU only")
    if device == "cuda" and config.constraints.mode == "soft":
        raise NotImplementedError("soft constraints currently support CPU only")

    n_epochs = config.umap.n_epochs
    for name, clipping in (
        ("repulsion_clip_epoch_range", config.umap.repulsion_clipping),
        (
            "total_update_clip_epoch_range",
            config.ibumap.experimental.total_update_clipping,
        ),
    ):
        if clipping is not None and clipping.epoch_range is not None and n_epochs is not None:
            if clipping.epoch_range[1] > n_epochs:
                raise ValueError(f"{name} end must not exceed n_epochs={n_epochs}")
    local = experimental.local_exact_repulsion
    if local.end_epoch is not None and n_epochs is not None and local.end_epoch > n_epochs:
        raise ValueError(f"local_exact_end_epoch must not exceed n_epochs={n_epochs}")


def _build_effective_config(
    flat_values: Dict[str, Any],
    provided: set[str],
    *,
    runtime: Optional[RuntimeConfig],
    graph: Optional[GraphConfig],
    umap: Optional[UMAPConfig],
    initialization: Optional[InitializationConfig],
    fft: Optional[FFTConfig],
    ibumap: Optional[IBUMAPConfig],
    cpu_umap: Optional[CPUUMAPConfig],
    hybrid: Optional[HybridOptimizerConfig],
    tfdp: Optional[TFDPConfig],
    constraints: Optional[ConstraintConfig],
    diagnostics: Optional[DiagnosticsConfig],
) -> tuple[ConfigBundle, list[str]]:
    supplied = {
        name: name in provided and value is not None
        for name, value in {
            "runtime": runtime,
            "graph": graph,
            "umap": umap,
            "initialization": initialization,
            "fft": fft,
            "ibumap": ibumap,
            "cpu_umap": cpu_umap,
            "hybrid": hybrid,
            "tfdp": tfdp,
            "constraints": constraints,
            "diagnostics": diagnostics,
        }.items()
    }
    runtime_cfg = deepcopy(runtime) if runtime is not None else RuntimeConfig()
    graph_cfg = deepcopy(graph) if graph is not None else GraphConfig()
    umap_cfg = deepcopy(umap) if umap is not None else UMAPConfig()
    init_cfg = deepcopy(initialization) if initialization is not None else InitializationConfig()
    fft_cfg = deepcopy(fft) if fft is not None else FFTConfig()
    ibumap_cfg = deepcopy(ibumap) if ibumap is not None else IBUMAPConfig()
    cpu_cfg = deepcopy(cpu_umap) if cpu_umap is not None else CPUUMAPConfig()
    hybrid_cfg = (
        deepcopy(hybrid) if hybrid is not None else HybridOptimizerConfig()
    )
    tfdp_cfg = deepcopy(tfdp) if tfdp is not None else TFDPConfig()
    constraint_cfg = deepcopy(constraints) if constraints is not None else ConstraintConfig()
    diagnostics_cfg = deepcopy(diagnostics) if diagnostics is not None else DiagnosticsConfig()

    _merge_flat_field(runtime_cfg, "algorithm", "algorithm", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.algorithm")
    _merge_flat_field(runtime_cfg, "device", "device", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.device")
    _merge_flat_field(runtime_cfg, "graph_backend", "graph_backend", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.graph_backend")
    _merge_flat_field(runtime_cfg, "random_state", "random_state", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.random_state")
    _merge_flat_field(runtime_cfg, "workspace_policy", "workspace_policy", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.workspace_policy")
    _merge_flat_field(runtime_cfg, "workspace_limit_bytes", "workspace_limit_bytes", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.workspace_limit_bytes", normalize=lambda value: None if value is None else int(value))
    _merge_flat_field(runtime_cfg, "verbose", "verbose", flat_values, provided, config_was_supplied=supplied["runtime"], canonical_name="runtime.verbose", normalize=bool)
    runtime_cfg.__post_init__()

    graph_fields = (
        ("n_neighbors", "n_neighbors", int),
        ("metric", "metric", str),
        ("metric_kwds", "metric_kwds", lambda value: dict(value or {})),
        ("n_jobs", "n_jobs", int),
        ("angular_rp_forest", "angular_rp_forest", bool),
        ("low_memory", "low_memory", bool),
        ("set_op_mix_ratio", "set_op_mix_ratio", float),
        ("local_connectivity", "local_connectivity", float),
    )
    for field_name, flat_name, normalize in graph_fields:
        _merge_flat_field(graph_cfg, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["graph"], canonical_name=f"graph.{field_name}", normalize=normalize)
    graph_cfg.__post_init__()

    umap_fields = (
        ("n_components", "n_components", int),
        ("n_epochs", "n_epochs", lambda value: None if value is None else int(value)),
        ("min_dist", "min_dist", float),
        ("spread", "spread", float),
        ("repulsion_strength", "repulsion_strength", float),
        ("negative_sample_rate", "negative_sample_rate", float),
        ("learning_rate", "learning_rate", float),
    )
    for field_name, flat_name, normalize in umap_fields:
        _merge_flat_field(umap_cfg, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["umap"], canonical_name=f"umap.{field_name}", normalize=normalize)

    hybrid_fields = (
        ("refinement_fraction", "refinement_fraction", float),
        ("refinement_learning_rate", "refinement_learning_rate", float),
        (
            "refinement_negative_sample_rate",
            "refinement_negative_sample_rate",
            float,
        ),
    )
    for field_name, flat_name, normalize in hybrid_fields:
        _merge_flat_field(
            hybrid_cfg,
            field_name,
            flat_name,
            flat_values,
            provided,
            config_was_supplied=supplied["hybrid"],
            canonical_name=f"hybrid.{field_name}",
            normalize=normalize,
        )
    hybrid_cfg.__post_init__()

    legacy_init = getattr(umap_cfg, "_legacy_init", None)
    legacy_init_was_supplied = not isinstance(legacy_init, _ConfigUnsetType)
    if legacy_init_was_supplied:
        if supplied["initialization"] and not _values_equal(init_cfg.init, legacy_init):
            raise ValueError("Conflicting values for 'umap.init' and 'initialization.init'")
        init_cfg.init = deepcopy(legacy_init)
    init_fields = (
        ("init", "init", lambda value: value),
        ("spectral_method", "spectral_method", lambda value: None if value is None else str(value)),
        ("spectral_tol", "spectral_tol", lambda value: None if value is None else float(value)),
        ("spectral_maxiter", "spectral_maxiter", lambda value: None if value is None else int(value)),
        ("spectral_ncv", "spectral_ncv", lambda value: None if value is None else int(value)),
        ("spectral_auto_defaults", "spectral_auto_defaults", lambda value: None if value is None else bool(value)),
        ("spectral_scale_policy", "spectral_scale_policy", str),
        ("spectral_max_span", "spectral_max_span", float),
        ("spectral_jitter_relative", "spectral_jitter_relative", float),
        ("spectral_jitter_max", "spectral_jitter_max", float),
        ("report_duplicate_ratio", "report_duplicate_ratio", bool),
    )
    for field_name, flat_name, normalize in init_fields:
        _merge_flat_field(
            init_cfg,
            field_name,
            flat_name,
            flat_values,
            provided,
            config_was_supplied=supplied["initialization"] or legacy_init_was_supplied,
            canonical_name=f"initialization.{field_name}",
            normalize=normalize,
        )
    init_cfg.__post_init__()

    legacy_numerics = flat_values["numerics_config"] if "numerics_config" in provided and flat_values["numerics_config"] is not None else NumericsConfig()
    legacy_numerics = deepcopy(legacy_numerics)
    epsilon_sources: list[tuple[str, float]] = []
    if "numerics_config" in provided and flat_values["numerics_config"] is not None:
        epsilon_sources.append(("numerics_config.epsilon", float(legacy_numerics.epsilon)))
    if "umap_epsilon" in provided and flat_values["umap_epsilon"] is not None:
        epsilon_sources.append(("umap_epsilon", float(flat_values["umap_epsilon"])))
    for source_name, epsilon in epsilon_sources:
        if supplied["umap"] and not _values_equal(umap_cfg.epsilon, epsilon):
            raise ValueError(
                f"Conflicting values for '{source_name}' ({epsilon!r}) and "
                f"'umap.epsilon' ({umap_cfg.epsilon!r})"
            )
        if epsilon_sources and not _values_equal(epsilon_sources[0][1], epsilon):
            raise ValueError("Conflicting epsilon values in legacy inputs")
        umap_cfg.epsilon = epsilon

    degree = deepcopy(umap_cfg.degree_damping)
    degree_fields = (
        ("enabled", "attraction_degree_damping", lambda value: None if value is None else bool(value)),
        ("mode", "attraction_degree_damping_mode", str),
        ("ref", "attraction_degree_damping_ref", str),
        ("ref_value", "attraction_degree_damping_ref_value", lambda value: None if value is None else float(value)),
        ("power", "attraction_degree_damping_power", float),
        ("min_scale", "attraction_degree_damping_min_scale", lambda value: None if value is None else float(value)),
    )
    for field_name, flat_name, normalize in degree_fields:
        _merge_flat_field(degree, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["umap"], canonical_name=f"umap.degree_damping.{field_name}", normalize=normalize)
    degree.__post_init__()
    umap_cfg.degree_damping = degree

    repulsion_flat_present = any(name in provided for name in ("repulsion_clip_norm", "repulsion_clip_with_alpha", "repulsion_clip_epoch_range"))
    if repulsion_flat_present:
        flat_epoch_range = (
            flat_values["repulsion_clip_epoch_range"]
            if "repulsion_clip_epoch_range" in provided
            else (
                None
                if umap_cfg.repulsion_clipping is None
                else umap_cfg.repulsion_clipping.epoch_range
            )
        )
        flat_epoch_range = _validate_clip_epoch_range(
            flat_epoch_range,
            "repulsion_clip_epoch_range",
            umap_cfg.n_epochs,
        )
        norm = flat_values["repulsion_clip_norm"] if "repulsion_clip_norm" in provided else (None if umap_cfg.repulsion_clipping is None else umap_cfg.repulsion_clipping.norm)
        clipping = None if norm is None else ForceClippingConfig(
            norm=float(norm),
            with_alpha=bool(flat_values["repulsion_clip_with_alpha"] if "repulsion_clip_with_alpha" in provided else (True if umap_cfg.repulsion_clipping is None else umap_cfg.repulsion_clipping.with_alpha)),
            epoch_range=flat_epoch_range,
        )
        if supplied["umap"] and not _values_equal(umap_cfg.repulsion_clipping, clipping):
            raise ValueError("Conflicting flat repulsion clipping and 'umap.repulsion_clipping'")
        umap_cfg.repulsion_clipping = clipping
    umap_cfg.__post_init__()

    fft_cfg = _merge_legacy_container(
        fft_cfg,
        flat_values["fft_config"],
        canonical_was_supplied=supplied["fft"],
        legacy_was_supplied="fft_config" in provided,
        canonical_name="fft",
        legacy_name="fft_config",
    )
    _merge_flat_field(fft_cfg, "kernel_clip", "ibfft_kernel_clip", flat_values, provided, config_was_supplied=supplied["fft"], canonical_name="fft.kernel_clip", normalize=float)
    _merge_flat_field(fft_cfg, "p2m_mode", "p2m_mode", flat_values, provided, config_was_supplied=supplied["fft"], canonical_name="fft.p2m_mode", normalize=str)
    _merge_flat_field(fft_cfg, "kernel_cache_policy", "fft_kernel_cache_policy", flat_values, provided, config_was_supplied=supplied["fft"], canonical_name="fft.kernel_cache_policy", normalize=str)
    _merge_flat_field(fft_cfg, "kernel_cache_limit_bytes", "fft_kernel_cache_limit_bytes", flat_values, provided, config_was_supplied=supplied["fft"], canonical_name="fft.kernel_cache_limit_bytes", normalize=lambda value: None if value is None else int(value))
    _merge_flat_field(fft_cfg, "kernel_cache_max_entries", "fft_kernel_cache_max_entries", flat_values, provided, config_was_supplied=supplied["fft"], canonical_name="fft.kernel_cache_max_entries", normalize=lambda value: None if value is None else int(value))
    fft_cfg.__post_init__()

    _merge_flat_field(ibumap_cfg, "attraction_mode", "attraction_mode", flat_values, provided, config_was_supplied=supplied["ibumap"], canonical_name="ibumap.attraction_mode")
    _merge_flat_field(ibumap_cfg, "repulsion_mode", "repulsion_mode", flat_values, provided, config_was_supplied=supplied["ibumap"], canonical_name="ibumap.repulsion_mode")
    experimental = deepcopy(ibumap_cfg.experimental)

    noise = deepcopy(experimental.noise)
    old_noise = flat_values["noise_config"] if "noise_config" in provided else None
    if old_noise is not None:
        if supplied["ibumap"] and not _values_equal(noise, old_noise):
            raise ValueError("Conflicting values for 'noise_config' and 'ibumap.experimental.noise'")
        noise = deepcopy(old_noise)
    noise_fields = (
        ("mode", "noise_mode", str),
        ("scale", "noise_scale", float),
        ("decay", "noise_decay", str),
        ("until_epoch", "noise_until_epoch", lambda value: None if value is None else int(value)),
        ("seed", "noise_seed", lambda value: None if value is None else int(value)),
        ("hybrid_mode", "hybrid_mode", str),
        ("hybrid_switch_epoch", "hybrid_switch_epoch", lambda value: None if value is None else int(value)),
    )
    for field_name, flat_name, normalize in noise_fields:
        _merge_flat_field(noise, field_name, flat_name, flat_values, provided, config_was_supplied=(supplied["ibumap"] or old_noise is not None), canonical_name=f"ibumap.experimental.noise.{field_name}", normalize=normalize)
    noise.__post_init__()
    experimental.noise = noise

    local = deepcopy(experimental.local_exact_repulsion)
    local_fields = (
        ("enabled", "local_exact_repulsion", bool), ("k", "local_exact_k", lambda value: value),
        ("radius_factor", "local_exact_radius_factor", float), ("weight", "local_exact_weight", float),
        ("every", "local_exact_every", lambda value: value), ("clip", "local_exact_clip", float),
        ("symmetric", "local_exact_symmetric", bool),
        ("start_epoch", "local_exact_start_epoch", lambda value: value),
        ("end_epoch", "local_exact_end_epoch", lambda value: value),
        ("start_frac", "local_exact_start_frac", lambda value: None if value is None else float(value)),
        ("end_frac", "local_exact_end_frac", lambda value: None if value is None else float(value)),
        ("density_filter", "local_exact_density_filter", bool),
        ("min_cell_count", "local_exact_min_cell_count", lambda value: value),
        ("min_cell_count_quantile", "local_exact_min_cell_count_quantile", lambda value: None if value is None else float(value)),
        ("density_include_neighbor_cells", "local_exact_density_include_neighbor_cells", bool),
        ("timing_sample_size", "local_exact_timing_sample_size", lambda value: value),
    )
    for field_name, flat_name, normalize in local_fields:
        _merge_flat_field(local, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["ibumap"], canonical_name=f"ibumap.experimental.local_exact_repulsion.{field_name}", normalize=normalize)
    local.__post_init__()
    experimental.local_exact_repulsion = local

    density = deepcopy(experimental.local_density_pressure)
    density_fields = (
        ("enabled", "local_density_pressure", bool), ("weight", "local_density_pressure_weight", float),
        ("every", "local_density_pressure_every", lambda value: value), ("min_count", "local_density_pressure_min_count", lambda value: value),
        ("clip", "local_density_pressure_clip", float), ("power", "local_density_pressure_power", float),
    )
    for field_name, flat_name, normalize in density_fields:
        _merge_flat_field(density, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["ibumap"], canonical_name=f"ibumap.experimental.local_density_pressure.{field_name}", normalize=normalize)
    density.__post_init__()
    experimental.local_density_pressure = density

    attraction_schedule = deepcopy(experimental.attraction_schedule)
    attraction_schedule_fields = (
        ("kernel_mode", "attraction_kernel_mode", str),
        ("schedule_mode", "attraction_schedule_mode", str),
        (
            "calendar_memory_limit_bytes",
            "attraction_calendar_memory_limit_bytes",
            lambda value: None if value is None else int(value),
        ),
        ("warp_min_degree", "attraction_warp_min_degree", lambda value: value),
    )
    for field_name, flat_name, normalize in attraction_schedule_fields:
        _merge_flat_field(
            attraction_schedule,
            field_name,
            flat_name,
            flat_values,
            provided,
            config_was_supplied=supplied["ibumap"],
            canonical_name=f"ibumap.experimental.attraction_schedule.{field_name}",
            normalize=normalize,
        )
    attraction_schedule.__post_init__()
    experimental.attraction_schedule = attraction_schedule

    subsample_present = any(name in provided for name in ("ibfft_kernel_subsample_mode", "ibfft_kernel_subsample_radius_cells", "ibfft_kernel_subsample_points"))
    if subsample_present:
        mode = flat_values["ibfft_kernel_subsample_mode"] if "ibfft_kernel_subsample_mode" in provided else (None if experimental.kernel_subsampling is None else experimental.kernel_subsampling.mode)
        validated_subsampling = KernelSubsamplingConfig(
            mode="near_origin" if mode is None else mode,
            radius_cells=flat_values["ibfft_kernel_subsample_radius_cells"] if "ibfft_kernel_subsample_radius_cells" in provided else (2 if experimental.kernel_subsampling is None else experimental.kernel_subsampling.radius_cells),
            points=flat_values["ibfft_kernel_subsample_points"] if "ibfft_kernel_subsample_points" in provided else (4 if experimental.kernel_subsampling is None else experimental.kernel_subsampling.points),
        )
        subsampling = None if mode is None else validated_subsampling
        if supplied["ibumap"] and not _values_equal(experimental.kernel_subsampling, subsampling):
            raise ValueError("Conflicting flat kernel subsampling and canonical config")
        experimental.kernel_subsampling = subsampling

    total_present = any(name in provided for name in ("total_update_clip_norm", "total_update_clip_with_alpha", "total_update_clip_epoch_range"))
    if total_present:
        flat_epoch_range = (
            flat_values["total_update_clip_epoch_range"]
            if "total_update_clip_epoch_range" in provided
            else (
                None
                if experimental.total_update_clipping is None
                else experimental.total_update_clipping.epoch_range
            )
        )
        flat_epoch_range = _validate_clip_epoch_range(
            flat_epoch_range,
            "total_update_clip_epoch_range",
            umap_cfg.n_epochs,
        )
        norm = flat_values["total_update_clip_norm"] if "total_update_clip_norm" in provided else (None if experimental.total_update_clipping is None else experimental.total_update_clipping.norm)
        total = None if norm is None else ForceClippingConfig(
            norm=float(norm),
            with_alpha=bool(flat_values["total_update_clip_with_alpha"] if "total_update_clip_with_alpha" in provided else (False if experimental.total_update_clipping is None else experimental.total_update_clipping.with_alpha)),
            epoch_range=flat_epoch_range,
        )
        if supplied["ibumap"] and not _values_equal(experimental.total_update_clipping, total):
            raise ValueError("Conflicting flat total-update clipping and canonical config")
        experimental.total_update_clipping = total
    experimental.__post_init__()
    ibumap_cfg.experimental = experimental
    ibumap_cfg.__post_init__()

    _merge_flat_field(cpu_cfg, "update_mode", "umap_sgd_update_mode", flat_values, provided, config_was_supplied=supplied["cpu_umap"], canonical_name="cpu_umap.update_mode")
    _merge_flat_field(cpu_cfg, "rng_lifecycle", "rng_lifecycle", flat_values, provided, config_was_supplied=supplied["cpu_umap"], canonical_name="cpu_umap.rng_lifecycle")
    _merge_flat_field(cpu_cfg, "deterministic", "deterministic", flat_values, provided, config_was_supplied=supplied["cpu_umap"], canonical_name="cpu_umap.deterministic", normalize=bool)
    _merge_flat_field(cpu_cfg, "tqdm_kwds", "tqdm_kwds", flat_values, provided, config_was_supplied=supplied["cpu_umap"], canonical_name="cpu_umap.tqdm_kwds", normalize=lambda value: None if value is None else dict(value))
    cpu_cfg.__post_init__()

    tfdp_cfg = _merge_legacy_container(tfdp_cfg, flat_values["tfdp_config"], canonical_was_supplied=supplied["tfdp"], legacy_was_supplied="tfdp_config" in provided, canonical_name="tfdp", legacy_name="tfdp_config")
    constraint_cfg = _merge_legacy_container(constraint_cfg, flat_values["constraint_config"], canonical_was_supplied=supplied["constraints"], legacy_was_supplied="constraint_config" in provided, canonical_name="constraints", legacy_name="constraint_config")

    diagnostics_fields = (
        ("summary_path", "diagnostics_path"), ("timing_path", "diagnostics_timing_path"),
        ("thresholds", "diagnostics_thresholds"), ("topk_path", "diagnostics_topk_path"),
        ("topk", "diagnostics_topk"), ("labels", "diagnostics_labels"),
        ("point_ids", "diagnostics_point_ids"),
        ("memory_path", "diagnostics_memory_path"),
        ("memory_epoch_stride", "diagnostics_memory_epoch_stride"),
        ("memory_sample_interval_ms", "diagnostics_memory_sample_interval_ms"),
    )
    for field_name, flat_name in diagnostics_fields:
        _merge_flat_field(diagnostics_cfg, field_name, flat_name, flat_values, provided, config_was_supplied=supplied["diagnostics"], canonical_name=f"diagnostics.{field_name}")
    diagnostics_cfg.__post_init__()

    bundle = ConfigBundle(
        runtime=runtime_cfg, graph=graph_cfg, umap=umap_cfg,
        initialization=init_cfg, fft=fft_cfg,
        ibumap=ibumap_cfg, cpu_umap=cpu_cfg, hybrid=hybrid_cfg, tfdp=tfdp_cfg,
        constraints=constraint_cfg, diagnostics=diagnostics_cfg,
        legacy_numerics=legacy_numerics,
    )
    _validate_effective_config(bundle)
    legacy_used = sorted(
        name
        for name in provided
        if name not in {
            "runtime", "graph", "umap", "initialization", "fft", "ibumap",
            "cpu_umap", "hybrid", "tfdp", "constraints", "diagnostics",
        }
    )
    return bundle, legacy_used


class _NestedConfigAttribute:
    def __init__(self, *path: str):
        self.path = path

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.config
        for part in self.path:
            value = getattr(value, part)
        return value

    def __set__(self, instance, value) -> None:
        if getattr(instance, "_projecting_legacy_values", False):
            return
        candidate = deepcopy(instance.config)
        target = candidate
        for part in self.path[:-1]:
            target = getattr(target, part)
        setattr(target, self.path[-1], deepcopy(value))
        validator = getattr(target, "__post_init__", None)
        if validator is not None:
            validator()
        _validate_effective_config(candidate)
        instance.config = candidate
        if self.path[0] in {
            "runtime",
            "umap",
            "initialization",
            "fft",
            "tfdp",
            "constraints",
            "cpu_umap",
            "hybrid",
        }:
            instance._refresh_runtime_view()


class IBUMAP:
    @_track_constructor_arguments
    def __init__(
        self,
        algorithm: str = "ibumap",
        device: str = "cpu",
        attraction_mode: str = "sampling",
        repulsion_mode: str = "true_loss",
        n_components: int = 2,
        n_neighbors: int = 15,
        n_epochs: Optional[int] = None,
        init: Any = "spectral",
        spectral_method: Optional[str] = None,
        spectral_tol: Optional[float] = None,
        spectral_maxiter: Optional[int] = None,
        spectral_ncv: Optional[int] = None,
        spectral_auto_defaults: Optional[bool] = None,
        spectral_scale_policy: str = "auto",
        spectral_max_span: float = 0.1,
        spectral_jitter_relative: float = 1e-4,
        spectral_jitter_max: float = 1e-4,
        report_duplicate_ratio: bool = False,
        min_dist: float = 0.2,
        spread: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: float = 5.0,
        learning_rate: float = 1.0,
        refinement_fraction: float = 0.10,
        refinement_learning_rate: float = 1.0,
        refinement_negative_sample_rate: float = 5.0,
        random_state: Optional[int] = None,
        metric: str = "euclidean",
        metric_kwds: Optional[Dict[str, Any]] = None,
        graph_backend: str = "auto",
        workspace_policy: str = "auto",
        workspace_limit_bytes: Optional[int] = None,
        p2m_mode: str = "auto",
        fft_kernel_cache_policy: Optional[str] = None,
        fft_kernel_cache_limit_bytes: Optional[int] = None,
        fft_kernel_cache_max_entries: Optional[int] = None,
        deterministic: bool = False,
        # Advanced configurations
        fft_config: Optional[FFTConfig] = None,
        tfdp_config: Optional[TFDPConfig] = None,
        constraint_config: Optional[ConstraintConfig] = None,
        numerics_config: Optional[NumericsConfig] = None,
        # Repulsion clipping inside ibFFT kernel
        ibfft_kernel_clip: float = 4.0,  # compatibility alias for fft.kernel_clip
        umap_epsilon: Optional[float] = None,  # compatibility alias for umap.epsilon
        # 改进 near-origin kernel 离散化，基本无效，不建议继续推进
        ibfft_kernel_subsample_mode: Optional[str] = None,
        ibfft_kernel_subsample_radius_cells: int = 2,
        ibfft_kernel_subsample_points: int = 4,
        # umap-learn hyperparameters
        n_jobs: int = -1,
        angular_rp_forest: bool = False,
        low_memory: bool = True,
        verbose: bool = False,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        tqdm_kwds: Optional[Dict[str, Any]] = None,
        noise_mode: str = "none",
        noise_scale: float = 0.0,
        noise_decay: str = "linear",
        noise_until_epoch: Optional[int] = None,
        noise_seed: Optional[int] = None,
        hybrid_mode: str = "none",
        hybrid_switch_epoch: Optional[int] = None,
        noise_config: Optional[NoiseConfig] = None,
        # Synchronous ibUMAP safeguard for attraction accumulated by graph hubs.
        # Defaults to enabled for ibumap and disabled for other algorithms.
        attraction_degree_damping: Optional[bool] = None,
        attraction_degree_damping_mode: str = "weighted_degree",
        attraction_degree_damping_ref: str = "p99",
        attraction_degree_damping_ref_value: Optional[float] = None,
        attraction_degree_damping_power: float = 0.5,
        attraction_degree_damping_min_scale: Optional[float] = None,
        # Clipping for combined repulsion forces each point receives
        # 防止 ibFFT 最终斥力长尾导致 outlier / bbox 爆炸
        repulsion_clip_norm: Optional[float] = 4.0,
        repulsion_clip_with_alpha: bool = True,
        repulsion_clip_epoch_range: Optional[Sequence[int]] = None,
        # Diagnostics tools for analyzing the distribution of forces and updates during optimization
        # 定位 force/update 异常来源，仅保留为调试工具
        diagnostics_path: Optional[str] = None,
        diagnostics_timing_path: Optional[str] = None,
        diagnostics_thresholds: Optional[Sequence[float]] = None,
        diagnostics_topk_path: Optional[str] = None,
        diagnostics_topk: int = 0,
        diagnostics_labels: Optional[Sequence[Any]] = None,
        diagnostics_point_ids: Optional[Sequence[Any]] = None,
        diagnostics_memory_path: Optional[str] = None,
        diagnostics_memory_epoch_stride: int = 1,
        diagnostics_memory_sample_interval_ms: Optional[float] = None,
        # Clipping for the combined update (attraction + repulsion) each point receives
        # 限制最终 attr + repl 总位移，会加重 line-collapse
        total_update_clip_norm: Optional[float] = None,
        total_update_clip_with_alpha: bool = False,
        total_update_clip_epoch_range: Optional[Sequence[int]] = None,
        # Extra local repulsion mechanism: local exact near-field repulsion term
        # 补充近场精确局部斥力，效果不错，但太慢
        local_exact_repulsion: bool = False,
        local_exact_k: int = 8,
        local_exact_radius_factor: float = 0.5,
        local_exact_weight: float = 0.1,
        local_exact_every: int = 1,
        local_exact_clip: float = 4.0,
        local_exact_symmetric: bool = True,
        local_exact_start_epoch: Optional[int] = None,
        local_exact_end_epoch: Optional[int] = None,
        local_exact_start_frac: Optional[float] = None,
        local_exact_end_frac: Optional[float] = None,
        local_exact_density_filter: bool = False,
        local_exact_min_cell_count: Optional[int] = None,
        local_exact_min_cell_count_quantile: Optional[float] = None,
        local_exact_density_include_neighbor_cells: bool = False,
        local_exact_timing_sample_size: int = 2048,
        # Extra local repulsion mechanism: local density pressure term based on the ibFFT grid
        # 低成本局部密度推力，可能产生方格伪影
        local_density_pressure: bool = False,
        local_density_pressure_weight: float = 0.05,
        local_density_pressure_every: int = 1,
        local_density_pressure_min_count: int = 4,
        local_density_pressure_clip: float = 4.0,
        local_density_pressure_power: float = 1.0,
        # Experimental attraction execution controls. The default kernel mode
        # is device-adaptive: CPU resolves to thread-per-row, while CUDA may
        # use warp-per-row when the graph degree reaches the configured
        # threshold.
        attraction_kernel_mode: str = "auto",
        attraction_schedule_mode: str = "row_scan",
        attraction_calendar_memory_limit_bytes: Optional[int] = None,
        attraction_warp_min_degree: int = 8,
        umap_sgd_update_mode: UMAPSGDUpdateMode = "asynchronous",
        rng_lifecycle: RNGLifecycle = "independent",
        *,
        runtime: Optional[RuntimeConfig] = None,
        graph: Optional[GraphConfig] = None,
        umap: Optional[UMAPConfig] = None,
        initialization: Optional[InitializationConfig] = None,
        fft: Optional[FFTConfig] = None,
        ibumap: Optional[IBUMAPConfig] = None,
        cpu_umap: Optional[CPUUMAPConfig] = None,
        hybrid: Optional[HybridOptimizerConfig] = None,
        tfdp: Optional[TFDPConfig] = None,
        constraints: Optional[ConstraintConfig] = None,
        diagnostics: Optional[DiagnosticsConfig] = None,
    ) -> None:
        provided = set(self._provided_constructor_parameters)
        raw_algorithm = (
            algorithm
            if "algorithm" in provided
            else (runtime.algorithm if runtime is not None else "ibumap")
        )
        raw_device = (
            device
            if "device" in provided
            else (runtime.device if runtime is not None else "cpu")
        )
        self._deprecated_algorithm_aliases = []
        self._deprecated_device_aliases = ["gpu"] if raw_device == "gpu" else []
        constructor_locals = dict(locals())
        canonical_names = {
            "self", "provided", "raw_algorithm", "raw_device",
            "runtime", "graph", "umap", "initialization", "fft", "ibumap",
            "cpu_umap", "hybrid", "tfdp", "constraints", "diagnostics",
        }
        flat_values = {
            name: value
            for name, value in constructor_locals.items()
            if name not in canonical_names
        }
        self.config, self._legacy_flat_parameters_used = _build_effective_config(
            flat_values,
            provided,
            runtime=runtime,
            graph=graph,
            umap=umap,
            initialization=initialization,
            fft=fft,
            ibumap=ibumap,
            cpu_umap=cpu_umap,
            hybrid=hybrid,
            tfdp=tfdp,
            constraints=constraints,
            diagnostics=diagnostics,
        )
        self._projecting_legacy_values = True

        cfg = self.config
        experimental = cfg.ibumap.experimental
        local_cfg = experimental.local_exact_repulsion
        density_cfg = experimental.local_density_pressure
        attraction_schedule_cfg = experimental.attraction_schedule
        subsampling = experimental.kernel_subsampling
        repulsion_clipping = cfg.umap.repulsion_clipping
        total_clipping = experimental.total_update_clipping
        degree = cfg.umap.degree_damping

        # Feed the existing validation/initialization code with canonical values.
        algorithm = cfg.runtime.algorithm
        device = cfg.runtime.device
        attraction_mode = cfg.ibumap.attraction_mode
        repulsion_mode = cfg.ibumap.repulsion_mode
        n_components = cfg.umap.n_components
        n_neighbors = cfg.graph.n_neighbors
        n_epochs = cfg.umap.n_epochs
        init = cfg.initialization.init
        spectral_method = cfg.initialization.spectral_method
        spectral_tol = cfg.initialization.spectral_tol
        spectral_maxiter = cfg.initialization.spectral_maxiter
        spectral_ncv = cfg.initialization.spectral_ncv
        spectral_auto_defaults = cfg.initialization.spectral_auto_defaults
        spectral_scale_policy = cfg.initialization.spectral_scale_policy
        spectral_max_span = cfg.initialization.spectral_max_span
        spectral_jitter_relative = cfg.initialization.spectral_jitter_relative
        spectral_jitter_max = cfg.initialization.spectral_jitter_max
        report_duplicate_ratio = cfg.initialization.report_duplicate_ratio
        min_dist = cfg.umap.min_dist
        spread = cfg.umap.spread
        repulsion_strength = cfg.umap.repulsion_strength
        negative_sample_rate = cfg.umap.negative_sample_rate
        learning_rate = cfg.umap.learning_rate
        hybrid_config = cfg.hybrid
        random_state = cfg.runtime.random_state
        metric = cfg.graph.metric
        metric_kwds = cfg.graph.metric_kwds
        graph_backend = cfg.runtime.graph_backend
        workspace_policy = cfg.runtime.workspace_policy
        workspace_limit_bytes = cfg.runtime.workspace_limit_bytes
        p2m_mode = cfg.fft.p2m_mode
        deterministic = cfg.cpu_umap.deterministic
        rng_lifecycle = cfg.cpu_umap.rng_lifecycle
        fft_config = cfg.fft
        tfdp_config = cfg.tfdp
        constraint_config = cfg.constraints
        numerics_config = cfg.legacy_numerics
        ibfft_kernel_clip = cfg.fft.kernel_clip
        umap_epsilon = cfg.umap.epsilon
        ibfft_kernel_subsample_mode = None if subsampling is None else subsampling.mode
        ibfft_kernel_subsample_radius_cells = 2 if subsampling is None else subsampling.radius_cells
        ibfft_kernel_subsample_points = 4 if subsampling is None else subsampling.points
        n_jobs = cfg.graph.n_jobs
        angular_rp_forest = cfg.graph.angular_rp_forest
        low_memory = cfg.graph.low_memory
        verbose = cfg.runtime.verbose
        set_op_mix_ratio = cfg.graph.set_op_mix_ratio
        local_connectivity = cfg.graph.local_connectivity
        tqdm_kwds = cfg.cpu_umap.tqdm_kwds
        noise_mode = experimental.noise.mode
        noise_scale = experimental.noise.scale
        noise_decay = experimental.noise.decay
        noise_until_epoch = experimental.noise.until_epoch
        noise_seed = experimental.noise.seed
        hybrid_mode = experimental.noise.hybrid_mode
        hybrid_switch_epoch = experimental.noise.hybrid_switch_epoch
        noise_config = None
        attraction_degree_damping = _resolve_degree_enabled(cfg)
        attraction_degree_damping_mode = degree.mode
        attraction_degree_damping_ref = degree.ref
        attraction_degree_damping_ref_value = degree.ref_value
        attraction_degree_damping_power = degree.power
        attraction_degree_damping_min_scale = degree.min_scale
        repulsion_clip_norm = None if repulsion_clipping is None else repulsion_clipping.norm
        repulsion_clip_with_alpha = True if repulsion_clipping is None else repulsion_clipping.with_alpha
        repulsion_clip_epoch_range = None if repulsion_clipping is None else repulsion_clipping.epoch_range
        diagnostics_path = cfg.diagnostics.summary_path
        diagnostics_timing_path = cfg.diagnostics.timing_path
        diagnostics_thresholds = cfg.diagnostics.thresholds
        diagnostics_topk_path = cfg.diagnostics.topk_path
        diagnostics_topk = cfg.diagnostics.topk
        diagnostics_labels = cfg.diagnostics.labels
        diagnostics_point_ids = cfg.diagnostics.point_ids
        diagnostics_memory_path = cfg.diagnostics.memory_path
        diagnostics_memory_epoch_stride = cfg.diagnostics.memory_epoch_stride
        diagnostics_memory_sample_interval_ms = cfg.diagnostics.memory_sample_interval_ms
        total_update_clip_norm = None if total_clipping is None else total_clipping.norm
        total_update_clip_with_alpha = False if total_clipping is None else total_clipping.with_alpha
        total_update_clip_epoch_range = None if total_clipping is None else total_clipping.epoch_range
        local_exact_repulsion = local_cfg.enabled
        local_exact_k = local_cfg.k
        local_exact_radius_factor = local_cfg.radius_factor
        local_exact_weight = local_cfg.weight
        local_exact_every = local_cfg.every
        local_exact_clip = local_cfg.clip
        local_exact_symmetric = local_cfg.symmetric
        local_exact_start_epoch = local_cfg.start_epoch
        local_exact_end_epoch = local_cfg.end_epoch
        local_exact_start_frac = local_cfg.start_frac
        local_exact_end_frac = local_cfg.end_frac
        local_exact_density_filter = local_cfg.density_filter
        local_exact_min_cell_count = local_cfg.min_cell_count
        local_exact_min_cell_count_quantile = local_cfg.min_cell_count_quantile
        local_exact_density_include_neighbor_cells = local_cfg.density_include_neighbor_cells
        local_exact_timing_sample_size = local_cfg.timing_sample_size
        local_density_pressure = density_cfg.enabled
        local_density_pressure_weight = density_cfg.weight
        local_density_pressure_every = density_cfg.every
        local_density_pressure_min_count = density_cfg.min_count
        local_density_pressure_clip = density_cfg.clip
        local_density_pressure_power = density_cfg.power
        attraction_kernel_mode = attraction_schedule_cfg.kernel_mode
        attraction_schedule_mode = attraction_schedule_cfg.schedule_mode
        attraction_calendar_memory_limit_bytes = (
            attraction_schedule_cfg.calendar_memory_limit_bytes
        )
        attraction_warp_min_degree = attraction_schedule_cfg.warp_min_degree
        umap_sgd_update_mode = cfg.cpu_umap.update_mode
        noise_direct_overrides = (
            noise_mode != "none"
            or noise_scale != 0.0
            or noise_decay != "linear"
            or noise_until_epoch is not None
            or noise_seed is not None
            or hybrid_mode != "none"
            or hybrid_switch_epoch is not None
        )
        if noise_config is not None and noise_direct_overrides:
            raise ValueError("Use either noise_config or flat noise/hybrid parameters, not both")
        noise = noise_config or NoiseConfig(
            mode=noise_mode,
            scale=noise_scale,
            decay=noise_decay,
            until_epoch=noise_until_epoch,
            seed=noise_seed,
            hybrid_mode=hybrid_mode,
            hybrid_switch_epoch=hybrid_switch_epoch,
        )
        ibfft_kernel_clip = float(ibfft_kernel_clip)
        if not np.isfinite(ibfft_kernel_clip) or ibfft_kernel_clip <= 0:
            raise ValueError("ibfft_kernel_clip must be positive and finite")
        if umap_epsilon is not None:
            umap_epsilon = float(umap_epsilon)
            if not np.isfinite(umap_epsilon) or umap_epsilon <= 0:
                raise ValueError("umap_epsilon must be positive and finite")
        if ibfft_kernel_subsample_mode not in (None, "near_origin"):
            raise ValueError(
                "ibfft_kernel_subsample_mode must be None or 'near_origin'"
            )
        if (
            isinstance(ibfft_kernel_subsample_radius_cells, bool)
            or not isinstance(ibfft_kernel_subsample_radius_cells, Integral)
            or ibfft_kernel_subsample_radius_cells < 0
        ):
            raise ValueError(
                "ibfft_kernel_subsample_radius_cells must be a non-negative integer"
            )
        if (
            isinstance(ibfft_kernel_subsample_points, bool)
            or not isinstance(ibfft_kernel_subsample_points, Integral)
            or ibfft_kernel_subsample_points < 1
        ):
            raise ValueError("ibfft_kernel_subsample_points must be a positive integer")

        numerics = numerics_config or NumericsConfig(deterministic=deterministic)
        if umap_epsilon is not None:
            numerics = replace(numerics, epsilon=umap_epsilon)

        fft_runtime_config = replace(fft_config or FFTConfig(), p2m_mode=p2m_mode)
        if fft_kernel_cache_policy is not None:
            fft_runtime_config = replace(
                fft_runtime_config,
                kernel_cache_policy=fft_kernel_cache_policy,
            )
        if fft_kernel_cache_limit_bytes is not None:
            fft_runtime_config = replace(
                fft_runtime_config,
                kernel_cache_limit_bytes=int(fft_kernel_cache_limit_bytes),
            )
        if fft_kernel_cache_max_entries is not None:
            fft_runtime_config = replace(
                fft_runtime_config,
                kernel_cache_max_entries=int(fft_kernel_cache_max_entries),
            )

        self.runtime = EffectiveConfig(
            algorithm=algorithm,
            device=device,
            attraction_mode=attraction_mode,
            repulsion_mode=repulsion_mode,
            umap_sgd_update_mode=umap_sgd_update_mode,
            rng_lifecycle=rng_lifecycle,
            graph_backend=graph_backend,
            workspace_policy=workspace_policy,
            workspace_limit_bytes=workspace_limit_bytes,
            deterministic=deterministic,
            fft=fft_runtime_config,
            tfdp=tfdp_config or TFDPConfig(),
            constraint=constraint_config or ConstraintConfig(),
            numerics=numerics,
            noise=noise,
            hybrid=hybrid_config,
        )
        self._numeric_dtype = resolve_numeric_dtype(self.runtime.numerics.dtype)

        self.ibfft_kernel_clip = ibfft_kernel_clip
        self.umap_epsilon = float(self.runtime.numerics.epsilon)
        self.ibfft_kernel_subsample_mode = ibfft_kernel_subsample_mode
        self.ibfft_kernel_subsample_radius_cells = int(
            ibfft_kernel_subsample_radius_cells
        )
        self.ibfft_kernel_subsample_points = int(ibfft_kernel_subsample_points)
        if (
            self.runtime.device == "cuda"
            and self.ibfft_kernel_subsample_mode is not None
        ):
            raise NotImplementedError(
                "ibFFT kernel sub-sampling currently supports CPU only"
            )

        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.n_epochs = n_epochs
        self.min_dist = float(min_dist)
        self.spread = float(spread)
        self.repulsion_strength = float(repulsion_strength)
        self.negative_sample_rate = float(negative_sample_rate)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.workspace_policy = workspace_policy
        self.workspace_limit_bytes = workspace_limit_bytes
        self.p2m_mode = p2m_mode
        self.fft_kernel_cache_policy = self.runtime.fft.kernel_cache_policy
        self.fft_kernel_cache_limit_bytes = self.runtime.fft.kernel_cache_limit_bytes
        self.fft_kernel_cache_max_entries = self.runtime.fft.kernel_cache_max_entries
        self.metric = metric
        self.metric_kwds = metric_kwds or {}
        self.n_jobs = int(n_jobs)
        self.angular_rp_forest = bool(angular_rp_forest)
        self.low_memory = bool(low_memory)
        self.verbose = bool(verbose)
        self.set_op_mix_ratio = float(set_op_mix_ratio)
        self.local_connectivity = float(local_connectivity)
        self.tqdm_kwds = tqdm_kwds

        supported_damping_modes = ("weighted_degree", "degree")
        supported_damping_refs = ("median", "mean", "p90", "p95", "p99", "manual")
        if attraction_degree_damping_mode not in supported_damping_modes:
            raise ValueError(
                "attraction_degree_damping_mode must be 'weighted_degree' or 'degree'"
            )
        if attraction_degree_damping_ref not in supported_damping_refs:
            raise ValueError(
                "attraction_degree_damping_ref must be one of "
                "'median', 'mean', 'p90', 'p95', 'p99', or 'manual'"
            )
        if attraction_degree_damping_ref_value is not None:
            attraction_degree_damping_ref_value = float(
                attraction_degree_damping_ref_value
            )
            if (
                not np.isfinite(attraction_degree_damping_ref_value)
                or attraction_degree_damping_ref_value <= 0.0
            ):
                raise ValueError(
                    "attraction_degree_damping_ref_value must be positive and finite"
                )
        if (
            attraction_degree_damping_ref == "manual"
            and attraction_degree_damping_ref_value is None
        ):
            raise ValueError(
                "attraction_degree_damping_ref_value is required when "
                "attraction_degree_damping_ref='manual'"
            )
        attraction_degree_damping_power = float(attraction_degree_damping_power)
        if (
            not np.isfinite(attraction_degree_damping_power)
            or attraction_degree_damping_power <= 0.0
        ):
            raise ValueError(
                "attraction_degree_damping_power must be positive and finite"
            )
        if attraction_degree_damping_min_scale is not None:
            attraction_degree_damping_min_scale = float(
                attraction_degree_damping_min_scale
            )
            if (
                not np.isfinite(attraction_degree_damping_min_scale)
                or not 0.0 < attraction_degree_damping_min_scale <= 1.0
            ):
                raise ValueError(
                    "attraction_degree_damping_min_scale must be in (0, 1]"
                )
        if attraction_degree_damping is None:
            attraction_degree_damping = self.runtime.algorithm in ("ibumap", "hybrid")
        self.attraction_degree_damping = bool(attraction_degree_damping)
        self.attraction_degree_damping_mode = attraction_degree_damping_mode
        self.attraction_degree_damping_ref = attraction_degree_damping_ref
        self.attraction_degree_damping_ref_value = attraction_degree_damping_ref_value
        self.attraction_degree_damping_power = attraction_degree_damping_power
        self.attraction_degree_damping_min_scale = attraction_degree_damping_min_scale
        if self.attraction_degree_damping and self.runtime.algorithm not in (
            "ibumap",
            "hybrid",
        ):
            raise ValueError(
                "attraction degree damping requires algorithm='ibumap' or 'hybrid'"
            )

        if repulsion_clip_norm is not None and repulsion_clip_norm <= 0:
            raise ValueError("repulsion_clip_norm must be positive when set")
        if total_update_clip_norm is not None and total_update_clip_norm <= 0:
            raise ValueError("total_update_clip_norm must be positive when set")
        self.repulsion_clip_norm = (
            None if repulsion_clip_norm is None else float(repulsion_clip_norm)
        )
        self.repulsion_clip_with_alpha = bool(repulsion_clip_with_alpha)
        self.repulsion_clip_epoch_range = _validate_clip_epoch_range(
            repulsion_clip_epoch_range,
            "repulsion_clip_epoch_range",
            self.n_epochs,
        )
        self.total_update_clip_norm = (
            None if total_update_clip_norm is None else float(total_update_clip_norm)
        )
        self.total_update_clip_with_alpha = bool(total_update_clip_with_alpha)
        self.total_update_clip_epoch_range = _validate_clip_epoch_range(
            total_update_clip_epoch_range,
            "total_update_clip_epoch_range",
            self.n_epochs,
        )
        self.diagnostics_path = diagnostics_path
        self.diagnostics_timing_path = diagnostics_timing_path
        default_thresholds = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
        thresholds = diagnostics_thresholds or default_thresholds
        self.diagnostics_thresholds = tuple(float(value) for value in thresholds)
        self.diagnostics_topk_path = diagnostics_topk_path
        self.diagnostics_topk = max(0, int(diagnostics_topk))
        self.diagnostics_labels = (
            None if diagnostics_labels is None else tuple(str(value) for value in diagnostics_labels)
        )
        self.diagnostics_point_ids = (
            None if diagnostics_point_ids is None else tuple(str(value) for value in diagnostics_point_ids)
        )
        self.diagnostics_memory_path = diagnostics_memory_path
        self.diagnostics_memory_epoch_stride = int(diagnostics_memory_epoch_stride)
        self.diagnostics_memory_sample_interval_ms = diagnostics_memory_sample_interval_ms

        if isinstance(local_exact_k, bool) or not isinstance(local_exact_k, Integral) or local_exact_k < 1:
            raise ValueError("local_exact_k must be a positive integer")
        if local_exact_radius_factor <= 0:
            raise ValueError("local_exact_radius_factor must be positive")
        if local_exact_weight < 0:
            raise ValueError("local_exact_weight must be non-negative")
        if isinstance(local_exact_every, bool) or not isinstance(local_exact_every, Integral) or local_exact_every < 1:
            raise ValueError("local_exact_every must be a positive integer")
        if local_exact_clip <= 0:
            raise ValueError("local_exact_clip must be positive")
        epoch_bounds = (local_exact_start_epoch, local_exact_end_epoch)
        for name, value in zip(("local_exact_start_epoch", "local_exact_end_epoch"), epoch_bounds):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer when set")
        if local_exact_end_epoch == 0:
            raise ValueError("local_exact_end_epoch must be positive when set")
        if (
            local_exact_start_epoch is not None
            and local_exact_end_epoch is not None
            and local_exact_start_epoch >= local_exact_end_epoch
        ):
            raise ValueError("local exact epoch bounds must satisfy start < end")
        if self.n_epochs is not None and local_exact_end_epoch is not None and local_exact_end_epoch > self.n_epochs:
            raise ValueError(f"local_exact_end_epoch must not exceed n_epochs={self.n_epochs}")
        frac_bounds = (local_exact_start_frac, local_exact_end_frac)
        if any(value is not None for value in epoch_bounds) and any(value is not None for value in frac_bounds):
            raise ValueError("Use local exact epoch bounds or fraction bounds, not both")
        if local_exact_start_frac is not None and not 0.0 <= float(local_exact_start_frac) < 1.0:
            raise ValueError("local_exact_start_frac must be in [0, 1)")
        if local_exact_end_frac is not None and not 0.0 < float(local_exact_end_frac) <= 1.0:
            raise ValueError("local_exact_end_frac must be in (0, 1]")
        if (
            local_exact_start_frac is not None
            and local_exact_end_frac is not None
            and float(local_exact_start_frac) >= float(local_exact_end_frac)
        ):
            raise ValueError("local exact fraction bounds must satisfy start < end")
        if local_exact_min_cell_count is not None and (
            isinstance(local_exact_min_cell_count, bool)
            or not isinstance(local_exact_min_cell_count, Integral)
            or local_exact_min_cell_count < 1
        ):
            raise ValueError("local_exact_min_cell_count must be a positive integer when set")
        if local_exact_min_cell_count_quantile is not None and not 0.0 <= float(local_exact_min_cell_count_quantile) <= 1.0:
            raise ValueError("local_exact_min_cell_count_quantile must be in [0, 1]")
        if local_exact_min_cell_count is not None and local_exact_min_cell_count_quantile is not None:
            raise ValueError("Use local_exact_min_cell_count or local_exact_min_cell_count_quantile, not both")
        if (
            isinstance(local_exact_timing_sample_size, bool)
            or not isinstance(local_exact_timing_sample_size, Integral)
            or local_exact_timing_sample_size < 1
        ):
            raise ValueError("local_exact_timing_sample_size must be a positive integer")
        if local_density_pressure_weight < 0:
            raise ValueError("local_density_pressure_weight must be non-negative")
        if (
            isinstance(local_density_pressure_every, bool)
            or not isinstance(local_density_pressure_every, Integral)
            or local_density_pressure_every < 1
        ):
            raise ValueError("local_density_pressure_every must be a positive integer")
        if (
            isinstance(local_density_pressure_min_count, bool)
            or not isinstance(local_density_pressure_min_count, Integral)
            or local_density_pressure_min_count < 1
        ):
            raise ValueError("local_density_pressure_min_count must be a positive integer")
        if local_density_pressure_clip <= 0:
            raise ValueError("local_density_pressure_clip must be positive")
        if local_density_pressure_power <= 0:
            raise ValueError("local_density_pressure_power must be positive")

        # Experimental CPU-only diagnostics for supplementing ibFFT near-field repulsion.
        self.local_exact_repulsion = bool(local_exact_repulsion)
        self.local_exact_k = int(local_exact_k)
        self.local_exact_radius_factor = float(local_exact_radius_factor)
        self.local_exact_weight = float(local_exact_weight)
        self.local_exact_every = int(local_exact_every)
        self.local_exact_clip = float(local_exact_clip)
        self.local_exact_symmetric = bool(local_exact_symmetric)
        self.local_exact_start_epoch = (
            None if local_exact_start_epoch is None else int(local_exact_start_epoch)
        )
        self.local_exact_end_epoch = (
            None if local_exact_end_epoch is None else int(local_exact_end_epoch)
        )
        self.local_exact_start_frac = (
            None if local_exact_start_frac is None else float(local_exact_start_frac)
        )
        self.local_exact_end_frac = (
            None if local_exact_end_frac is None else float(local_exact_end_frac)
        )
        self.local_exact_density_filter = bool(
            local_exact_density_filter
            or local_exact_min_cell_count is not None
            or local_exact_min_cell_count_quantile is not None
        )
        self.local_exact_min_cell_count = (
            None if local_exact_min_cell_count is None else int(local_exact_min_cell_count)
        )
        self.local_exact_min_cell_count_quantile = (
            None
            if local_exact_min_cell_count_quantile is None
            else float(local_exact_min_cell_count_quantile)
        )
        self.local_exact_density_include_neighbor_cells = bool(
            local_exact_density_include_neighbor_cells
        )
        self.local_exact_timing_sample_size = int(local_exact_timing_sample_size)
        self.local_density_pressure = bool(local_density_pressure)
        self.local_density_pressure_weight = float(local_density_pressure_weight)
        self.local_density_pressure_every = int(local_density_pressure_every)
        self.local_density_pressure_min_count = int(local_density_pressure_min_count)
        self.local_density_pressure_clip = float(local_density_pressure_clip)
        self.local_density_pressure_power = float(local_density_pressure_power)
        self.attraction_kernel_mode = attraction_kernel_mode
        self.attraction_schedule_mode = attraction_schedule_mode
        self.attraction_calendar_memory_limit_bytes = (
            attraction_calendar_memory_limit_bytes
        )
        self.attraction_warp_min_degree = attraction_warp_min_degree
        local_mechanism_enabled = self.local_exact_repulsion or self.local_density_pressure
        if local_mechanism_enabled and self.runtime.algorithm != "ibumap":
            raise ValueError("local repulsion mechanisms require algorithm='ibumap'")
        if local_mechanism_enabled and self.runtime.device == "cuda":
            raise NotImplementedError(
                "local_exact_repulsion and local_density_pressure currently support CPU only"
            )

        self._projecting_legacy_values = False
        self.input_distance_func = None

        self._a, self._b = _find_ab_params(self.spread, self.min_dist)
        self._resolved_n_epochs: Optional[int] = None

        self._time_costs: Dict[str, float] = {}
        self._init_diagnostics: Dict[str, Any] = {}

        self._pipeline = None
        self._pipeline_device: Optional[str] = None
        self._memory_recorder: Optional[StageMemoryRecorder] = None
        self._fitted = False
        self.graph_ = None
        self.embedding_ = None
        self.init_embedding_ = None
        self._raw_data = None
        self._last_entrypoint: Optional[str] = None
        self._last_update_init_override = False

    @property
    def runtime(self) -> EffectiveConfig:
        self._refresh_runtime_view()
        return self._runtime_compat

    @runtime.setter
    def runtime(self, value: EffectiveConfig) -> None:
        if not hasattr(self, "config"):
            self._runtime_compat = deepcopy(value)
            return
        candidate = deepcopy(self.config)
        candidate.runtime = RuntimeConfig(
            algorithm=value.algorithm,
            device=value.device,
            graph_backend=value.graph_backend,
            random_state=candidate.runtime.random_state,
            workspace_policy=value.workspace_policy,
            workspace_limit_bytes=value.workspace_limit_bytes,
            verbose=candidate.runtime.verbose,
        )
        candidate.ibumap.attraction_mode = value.attraction_mode
        candidate.ibumap.repulsion_mode = value.repulsion_mode
        candidate.cpu_umap.update_mode = value.umap_sgd_update_mode
        candidate.cpu_umap.rng_lifecycle = value.rng_lifecycle
        candidate.cpu_umap.deterministic = value.deterministic
        candidate.fft = deepcopy(value.fft)
        candidate.hybrid = deepcopy(value.hybrid)
        candidate.tfdp = deepcopy(value.tfdp)
        candidate.constraints = deepcopy(value.constraint)
        candidate.legacy_numerics = deepcopy(value.numerics)
        candidate.umap.epsilon = float(value.numerics.epsilon)
        candidate.ibumap.experimental.noise = deepcopy(value.noise)
        _validate_effective_config(candidate)
        if candidate.runtime.device == "cuda" and candidate.cpu_umap.deterministic:
            warnings.warn(
                "deterministic=True on CUDA seeds all random streams for "
                "same-device repeatability, selects deterministic ibFFT P2M "
                "accumulation, reuses workspace-owned charge FFT plans and "
                "buffers outside minimal-workspace mode, uses fixed-order "
                "CSR degree reductions, "
                "canonicalizes connected-component labels, and uses "
                "deterministic CSR Laplacian construction and SpMV for "
                "spectral eigsh; this may use additional workspace or reduce "
                "performance",
                RuntimeWarning,
                stacklevel=3,
            )
        self.config = candidate
        self._refresh_runtime_view()

    def _refresh_runtime_view(self) -> None:
        self._numeric_dtype = resolve_numeric_dtype(self.config.legacy_numerics.dtype)
        self._runtime_compat = EffectiveConfig(
            algorithm=self.config.runtime.algorithm,
            device=self.config.runtime.device,
            attraction_mode=self.config.ibumap.attraction_mode,
            repulsion_mode=self.config.ibumap.repulsion_mode,
            graph_backend=self.config.runtime.graph_backend,
            workspace_policy=self.config.runtime.workspace_policy,
            workspace_limit_bytes=self.config.runtime.workspace_limit_bytes,
            deterministic=self.config.cpu_umap.deterministic,
            rng_lifecycle=self.config.cpu_umap.rng_lifecycle,
            fft=deepcopy(self.config.fft),
            tfdp=deepcopy(self.config.tfdp),
            constraint=deepcopy(self.config.constraints),
            numerics=NumericsConfig(
                dtype=self.config.legacy_numerics.dtype,
                deterministic=self.config.legacy_numerics.deterministic,
                epsilon=self.config.umap.epsilon,
            ),
            noise=deepcopy(self.config.ibumap.experimental.noise),
            hybrid=deepcopy(self.config.hybrid),
            umap_sgd_update_mode=self.config.cpu_umap.update_mode,
        )

    def _commit_config_mutation(self, mutation, *, refresh_runtime: bool = False) -> None:
        candidate = deepcopy(self.config)
        mutation(candidate)
        _validate_effective_config(candidate)
        self.config = candidate
        self._numeric_dtype = resolve_numeric_dtype(self.config.legacy_numerics.dtype)
        if refresh_runtime:
            self._refresh_runtime_view()

    @property
    def numeric_dtype(self) -> np.dtype:
        return resolve_numeric_dtype(self.config.legacy_numerics.dtype)

    @property
    def attraction_degree_damping(self) -> bool:
        return _resolve_degree_enabled(self.config)

    @attraction_degree_damping.setter
    def attraction_degree_damping(self, value: Optional[bool]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            config.umap.degree_damping.enabled = value
            config.umap.degree_damping.__post_init__()

        self._commit_config_mutation(mutate)

    @property
    def repulsion_clip_norm(self) -> Optional[float]:
        clipping = self.config.umap.repulsion_clipping
        return None if clipping is None else clipping.norm

    @repulsion_clip_norm.setter
    def repulsion_clip_norm(self, value: Optional[float]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            current = config.umap.repulsion_clipping
            config.umap.repulsion_clipping = (
                None
                if value is None
                else ForceClippingConfig(
                    norm=value,
                    with_alpha=True if current is None else current.with_alpha,
                    epoch_range=None if current is None else current.epoch_range,
                )
            )

        self._commit_config_mutation(mutate)

    @property
    def repulsion_clip_with_alpha(self) -> bool:
        clipping = self.config.umap.repulsion_clipping
        return True if clipping is None else clipping.with_alpha

    @repulsion_clip_with_alpha.setter
    def repulsion_clip_with_alpha(self, value: bool) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            if config.umap.repulsion_clipping is not None:
                config.umap.repulsion_clipping.with_alpha = bool(value)

        self._commit_config_mutation(mutate)

    @property
    def repulsion_clip_epoch_range(self) -> Optional[tuple[int, int]]:
        clipping = self.config.umap.repulsion_clipping
        return None if clipping is None else clipping.epoch_range

    @repulsion_clip_epoch_range.setter
    def repulsion_clip_epoch_range(self, value: Optional[Sequence[int]]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            if config.umap.repulsion_clipping is not None:
                config.umap.repulsion_clipping.epoch_range = _validate_clip_epoch_range(
                    value, "repulsion_clip_epoch_range", config.umap.n_epochs
                )

        self._commit_config_mutation(mutate)

    @property
    def total_update_clip_norm(self) -> Optional[float]:
        clipping = self.config.ibumap.experimental.total_update_clipping
        return None if clipping is None else clipping.norm

    @total_update_clip_norm.setter
    def total_update_clip_norm(self, value: Optional[float]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            current = config.ibumap.experimental.total_update_clipping
            config.ibumap.experimental.total_update_clipping = (
                None
                if value is None
                else ForceClippingConfig(
                    norm=value,
                    with_alpha=False if current is None else current.with_alpha,
                    epoch_range=None if current is None else current.epoch_range,
                )
            )

        self._commit_config_mutation(mutate)

    @property
    def total_update_clip_with_alpha(self) -> bool:
        clipping = self.config.ibumap.experimental.total_update_clipping
        return False if clipping is None else clipping.with_alpha

    @total_update_clip_with_alpha.setter
    def total_update_clip_with_alpha(self, value: bool) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            clipping = config.ibumap.experimental.total_update_clipping
            if clipping is not None:
                clipping.with_alpha = bool(value)

        self._commit_config_mutation(mutate)

    @property
    def total_update_clip_epoch_range(self) -> Optional[tuple[int, int]]:
        clipping = self.config.ibumap.experimental.total_update_clipping
        return None if clipping is None else clipping.epoch_range

    @total_update_clip_epoch_range.setter
    def total_update_clip_epoch_range(self, value: Optional[Sequence[int]]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            clipping = config.ibumap.experimental.total_update_clipping
            if clipping is not None:
                clipping.epoch_range = _validate_clip_epoch_range(
                    value, "total_update_clip_epoch_range", config.umap.n_epochs
                )

        self._commit_config_mutation(mutate)

    @property
    def ibfft_kernel_subsample_mode(self) -> Optional[str]:
        config = self.config.ibumap.experimental.kernel_subsampling
        return None if config is None else config.mode

    @ibfft_kernel_subsample_mode.setter
    def ibfft_kernel_subsample_mode(self, value: Optional[str]) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            current = config.ibumap.experimental.kernel_subsampling
            config.ibumap.experimental.kernel_subsampling = (
                None
                if value is None
                else KernelSubsamplingConfig(
                    mode=value,
                    radius_cells=2 if current is None else current.radius_cells,
                    points=4 if current is None else current.points,
                )
            )

        self._commit_config_mutation(mutate)

    @property
    def ibfft_kernel_subsample_radius_cells(self) -> int:
        config = self.config.ibumap.experimental.kernel_subsampling
        return 2 if config is None else config.radius_cells

    @ibfft_kernel_subsample_radius_cells.setter
    def ibfft_kernel_subsample_radius_cells(self, value: int) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            current = config.ibumap.experimental.kernel_subsampling
            if current is not None:
                config.ibumap.experimental.kernel_subsampling = KernelSubsamplingConfig(
                    mode=current.mode,
                    radius_cells=value,
                    points=current.points,
                )

        self._commit_config_mutation(mutate)

    @property
    def ibfft_kernel_subsample_points(self) -> int:
        config = self.config.ibumap.experimental.kernel_subsampling
        return 4 if config is None else config.points

    @ibfft_kernel_subsample_points.setter
    def ibfft_kernel_subsample_points(self, value: int) -> None:
        if getattr(self, "_projecting_legacy_values", False):
            return

        def mutate(config):
            current = config.ibumap.experimental.kernel_subsampling
            if current is not None:
                config.ibumap.experimental.kernel_subsampling = KernelSubsamplingConfig(
                    mode=current.mode,
                    radius_cells=current.radius_cells,
                    points=value,
                )

        self._commit_config_mutation(mutate)

    def _select_pipeline(self, require_graph_backend: bool = True):
        from .pipeline.cpu_pipeline import CPUPipeline
        from .pipeline.gpu_pipeline import GPUPipeline
        from .pipeline.metal_pipeline import MetalPipeline

        if self.runtime.device == "cuda":
            if require_graph_backend:
                backend = resolve_graph_backend(
                    self.runtime.device,
                    self.runtime.graph_backend,
                )
                self.runtime = replace(self.runtime, graph_backend=backend)
                ensure_gpu_runtime(self.runtime.graph_backend)
            else:
                ensure_cupy_runtime()
            return GPUPipeline(self)

        if self.runtime.device == "metal":
            backend = resolve_graph_backend(
                self.runtime.device,
                self.runtime.graph_backend,
            )
            self.runtime = replace(self.runtime, graph_backend=backend)
            ensure_metal_runtime()
            return MetalPipeline(self)

        backend = resolve_graph_backend(self.runtime.device, self.runtime.graph_backend)
        self.runtime = replace(self.runtime, graph_backend=backend)
        return CPUPipeline(self)

    def _ensure_pipeline(self, require_graph_backend: bool = True):
        if self._pipeline is None or self._pipeline_device != self.runtime.device:
            self._pipeline = self._select_pipeline(
                require_graph_backend=require_graph_backend,
            )
            self._pipeline_device = self.runtime.device

    def _validate_fixed_input_backend(self) -> None:
        if self.runtime.device == "cuda" and self.runtime.algorithm == "umap":
            raise NotImplementedError(
                "GPU algorithm='umap' uses cuML's full fit_transform path and does "
                "not expose a graph-to-embedding optimizer. Use device='cpu' for "
                "algorithm='umap' or algorithm='ibumap' for GPU fixed-input "
                "benchmarks."
            )

    def _start_memory_diagnostics(self, entrypoint: str) -> Optional[StageMemoryRecorder]:
        self._finish_memory_diagnostics()
        if self.diagnostics_memory_path is None:
            self._memory_recorder = None
            return None
        recorder = StageMemoryRecorder(
            self.diagnostics_memory_path,
            run_id=f"{entrypoint}-{time():.6f}",
            entrypoint=entrypoint,
            algorithm=self.config.runtime.algorithm,
            device=self.config.runtime.device,
            epoch_stride=self.diagnostics_memory_epoch_stride,
            sample_interval_ms=self.diagnostics_memory_sample_interval_ms,
            metadata={
                "n_components": self.config.umap.n_components,
                "n_neighbors": self.config.graph.n_neighbors,
                "n_epochs": self.config.umap.n_epochs,
                "workspace_policy": self.config.runtime.workspace_policy,
            },
        )
        self._memory_recorder = recorder
        recorder.record("total", "begin")
        return recorder

    def _finish_memory_diagnostics(self) -> None:
        recorder = getattr(self, "_memory_recorder", None)
        self._memory_recorder = None
        if recorder is not None:
            recorder.close()

    def _reject_hybrid_entrypoint(self, entrypoint: str) -> None:
        if self.runtime.algorithm == "hybrid":
            raise NotImplementedError(
                "algorithm='hybrid' currently supports only "
                "optimize_from_graph(fuzzy_graph, init_embedding); "
                f"{entrypoint} is not implemented"
            )

    def _normalize_optimization_rng_state(self, rng_state):
        if rng_state is None:
            return None
        if self.runtime.algorithm != "umap" or self.runtime.device != "cpu":
            raise ValueError(
                "rng_state is only supported by CPU algorithm='umap' "
                "optimization-only entrypoints"
            )
        return normalize_rng_state(rng_state)

    def fit(self, X, y=None):
        self._reject_hybrid_entrypoint("fit/fit_transform")
        recorder = self._start_memory_diagnostics("fit")
        t0 = time()
        try:
            if recorder is not None:
                recorder.record("input_validation", "begin")
            X = check_array(X, dtype=self.numeric_dtype, accept_sparse="csr", order="C")
            self._raw_data = X
            _validate_init_shape(
                self.config.initialization.init,
                X.shape[0],
                self.config.umap.n_components,
            )
            if recorder is not None:
                recorder.record(
                    "input_validation",
                    "end",
                    metadata={"shape": tuple(int(value) for value in X.shape)},
                )

            self._a, self._b = _find_ab_params(self.spread, self.min_dist)
            self._time_costs = {}
            self._init_diagnostics = {}

            self._ensure_pipeline(require_graph_backend=True)

            self.embedding_ = self._pipeline.fit(X, init=self.config.initialization.init)
            self.graph_ = self._pipeline.state.graph
            self._time_costs["embedding_time"] = time() - t0
            self._fitted = True
            self._last_entrypoint = "fit"
            self._last_update_init_override = False
            if recorder is not None:
                recorder.record("total", "end")
            return self
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "total",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise
        finally:
            self._finish_memory_diagnostics()

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.embedding_

    def _preparation_model(self, device: Optional[str]):
        """Return an isolated model when preparation targets another device."""
        if device is None:
            return self
        resolved_device = str(device).lower()
        if resolved_device == "gpu":
            resolved_device = "cuda"
        if resolved_device not in {"cpu", "cuda", "metal"}:
            raise ValueError(
                "device must be 'cpu', 'cuda', or 'metal' for fixed-input preparation, "
                f"got {device!r}"
            )
        if resolved_device == self.runtime.device:
            return self

        # Do not mutate a fitted estimator merely to prepare artifacts for a
        # different device. Passing the canonical groups preserves every
        # resolved configuration value while rebuilding only light API state.
        return type(self)(
            runtime=replace(deepcopy(self.config.runtime), device=resolved_device),
            graph=deepcopy(self.config.graph),
            umap=deepcopy(self.config.umap),
            initialization=deepcopy(self.config.initialization),
            fft=deepcopy(self.config.fft),
            ibumap=deepcopy(self.config.ibumap),
            cpu_umap=deepcopy(self.config.cpu_umap),
            hybrid=deepcopy(self.config.hybrid),
            tfdp=deepcopy(self.config.tfdp),
            constraints=deepcopy(self.config.constraints),
            diagnostics=deepcopy(self.config.diagnostics),
            numerics_config=deepcopy(self.config.legacy_numerics),
        )

    def prepare_fixed_inputs(
        self,
        X,
        *,
        device: Optional[str] = None,
        return_knn: bool = True,
        return_raw_graph: bool = True,
        return_optimizer_graph: bool = True,
        return_init: bool = True,
        host_output: bool = True,
    ) -> "PreparedInputs":
        """Prepare reusable KNN, graph, and initialization artifacts.

        ``fuzzy_graph`` in the returned :class:`PreparedInputs` is the raw
        fuzzy simplicial-set graph for persistence. ``optimizer_graph`` is the
        thresholded, canonical CSR graph actually consumed by IBUMAP's
        optimizer. The method does not fit or mutate this estimator; passing a
        different ``device`` creates a transient configuration-equivalent
        preparation model instead.
        """
        from .prepared import prepare_fixed_inputs

        model = self._preparation_model(device)
        return prepare_fixed_inputs(
            model,
            X,
            return_knn=return_knn,
            return_raw_graph=return_raw_graph,
            return_optimizer_graph=return_optimizer_graph,
            return_init=return_init,
            host_output=host_output,
        )

    def prepare_fixed_inputs_from_knn(
        self,
        X,
        knn_indices,
        knn_distances,
        *,
        device: Optional[str] = None,
        return_knn: bool = True,
        return_raw_graph: bool = True,
        return_optimizer_graph: bool = True,
        return_init: bool = True,
        host_output: bool = True,
    ) -> "PreparedInputs":
        """Prepare reusable artifacts from external KNN arrays.

        This skips nearest-neighbor search while keeping fuzzy-graph creation,
        graph preprocessing, CUDA transfers, and initialization under
        IBUMAP's ownership.
        """
        from .prepared import prepare_fixed_inputs_from_knn

        model = self._preparation_model(device)
        return prepare_fixed_inputs_from_knn(
            model,
            X,
            knn_indices,
            knn_distances,
            return_knn=return_knn,
            return_raw_graph=return_raw_graph,
            return_optimizer_graph=return_optimizer_graph,
            return_init=return_init,
            host_output=host_output,
        )

    def prepare_fixed_inputs_from_graph(
        self,
        X,
        fuzzy_graph,
        *,
        device: Optional[str] = None,
        return_raw_graph: bool = True,
        return_optimizer_graph: bool = True,
        return_init: bool = True,
        host_output: bool = True,
    ) -> "PreparedInputs":
        """Prepare optimizer artifacts from an already-built raw fuzzy graph."""
        from .prepared import prepare_fixed_inputs_from_graph

        model = self._preparation_model(device)
        return prepare_fixed_inputs_from_graph(
            model,
            X,
            fuzzy_graph,
            return_raw_graph=return_raw_graph,
            return_optimizer_graph=return_optimizer_graph,
            return_init=return_init,
            host_output=host_output,
        )

    def fit_transform_from_knn(self, X, knn_indices, knn_distances):
        """Fit an embedding from caller-provided nearest-neighbor results.

        This entrypoint is for the fixed-KNN benchmark. It skips nearest-neighbor
        search, constructs the fuzzy simplicial set from ``knn_indices`` and
        ``knn_distances``, then runs the normal initialization and optimization
        stages. Timing fields include ``fixed_knn_total_time``,
        ``fuzzy_graph_time``/``build_graph_time``, ``embd_init_time``, and
        ``embd_opt_time`` plus optimizer-specific fields such as
        ``opt_prep_time``, ``attr_time``, ``repl_time``, and ``appl_time``.
        """
        self._reject_hybrid_entrypoint("fit_transform_from_knn")
        self._validate_fixed_input_backend()
        recorder = self._start_memory_diagnostics("fit_transform_from_knn")
        t0 = time()
        try:
            if recorder is not None:
                recorder.record("input_validation", "begin")
            X = check_array(X, dtype=self.numeric_dtype, accept_sparse="csr", order="C")
            self._raw_data = X
            _validate_init_shape(
                self.config.initialization.init,
                X.shape[0],
                self.config.umap.n_components,
            )
            if recorder is not None:
                recorder.record(
                    "input_validation",
                    "end",
                    metadata={"shape": tuple(int(value) for value in X.shape)},
                )

            self._a, self._b = _find_ab_params(self.spread, self.min_dist)
            self._time_costs = {}
            self._init_diagnostics = {}
            self._ensure_pipeline(require_graph_backend=False)

            self.embedding_ = self._pipeline.fit_from_knn(
                X,
                knn_indices=knn_indices,
                knn_distances=knn_distances,
                init=self.config.initialization.init,
            )
            self.graph_ = self._pipeline.state.graph
            total_time = time() - t0
            self._time_costs["fixed_knn_total_time"] = total_time
            self._time_costs["embedding_time"] = total_time
            self._fitted = True
            self._last_entrypoint = "fit_transform_from_knn"
            self._last_update_init_override = False
            if recorder is not None:
                recorder.record("total", "end")
            return self.embedding_
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "total",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise
        finally:
            self._finish_memory_diagnostics()

    def optimize_from_graph(
        self,
        X=None,
        fuzzy_graph=None,
        init_embedding=None,
        *,
        rng_state: Optional[Sequence[int]] = None,
    ):
        """Optimize an embedding from a caller-provided graph and initialization.

        This entrypoint is for the optimization-only benchmark. It skips
        nearest-neighbor search, fuzzy graph construction, and spectral/random
        initialization. The preferred CPU form is
        ``optimize_from_graph(fuzzy_graph, init_embedding)``. The historical
        ``optimize_from_graph(X, fuzzy_graph, init_embedding)`` form remains
        temporarily available for compatibility. The supplied ``fuzzy_graph``
        should be a scipy sparse CSR matrix or another scipy sparse matrix
        convertible to CSR. The supplied ``init_embedding`` is copied as the
        configured numeric dtype before optimization so the caller's array is
        not modified in place. Timing fields include
        ``optimization_only_time``, ``embd_opt_time``, ``opt_prep_time``, and
        optimizer-specific ``attr_time``/``repl_time``/``appl_time`` when
        available. For strict low-level compatibility, CPU UMAP callers may
        pass the explicit three-element integer ``rng_state`` that would
        otherwise be drawn immediately before layout optimization.
        """
        if init_embedding is None:
            if X is None or fuzzy_graph is None:
                raise TypeError(
                    "optimize_from_graph requires fuzzy_graph and init_embedding"
                )
            init_embedding = fuzzy_graph
            fuzzy_graph = X
            X = None
        elif X is not None:
            warnings.warn(
                "optimize_from_graph(X, fuzzy_graph, init_embedding) is "
                "deprecated for CPU optimization-only runs; pass "
                "optimize_from_graph(fuzzy_graph, init_embedding) instead",
                FutureWarning,
                stacklevel=2,
            )
        elif fuzzy_graph is None:
            raise TypeError(
                "optimize_from_graph requires fuzzy_graph and init_embedding"
            )
        if X is None and self.runtime.device != "cpu":
            raise NotImplementedError(
                "the graph+init optimize_from_graph form currently supports CPU only"
            )
        rng_state = self._normalize_optimization_rng_state(rng_state)
        self._validate_fixed_input_backend()
        recorder = self._start_memory_diagnostics("optimize_from_graph")
        t0 = time()
        try:
            if recorder is not None:
                recorder.record("input_validation", "begin")
            if X is not None:
                X = check_array(
                    X,
                    dtype=self.numeric_dtype,
                    accept_sparse="csr",
                    order="C",
                )
            self._raw_data = X
            if recorder is not None:
                graph_shape = getattr(fuzzy_graph, "shape", None)
                recorder.record(
                    "input_validation",
                    "end",
                    metadata={
                        "shape": (
                            tuple(int(value) for value in X.shape)
                            if X is not None
                            else (
                                None
                                if graph_shape is None
                                else tuple(int(value) for value in graph_shape)
                            )
                        ),
                        "graph_shape": (
                            None
                            if graph_shape is None
                            else tuple(int(value) for value in graph_shape)
                        ),
                        "raw_data_supplied": X is not None,
                    },
                )

            self._a, self._b = _find_ab_params(self.spread, self.min_dist)
            self._time_costs = {}
            self._init_diagnostics = {
                "init_method": "external",
                "initialization_bypassed": True,
            }
            self._ensure_pipeline(require_graph_backend=False)

            if self.runtime.device == "cpu":
                self.embedding_ = self._pipeline.optimize_from_graph(
                    fuzzy_graph=fuzzy_graph,
                    init_embedding=init_embedding,
                    X=X,
                    rng_state=rng_state,
                )
            else:
                self.embedding_ = self._pipeline.optimize_from_graph(
                    X,
                    fuzzy_graph=fuzzy_graph,
                    init_embedding=init_embedding,
                )
            self.graph_ = self._pipeline.state.graph
            total_time = time() - t0
            self._time_costs["optimization_only_time"] = total_time
            self._time_costs["embedding_time"] = total_time
            self._fitted = True
            self._last_entrypoint = "optimize_from_graph"
            self._last_update_init_override = False
            if recorder is not None:
                recorder.record("total", "end")
            return self.embedding_
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "total",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise
        finally:
            self._finish_memory_diagnostics()

    def optimize_from_prepared_graph(
        self,
        X=None,
        optimizer_graph=None,
        init_embedding=None,
        *,
        rng_state: Optional[Sequence[int]] = None,
    ):
        """Optimize directly from a graph returned as ``optimizer_graph``.

        Unlike :meth:`optimize_from_graph`, this entrypoint does not repeat
        fuzzy-graph thresholding, duplicate elimination, or CSR sorting. Use
        it with the ``optimizer_graph`` and ``init_embedding`` returned by
        :meth:`prepare_fixed_inputs` when repeated optimization runs share the
        same effective configuration.

        The preferred optimization-only form is
        ``optimize_from_prepared_graph(optimizer_graph, init_embedding)``.
        The historical
        ``optimize_from_prepared_graph(X, optimizer_graph, init_embedding)``
        form remains available when raw data must be retained for a later
        :meth:`update_embedding` call.

        CPU UMAP callers may pass a three-element integer ``rng_state`` to
        reproduce an externally managed low-level RNG lifecycle exactly.
        """
        if init_embedding is None:
            if X is None or optimizer_graph is None:
                raise TypeError(
                    "optimize_from_prepared_graph requires optimizer_graph "
                    "and init_embedding"
                )
            init_embedding = optimizer_graph
            optimizer_graph = X
            X = None
        elif optimizer_graph is None:
            raise TypeError(
                "optimize_from_prepared_graph requires optimizer_graph "
                "and init_embedding"
            )
        self._reject_hybrid_entrypoint("optimize_from_prepared_graph")
        rng_state = self._normalize_optimization_rng_state(rng_state)
        self._validate_fixed_input_backend()
        recorder = self._start_memory_diagnostics("optimize_from_prepared_graph")
        t0 = time()
        try:
            if recorder is not None:
                recorder.record("input_validation", "begin")
            if X is not None:
                X = check_array(
                    X,
                    dtype=self.numeric_dtype,
                    accept_sparse="csr",
                    order="C",
                )
            self._raw_data = X
            if recorder is not None:
                graph_shape = getattr(optimizer_graph, "shape", None)
                recorder.record(
                    "input_validation",
                    "end",
                    metadata={
                        "graph_shape": (
                            None
                            if graph_shape is None
                            else tuple(int(value) for value in graph_shape)
                        ),
                        "raw_data_supplied": X is not None,
                    },
                )

            self._a, self._b = _find_ab_params(self.spread, self.min_dist)
            self._time_costs = {}
            self._init_diagnostics = {
                "init_method": "prepared_external",
                "initialization_bypassed": True,
                "graph_preprocessing_bypassed": True,
            }
            self._ensure_pipeline(require_graph_backend=False)

            self.embedding_ = self._pipeline.optimize_from_prepared_graph(
                X,
                optimizer_graph=optimizer_graph,
                init_embedding=init_embedding,
                rng_state=rng_state,
            )
            self.graph_ = self._pipeline.state.graph
            total_time = time() - t0
            self._time_costs["optimization_from_prepared_graph_time"] = total_time
            self._time_costs["embedding_time"] = total_time
            self._fitted = True
            self._last_entrypoint = "optimize_from_prepared_graph"
            self._last_update_init_override = False
            if recorder is not None:
                recorder.record("total", "end")
            return self.embedding_
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "total",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise
        finally:
            self._finish_memory_diagnostics()

    def update_embedding(
        self,
        init=None,
        n_epochs: Optional[int] = None,
        min_dist: Optional[float] = None,
        repulsion_strength: Optional[float] = None,
    ):
        self._reject_hybrid_entrypoint("update_embedding")
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Model is not fitted")
        if self._raw_data is None:
            raise NotImplementedError(
                "update_embedding is unavailable after graph-only optimization "
                "because no raw X was supplied"
            )

        if init is not None:
            _validate_init_shape(init, self._raw_data.shape[0], self.n_components)

        if min_dist is not None and min_dist != self.min_dist:
            self.min_dist = float(min_dist)
            self._a, self._b = _find_ab_params(self.spread, self.min_dist)

        if repulsion_strength is not None:
            self.repulsion_strength = float(repulsion_strength)

        recorder = self._start_memory_diagnostics("update_embedding")
        t0 = time()
        try:
            self._init_diagnostics = {"init_method": "existing"} if init is None else {}
            self.embedding_ = self._pipeline.update(self._raw_data, init=init, n_epochs=n_epochs)
            self._time_costs["update_time"] = time() - t0
            self._last_entrypoint = "update_embedding"
            self._last_update_init_override = init is not None
            if recorder is not None:
                recorder.record("total", "end")
            return self.embedding_
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "total",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise
        finally:
            self._finish_memory_diagnostics()

    def get_time_costs(self) -> Dict[str, float]:
        return dict(self._time_costs)

    def get_init_diagnostics(self) -> Dict[str, Any]:
        return dict(self._init_diagnostics)

    def get_graph(self):
        return self.graph_

    def get_init_embedding(self):
        return self.init_embedding_

    def explain_effective_config(self) -> Dict[str, Any]:
        """Return path-aware configuration usage for this estimator.

        The report is derived from the checked-in parameter usage registry and
        distinguishes parameters consumed by the selected implementation path
        from parameters accepted for compatibility but ignored on that path.
        """
        from .parameter_usage import explain_model_config

        return explain_model_config(self)

    def set_runtime(
        self,
        algorithm: Optional[str] = None,
        device: Optional[str] = None,
        attraction_mode: Optional[str] = None,
        repulsion_mode: Optional[str] = None,
        graph_backend: Optional[str] = None,
        workspace_policy: Optional[str] = None,
        workspace_limit_bytes: Optional[int] = None,
        p2m_mode: Optional[str] = None,
        deterministic: Optional[bool] = None,
        umap_sgd_update_mode: Optional[UMAPSGDUpdateMode] = None,
        rng_lifecycle: Optional[RNGLifecycle] = None,
    ) -> None:
        kwargs = {
            "algorithm": algorithm or self.runtime.algorithm,
            "device": device or self.runtime.device,
            "attraction_mode": attraction_mode or self.runtime.attraction_mode,
            "repulsion_mode": repulsion_mode or self.runtime.repulsion_mode,
            "umap_sgd_update_mode": (
                self.runtime.umap_sgd_update_mode
                if umap_sgd_update_mode is None
                else umap_sgd_update_mode
            ),
            "rng_lifecycle": (
                self.runtime.rng_lifecycle
                if rng_lifecycle is None
                else rng_lifecycle
            ),
            "graph_backend": graph_backend or self.runtime.graph_backend,
            "workspace_policy": (
                self.runtime.workspace_policy
                if workspace_policy is None
                else workspace_policy
            ),
            "workspace_limit_bytes": (
                self.runtime.workspace_limit_bytes
                if workspace_limit_bytes is None
                else workspace_limit_bytes
            ),
            "deterministic": self.runtime.deterministic if deterministic is None else deterministic,
            "fft": replace(
                self.runtime.fft,
                p2m_mode=(
                    self.runtime.fft.p2m_mode if p2m_mode is None else p2m_mode
                ),
            ),
            "tfdp": self.runtime.tfdp,
            "constraint": self.runtime.constraint,
            "numerics": self.runtime.numerics,
            "noise": self.runtime.noise,
            "hybrid": self.runtime.hybrid,
        }
        runtime = EffectiveConfig(**kwargs)
        self.runtime = runtime
        if device == "gpu":
            self._deprecated_device_aliases.append("gpu")

    def set_fft_config(self, fft_config: FFTConfig) -> None:
        self.runtime = replace(self.runtime, fft=fft_config)

    def set_tfdp_config(self, tfdp_config: TFDPConfig) -> None:
        self.runtime = replace(self.runtime, tfdp=tfdp_config)

    def set_constraint_config(self, constraint_config: ConstraintConfig) -> None:
        self.runtime = replace(self.runtime, constraint=constraint_config)

    def set_numerics_config(self, numerics_config: NumericsConfig) -> None:
        self.runtime = replace(self.runtime, numerics=numerics_config)
        self.umap_epsilon = float(numerics_config.epsilon)

    def set_noise_config(self, noise_config: NoiseConfig) -> None:
        self.runtime = replace(self.runtime, noise=noise_config)

    def clear_workspace(self) -> None:
        """Drop reusable buffers owned by this estimator's execution backend."""
        if self._pipeline is not None:
            self._pipeline.clear_workspace()


_LEGACY_CONFIG_PATHS = {
    "n_components": ("umap", "n_components"),
    "n_epochs": ("umap", "n_epochs"),
    "init": ("initialization", "init"),
    "spectral_method": ("initialization", "spectral_method"),
    "spectral_tol": ("initialization", "spectral_tol"),
    "spectral_maxiter": ("initialization", "spectral_maxiter"),
    "spectral_ncv": ("initialization", "spectral_ncv"),
    "spectral_auto_defaults": ("initialization", "spectral_auto_defaults"),
    "spectral_scale_policy": ("initialization", "spectral_scale_policy"),
    "spectral_max_span": ("initialization", "spectral_max_span"),
    "spectral_jitter_relative": ("initialization", "spectral_jitter_relative"),
    "spectral_jitter_max": ("initialization", "spectral_jitter_max"),
    "report_duplicate_ratio": ("initialization", "report_duplicate_ratio"),
    "min_dist": ("umap", "min_dist"),
    "spread": ("umap", "spread"),
    "repulsion_strength": ("umap", "repulsion_strength"),
    "negative_sample_rate": ("umap", "negative_sample_rate"),
    "learning_rate": ("umap", "learning_rate"),
    "refinement_fraction": ("hybrid", "refinement_fraction"),
    "refinement_learning_rate": ("hybrid", "refinement_learning_rate"),
    "refinement_negative_sample_rate": (
        "hybrid",
        "refinement_negative_sample_rate",
    ),
    "umap_epsilon": ("umap", "epsilon"),
    "n_neighbors": ("graph", "n_neighbors"),
    "metric": ("graph", "metric"),
    "metric_kwds": ("graph", "metric_kwds"),
    "n_jobs": ("graph", "n_jobs"),
    "angular_rp_forest": ("graph", "angular_rp_forest"),
    "low_memory": ("graph", "low_memory"),
    "set_op_mix_ratio": ("graph", "set_op_mix_ratio"),
    "local_connectivity": ("graph", "local_connectivity"),
    "random_state": ("runtime", "random_state"),
    "workspace_policy": ("runtime", "workspace_policy"),
    "workspace_limit_bytes": ("runtime", "workspace_limit_bytes"),
    "p2m_mode": ("fft", "p2m_mode"),
    "fft_kernel_cache_policy": ("fft", "kernel_cache_policy"),
    "fft_kernel_cache_limit_bytes": ("fft", "kernel_cache_limit_bytes"),
    "fft_kernel_cache_max_entries": ("fft", "kernel_cache_max_entries"),
    "verbose": ("runtime", "verbose"),
    "tqdm_kwds": ("cpu_umap", "tqdm_kwds"),
    "rng_lifecycle": ("cpu_umap", "rng_lifecycle"),
    "ibfft_kernel_clip": ("fft", "kernel_clip"),
    "attraction_degree_damping_mode": ("umap", "degree_damping", "mode"),
    "attraction_degree_damping_ref": ("umap", "degree_damping", "ref"),
    "attraction_degree_damping_ref_value": ("umap", "degree_damping", "ref_value"),
    "attraction_degree_damping_power": ("umap", "degree_damping", "power"),
    "attraction_degree_damping_min_scale": ("umap", "degree_damping", "min_scale"),
    "diagnostics_path": ("diagnostics", "summary_path"),
    "diagnostics_timing_path": ("diagnostics", "timing_path"),
    "diagnostics_thresholds": ("diagnostics", "thresholds"),
    "diagnostics_topk_path": ("diagnostics", "topk_path"),
    "diagnostics_topk": ("diagnostics", "topk"),
    "diagnostics_labels": ("diagnostics", "labels"),
    "diagnostics_point_ids": ("diagnostics", "point_ids"),
    "diagnostics_memory_path": ("diagnostics", "memory_path"),
    "diagnostics_memory_epoch_stride": ("diagnostics", "memory_epoch_stride"),
    "diagnostics_memory_sample_interval_ms": ("diagnostics", "memory_sample_interval_ms"),
    "local_exact_repulsion": ("ibumap", "experimental", "local_exact_repulsion", "enabled"),
    "local_exact_k": ("ibumap", "experimental", "local_exact_repulsion", "k"),
    "local_exact_radius_factor": ("ibumap", "experimental", "local_exact_repulsion", "radius_factor"),
    "local_exact_weight": ("ibumap", "experimental", "local_exact_repulsion", "weight"),
    "local_exact_every": ("ibumap", "experimental", "local_exact_repulsion", "every"),
    "local_exact_clip": ("ibumap", "experimental", "local_exact_repulsion", "clip"),
    "local_exact_symmetric": ("ibumap", "experimental", "local_exact_repulsion", "symmetric"),
    "local_exact_start_epoch": ("ibumap", "experimental", "local_exact_repulsion", "start_epoch"),
    "local_exact_end_epoch": ("ibumap", "experimental", "local_exact_repulsion", "end_epoch"),
    "local_exact_start_frac": ("ibumap", "experimental", "local_exact_repulsion", "start_frac"),
    "local_exact_end_frac": ("ibumap", "experimental", "local_exact_repulsion", "end_frac"),
    "local_exact_density_filter": ("ibumap", "experimental", "local_exact_repulsion", "density_filter"),
    "local_exact_min_cell_count": ("ibumap", "experimental", "local_exact_repulsion", "min_cell_count"),
    "local_exact_min_cell_count_quantile": ("ibumap", "experimental", "local_exact_repulsion", "min_cell_count_quantile"),
    "local_exact_density_include_neighbor_cells": ("ibumap", "experimental", "local_exact_repulsion", "density_include_neighbor_cells"),
    "local_exact_timing_sample_size": ("ibumap", "experimental", "local_exact_repulsion", "timing_sample_size"),
    "local_density_pressure": ("ibumap", "experimental", "local_density_pressure", "enabled"),
    "local_density_pressure_weight": ("ibumap", "experimental", "local_density_pressure", "weight"),
    "local_density_pressure_every": ("ibumap", "experimental", "local_density_pressure", "every"),
    "local_density_pressure_min_count": ("ibumap", "experimental", "local_density_pressure", "min_count"),
    "local_density_pressure_clip": ("ibumap", "experimental", "local_density_pressure", "clip"),
    "local_density_pressure_power": ("ibumap", "experimental", "local_density_pressure", "power"),
    "attraction_kernel_mode": ("ibumap", "experimental", "attraction_schedule", "kernel_mode"),
    "attraction_schedule_mode": ("ibumap", "experimental", "attraction_schedule", "schedule_mode"),
    "attraction_calendar_memory_limit_bytes": (
        "ibumap",
        "experimental",
        "attraction_schedule",
        "calendar_memory_limit_bytes",
    ),
    "attraction_warp_min_degree": ("ibumap", "experimental", "attraction_schedule", "warp_min_degree"),
}

for _legacy_name, _config_path in _LEGACY_CONFIG_PATHS.items():
    setattr(IBUMAP, _legacy_name, _NestedConfigAttribute(*_config_path))
