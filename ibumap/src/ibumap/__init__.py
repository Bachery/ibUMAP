from .api import UMAPFFT
from .fft_schedule import FFTStage
from .jit_warmup import get_jit_warmup_status, warmup_jit
from .config import (
    CPUUMAPConfig,
    FFTConfig,
    ForceClippingConfig,
    DegreeDampingConfig,
    GraphConfig,
    RuntimeConfig,
    UMAPConfig,
    InitializationConfig,
    HybridOptimizerConfig,
    IBUMAPConfig,
    IBUMAPExperimentalConfig,
    AttractionScheduleConfig,
    LocalExactRepulsionConfig,
    LocalDensityPressureConfig,
    KernelSubsamplingConfig,
    DiagnosticsConfig,
    TFDPConfig,
    ConstraintConfig,
    NoiseConfig,
    NumericsConfig,
    UMAPSGDUpdateMode,
    RNGLifecycle,
    P2MMode,
    SpectralScalePolicy,
    WorkspacePolicy,
    UMAPFFTConfig,
    UMAPFFTConfigBundle,
)

__all__ = [
    "UMAPFFT",
    "PreparedInputs",
    "RuntimeConfig",
    "GraphConfig",
    "UMAPConfig",
    "InitializationConfig",
    "HybridOptimizerConfig",
    "FFTConfig",
    "FFTStage",
    "warmup_jit",
    "get_jit_warmup_status",
    "ForceClippingConfig",
    "DegreeDampingConfig",
    "IBUMAPConfig",
    "IBUMAPExperimentalConfig",
    "AttractionScheduleConfig",
    "LocalExactRepulsionConfig",
    "LocalDensityPressureConfig",
    "KernelSubsamplingConfig",
    "CPUUMAPConfig",
    "DiagnosticsConfig",
    "TFDPConfig",
    "ConstraintConfig",
    "NoiseConfig",
    "NumericsConfig",
    "UMAPSGDUpdateMode",
    "RNGLifecycle",
    "P2MMode",
    "SpectralScalePolicy",
    "WorkspacePolicy",
    "UMAPFFTConfig",
    "UMAPFFTConfigBundle",
]


def __getattr__(name: str):
    # Keep the historical lightweight ``import umap_fft`` behavior: graph and
    # initializer dependencies are imported only when preparation is used.
    if name == "PreparedInputs":
        from .prepared import PreparedInputs

        return PreparedInputs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
