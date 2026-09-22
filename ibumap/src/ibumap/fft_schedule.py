from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class FFTStage:
    """One coarse-to-fine ibFFT interpolation stage.

    ``start_fraction`` is measured against the complete optimization and uses
    half-open stage intervals.  A stage remains active until the next stage
    starts; interpolation orders never fall back to an earlier value.
    """

    start_fraction: float
    n_interpolation_points: int

    def __post_init__(self) -> None:
        if isinstance(self.start_fraction, bool) or not isinstance(
            self.start_fraction, Real
        ):
            raise ValueError("FFTStage.start_fraction must be a finite number")
        start_fraction = float(self.start_fraction)
        if not math.isfinite(start_fraction) or not 0.0 <= start_fraction < 1.0:
            raise ValueError("FFTStage.start_fraction must be in [0, 1)")
        if isinstance(self.n_interpolation_points, bool) or not isinstance(
            self.n_interpolation_points, Integral
        ):
            raise ValueError(
                "FFTStage.n_interpolation_points must be a positive integer"
            )
        n_interpolation_points = int(self.n_interpolation_points)
        if n_interpolation_points < 1:
            raise ValueError(
                "FFTStage.n_interpolation_points must be a positive integer"
            )
        object.__setattr__(self, "start_fraction", start_fraction)
        object.__setattr__(
            self, "n_interpolation_points", n_interpolation_points
        )


@dataclass(frozen=True, slots=True)
class ResolvedFFTStage:
    start_epoch: int
    end_epoch: int
    n_interpolation_points: int

    @property
    def n_epochs(self) -> int:
        return self.end_epoch - self.start_epoch


COARSE_TO_FINE_FFT_STAGES = (
    FFTStage(0.0, 1),
    FFTStage(0.90, 2),
    FFTStage(0.95, 3),
)


def _coerce_fft_stage(value: Any, index: int) -> FFTStage:
    if isinstance(value, FFTStage):
        return value
    if isinstance(value, Mapping):
        try:
            return FFTStage(
                start_fraction=value["start_fraction"],
                n_interpolation_points=value["n_interpolation_points"],
            )
        except KeyError as exc:
            raise ValueError(
                "FFT stage mappings require start_fraction and "
                "n_interpolation_points"
            ) from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                f"FFT stage at index {index} must contain exactly two values"
            )
        return FFTStage(value[0], value[1])
    raise ValueError(
        "FFT stages must be FFTStage instances, two-value sequences, or mappings"
    )


def normalize_fft_schedule(
    schedule: Optional[Sequence[Any]],
) -> Optional[tuple[FFTStage, ...]]:
    if schedule is None:
        return None
    if isinstance(schedule, (str, bytes)):
        raise ValueError("interpolation_schedule must be a sequence of FFT stages")
    try:
        stages = tuple(
            _coerce_fft_stage(value, index) for index, value in enumerate(schedule)
        )
    except TypeError as exc:
        raise ValueError(
            "interpolation_schedule must be a sequence of FFT stages"
        ) from exc
    if not stages:
        raise ValueError("interpolation_schedule must contain at least one stage")
    if stages[0].start_fraction != 0.0:
        raise ValueError("interpolation_schedule must start at fraction 0.0")
    for previous, current in zip(stages, stages[1:]):
        if current.start_fraction <= previous.start_fraction:
            raise ValueError(
                "FFT stage start fractions must be strictly increasing"
            )
        if current.n_interpolation_points <= previous.n_interpolation_points:
            raise ValueError(
                "FFT stage interpolation orders must be strictly increasing"
            )
    return stages


def requested_fft_schedule(
    *,
    n_interpolation_points: int,
    combine_stages: bool,
    interpolation_schedule: Optional[Sequence[Any]],
) -> tuple[FFTStage, ...]:
    normalized = normalize_fft_schedule(interpolation_schedule)
    if normalized is not None:
        if combine_stages:
            raise ValueError(
                "Use interpolation_schedule or combine_stages=True, not both"
            )
        return normalized
    if combine_stages:
        if int(n_interpolation_points) != 1:
            raise ValueError(
                "combine_stages=True requires n_interpolation_points=1; use an "
                "explicit interpolation_schedule for other starting orders"
            )
        return COARSE_TO_FINE_FFT_STAGES
    return (FFTStage(0.0, n_interpolation_points),)


def resolve_fft_schedule(
    *,
    n_epochs: int,
    n_interpolation_points: int,
    combine_stages: bool = False,
    interpolation_schedule: Optional[Sequence[Any]] = None,
) -> tuple[ResolvedFFTStage, ...]:
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, Integral):
        raise ValueError("n_epochs must be a positive integer")
    n_epochs = int(n_epochs)
    if n_epochs < 1:
        raise ValueError("n_epochs must be a positive integer")
    requested = requested_fft_schedule(
        n_interpolation_points=n_interpolation_points,
        combine_stages=bool(combine_stages),
        interpolation_schedule=interpolation_schedule,
    )
    starts = [int(math.ceil(stage.start_fraction * n_epochs)) for stage in requested]
    for index, start_epoch in enumerate(starts[1:], start=1):
        if start_epoch >= n_epochs or start_epoch <= starts[index - 1]:
            raise ValueError(
                f"n_epochs={n_epochs} is too small to realize every configured "
                "FFT interpolation stage"
            )
    return tuple(
        ResolvedFFTStage(
            start_epoch=start_epoch,
            end_epoch=(starts[index + 1] if index + 1 < len(starts) else n_epochs),
            n_interpolation_points=requested[index].n_interpolation_points,
        )
        for index, start_epoch in enumerate(starts)
    )


def validate_fft_schedule_execution(
    stages: Sequence[ResolvedFFTStage],
    *,
    device: str,
    p2m_mode: str,
) -> None:
    if device not in ("cpu", "cuda", "metal"):
        raise ValueError(
            "FFT stage execution device must be 'cpu', 'cuda', or 'metal'"
        )
    has_higher_order_stage = any(
        stage.n_interpolation_points > 1 for stage in stages
    )
    if not has_higher_order_stage:
        return
    if device == "metal" and p2m_mode not in ("auto", "atomic"):
        raise NotImplementedError(
            "Metal FFT schedules containing p>1 require p2m_mode='auto' or 'atomic'"
        )
    if device == "cpu" and p2m_mode not in ("auto", "serial"):
        raise NotImplementedError(
            f"p2m_mode={p2m_mode!r} is incompatible with an FFT schedule that "
            "contains p>1 on CPU; use 'auto' or 'serial'"
        )
    if device == "cuda" and p2m_mode == "block_atomic":
        raise NotImplementedError(
            "p2m_mode='block_atomic' is incompatible with an FFT schedule that "
            "contains p>1 on CUDA"
        )


def fft_stage_index_for_epoch(
    stages: Sequence[ResolvedFFTStage], epoch: int
) -> int:
    epoch = int(epoch)
    for index in range(len(stages) - 1, -1, -1):
        if epoch >= stages[index].start_epoch:
            return index
    raise ValueError(f"epoch={epoch} precedes the first FFT stage")


def format_resolved_fft_schedule(stages: Sequence[ResolvedFFTStage]) -> str:
    return ",".join(
        f"{stage.start_epoch}:{stage.end_epoch}:p{stage.n_interpolation_points}"
        for stage in stages
    )
