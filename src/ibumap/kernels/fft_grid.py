from __future__ import annotations

import math

import numpy as np


FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE = 64


def quantize_fft_box_width(
    span: float,
    n_boxes_per_dim: int,
    dtype: np.dtype | type = np.float32,
    steps_per_octave: int = FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE,
) -> tuple[float, float, int]:
    """Round box width upward to a stable logarithmic grid.

    Returns ``(raw_width, quantized_width, level)``. The quantized width is
    represented in the requested dtype and is never smaller than the raw
    width, so the square grid continues to cover the complete coordinate span.
    """
    span = float(span)
    n_boxes_per_dim = int(n_boxes_per_dim)
    steps_per_octave = int(steps_per_octave)
    target_dtype = np.dtype(dtype)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("FFT grid span must be positive and finite")
    if n_boxes_per_dim <= 0:
        raise ValueError("n_boxes_per_dim must be positive")
    if steps_per_octave <= 0:
        raise ValueError("steps_per_octave must be positive")
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("FFT grid dtype must be float32 or float64")

    raw_width = span / n_boxes_per_dim
    scaled_log = math.log2(raw_width) * steps_per_octave
    level = int(math.ceil(scaled_log))
    quantized = math.exp2(level / steps_per_octave)
    quantized_typed = target_dtype.type(quantized)

    # A correctly rounded float32 representation can land one ULP below the
    # mathematical bin boundary. Move upward until coverage is guaranteed.
    while float(quantized_typed) < raw_width:
        quantized_typed = np.nextafter(
            quantized_typed,
            target_dtype.type(np.inf),
            dtype=target_dtype,
        )

    return raw_width, float(quantized_typed), level
