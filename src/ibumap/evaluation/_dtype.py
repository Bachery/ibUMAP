from __future__ import annotations

from typing import Any

import numpy as np


def resolve_evaluation_dtype(dtype: Any) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be either np.float32 or np.float64")
    return resolved
