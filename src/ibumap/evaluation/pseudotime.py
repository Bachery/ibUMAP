from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ._device import resolve_device
from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class PseudotimeCorrelationResult:
    correlation: float
    pvalue: float
    method: str
    arc_length: np.ndarray
    ordering: np.ndarray


def _validate_embedding(embedding: Any, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(embedding, dtype=dtype)
    if array.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError("embedding must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding must contain only finite values")
    return array


def _validate_pseudotime(pseudotime: Any, n_samples: int, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(pseudotime, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("pseudotime must be a 1D array")
    if array.shape[0] != n_samples:
        raise ValueError("pseudotime and embedding must contain the same number of samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("pseudotime must contain only finite values")
    return array


def _trajectory_order_from_embedding(embedding: np.ndarray) -> np.ndarray:
    centered = embedding - np.mean(embedding, axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    principal_axis = right_vectors[0]
    projection = centered @ principal_axis
    return np.argsort(projection, kind="mergesort")


def _arc_length_from_order(embedding: np.ndarray, ordering: np.ndarray) -> np.ndarray:
    ordered_embedding = embedding[ordering]
    step_lengths = np.linalg.norm(np.diff(ordered_embedding, axis=0), axis=1)
    cumulative_arc_length = np.concatenate((
        np.asarray([0.0], dtype=embedding.dtype),
        np.cumsum(step_lengths).astype(embedding.dtype, copy=False),
    ))
    arc_length = np.empty(embedding.shape[0], dtype=embedding.dtype)
    arc_length[ordering] = cumulative_arc_length
    return arc_length


def _compute_correlation(values: np.ndarray, arc_length: np.ndarray, method: str) -> tuple[float, float]:
    if method == "spearman":
        correlation, pvalue = spearmanr(values, arc_length)
    elif method == "pearson":
        correlation, pvalue = pearsonr(values, arc_length)
    else:
        raise ValueError("method must be either 'spearman' or 'pearson'")
    return float(correlation), float(pvalue)


def pseudotime_correlation(
    embedding: Any,
    pseudotime: Any,
    *,
    method: str = "spearman",
    dtype: Any = np.float32,
    device: str = "cpu",
) -> PseudotimeCorrelationResult:
    resolved_dtype = resolve_evaluation_dtype(dtype)
    embedding = _validate_embedding(embedding, resolved_dtype)
    pseudotime = _validate_pseudotime(pseudotime, embedding.shape[0], resolved_dtype)

    resolved_method = method.lower()
    resolved_device = resolve_device(device)
    if resolved_device == "gpu":
        from ._gpu import pseudotime_correlation_gpu

        return pseudotime_correlation_gpu(
            embedding,
            pseudotime,
            method=resolved_method,
        )

    ordering = _trajectory_order_from_embedding(embedding)
    arc_length = _arc_length_from_order(embedding, ordering)
    correlation, pvalue = _compute_correlation(pseudotime, arc_length, resolved_method)

    reversed_ordering = ordering[::-1]
    reversed_arc_length = _arc_length_from_order(embedding, reversed_ordering)
    reversed_correlation, reversed_pvalue = _compute_correlation(
        pseudotime,
        reversed_arc_length,
        resolved_method,
    )
    if reversed_correlation > correlation:
        ordering = reversed_ordering
        arc_length = reversed_arc_length
        correlation = reversed_correlation
        pvalue = reversed_pvalue

    return PseudotimeCorrelationResult(
        correlation=correlation,
        pvalue=pvalue,
        method=resolved_method,
        arc_length=arc_length,
        ordering=ordering,
    )
