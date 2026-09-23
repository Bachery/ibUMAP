from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_WORKSPACE_ALIASES = {
    "fast": "performance",
    "low_memory": "minimal",
}


def normalize_workspace_policy(policy: str) -> str:
    resolved = _WORKSPACE_ALIASES.get(str(policy), str(policy))
    if resolved not in ("auto", "performance", "minimal"):
        raise ValueError(
            "workspace_policy must be one of: auto, performance, minimal "
            "(fast and low_memory are compatibility aliases)"
        )
    return resolved


def estimate_segmented_workspace_bytes(n_points: int, *, device: str) -> int:
    """Conservative key/order/offset/sort workspace estimate.

    CUDA groups points by box and the retained arrays therefore remain O(N),
    independent of interpolation order. The CUDA stable single-key sort avoids
    materializing point ids and a two-row lexsort key, but still needs sorting
    and segment-boundary temporary storage. Keep the existing conservative
    reserve until an absolute-memory experiment establishes a safe lower bound.
    The estimate is used only by the auto selector; actual allocations remain
    authoritative.
    """

    per_point = 24 if device == "cpu" else 48
    return max(int(n_points), 0) * per_point + 4096


@dataclass(frozen=True)
class P2MResolution:
    requested: str
    resolved: str
    reason: str
    estimated_workspace_bytes: int


def resolve_p2m_mode(
    requested: str,
    *,
    deterministic: bool,
    device: str,
    n_points: int,
    workspace_limit_bytes: Optional[int],
    atomic_available: bool = True,
) -> P2MResolution:
    requested = str(requested)
    if requested not in (
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
    if device not in ("cpu", "cuda", "metal"):
        raise ValueError("device must be 'cpu', 'cuda', or 'metal'")
    if requested == "block_atomic" and device != "cuda":
        raise ValueError("p2m_mode='block_atomic' is CUDA-only")
    if deterministic and requested in ("atomic", "block_atomic"):
        raise ValueError(
            f"deterministic=True is incompatible with p2m_mode={requested!r}"
        )

    estimate = estimate_segmented_workspace_bytes(n_points, device=device)
    within_budget = (
        workspace_limit_bytes is None
        or estimate <= int(workspace_limit_bytes)
    )

    if requested != "auto":
        if requested == "segmented" and not within_budget:
            raise MemoryError(
                "p2m_mode='segmented' requires approximately "
                f"{estimate} workspace bytes, exceeding workspace_limit_bytes="
                f"{workspace_limit_bytes}"
            )
        return P2MResolution(requested, requested, "explicit", estimate)

    # The p=1 CPU serial loop is a compact, cache-friendly O(N) stream.  The
    # native atomic path and per-epoch grouping both cost more on the supported
    # CPU backends, so auto preserves the measured faster serial implementation.
    # Both parallel modes remain available explicitly for controlled studies.
    if device == "cpu":
        return P2MResolution(
            requested, "serial", "cpu_serial_benchmark_default", estimate
        )

    if deterministic:
        if within_budget:
            return P2MResolution(
                requested, "segmented", "deterministic_parallel", estimate
            )
        return P2MResolution(
            requested, "serial", "deterministic_workspace_fallback", estimate
        )

    return P2MResolution(requested, "atomic", "maximum_throughput", estimate)
