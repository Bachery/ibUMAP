from __future__ import annotations

import pytest

from ibumap.graph.cpu_thread_policy import (
    CPUGraphRuntime,
    resolve_cpu_graph_thread_policy,
)


VALIDATED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "scikit-learn": "1.8.0",
    "numba": "0.61.2",
    "llvmlite": "0.44.0",
    "pynndescent": "0.5.13",
    "umap-learn": "0.5.12",
}


def _runtime(
    capacity: int = 16,
    *,
    system: str = "Linux",
    machine: str = "x86_64",
    threading_layer: str = "tbb",
    versions: dict[str, str] | None = None,
) -> CPUGraphRuntime:
    return CPUGraphRuntime(
        system=system,
        machine=machine,
        threading_layer=threading_layer,
        thread_capacity=capacity,
        versions=VALIDATED_VERSIONS if versions is None else versions,
    )


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [(16, 8), (8, 8), (7, 4), (4, 4), (3, 2), (2, 2), (1, 1)],
)
def test_deterministic_auto_selects_largest_validated_tier(
    capacity: int, expected: int
) -> None:
    policy = resolve_cpu_graph_thread_policy(
        -1,
        deterministic=True,
        algorithm="ibumap",
        device="cpu",
        runtime=_runtime(capacity),
    )
    assert policy.effective_n_jobs == expected
    assert policy.runtime_verified
    assert policy.thread_capacity == capacity


@pytest.mark.parametrize("n_jobs", [2, 4, 8])
def test_explicit_validated_thread_count_is_preserved(n_jobs: int) -> None:
    policy = resolve_cpu_graph_thread_policy(
        n_jobs,
        deterministic=True,
        algorithm="ibumap",
        device="cpu",
        runtime=_runtime(),
    )
    assert policy.effective_n_jobs == n_jobs
    assert policy.policy == "explicit_fixed_parallel"


def test_explicit_thread_count_over_capacity_is_an_error() -> None:
    with pytest.raises(ValueError, match="exceeds the runtime thread capacity"):
        resolve_cpu_graph_thread_policy(
            8,
            deterministic=True,
            algorithm="ibumap",
            device="cpu",
            runtime=_runtime(4),
        )


def test_unverified_runtime_falls_back_directly_to_serial() -> None:
    versions = dict(VALIDATED_VERSIONS)
    versions["pynndescent"] = "0.5.14"
    with pytest.warns(RuntimeWarning, match="outside the validated"):
        policy = resolve_cpu_graph_thread_policy(
            -1,
            deterministic=True,
            algorithm="ibumap",
            device="cpu",
            runtime=_runtime(versions=versions),
        )
    assert policy.effective_n_jobs == 1
    assert policy.policy == "serial_unverified_runtime"
    assert not policy.runtime_verified


def test_unvalidated_explicit_thread_count_falls_back_to_serial() -> None:
    with pytest.warns(RuntimeWarning, match="only validates explicit"):
        policy = resolve_cpu_graph_thread_policy(
            3,
            deterministic=True,
            algorithm="ibumap",
            device="cpu",
            runtime=_runtime(),
        )
    assert policy.effective_n_jobs == 1
    assert policy.policy == "serial_unvalidated_thread_count"


def test_non_ibumap_deterministic_path_remains_serial() -> None:
    policy = resolve_cpu_graph_thread_policy(
        -1,
        deterministic=True,
        algorithm="umap",
        device="cpu",
        runtime=_runtime(),
    )
    assert policy.effective_n_jobs == 1
    assert policy.policy == "serial_unvalidated_route"


def test_metal_route_remains_serial() -> None:
    policy = resolve_cpu_graph_thread_policy(
        -1,
        deterministic=True,
        algorithm="ibumap",
        device="metal",
        runtime=_runtime(),
    )
    assert policy.effective_n_jobs == 1
    assert policy.policy == "serial_unvalidated_route"


def test_fast_path_preserves_existing_n_jobs_without_runtime_probe() -> None:
    policy = resolve_cpu_graph_thread_policy(
        -1,
        deterministic=False,
        algorithm="ibumap",
        device="cpu",
    )
    assert policy.effective_n_jobs == -1
    assert policy.policy == "fast_passthrough"


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_invalid_n_jobs_is_rejected(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="positive integer or -1"):
        resolve_cpu_graph_thread_policy(
            n_jobs,
            deterministic=True,
            algorithm="ibumap",
            device="cpu",
            runtime=_runtime(),
        )
