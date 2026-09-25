from __future__ import annotations

import numpy as np

from ibumap.init._postprocess import (
    canonicalize_spectral_axes,
    compact_spectral_embedding,
    duplicate_row_stats,
)


class _ArrayModuleWithoutGroupRankOps:
    float64 = np.float64

    def __getattr__(self, name: str):
        if name == "maximum":
            raise AssertionError("maximum.accumulate must not be used")
        if name == "repeat":
            raise AssertionError("device-array repeat must not be used")
        if name == "cumsum":
            raise AssertionError("group-rank cumsum must not be used")
        return getattr(np, name)


def test_axis_canonicalization_maps_sign_flips_to_same_orientation() -> None:
    values = np.asarray(
        [[0.0, 0.0], [0.1, 0.2], [10.0, 10.0]],
        dtype=np.float32,
    )
    mirrored = 10.0 - values

    canonical, first = canonicalize_spectral_axes(np, values.copy())
    canonical_mirror, second = canonicalize_spectral_axes(
        np,
        mirrored.copy(),
    )

    np.testing.assert_allclose(
        canonical,
        canonical_mirror,
        rtol=0.0,
        atol=4 * np.spacing(np.float32(10.0)),
    )
    assert first["reflected"] == [False, False]
    assert second["reflected"] == [True, True]
    assert all(
        mean <= midpoint
        for mean, midpoint in zip(
            second["means_after"],
            second["midpoints"],
        )
    )


def test_duplicate_stats_do_not_compute_unused_group_ranks() -> None:
    array_module = _ArrayModuleWithoutGroupRankOps()
    values = np.asarray(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )

    stats = duplicate_row_stats(array_module, values)

    assert stats["unique_rows"] == 2
    assert stats["duplicate_rows"] == 1
    assert stats["duplicate_ratio"] == 1 / 3
    assert stats["max_duplicate_group"] == 2


def test_compact_spectral_embedding_never_expands_small_layout() -> None:
    values = np.asarray(
        [[-0.02415, -0.01], [0.0, 0.005], [0.02415, 0.01]],
        dtype=np.float32,
    )

    compact, diagnostics = compact_spectral_embedding(
        np,
        values.copy(),
        np.random.RandomState(11),
        max_span=0.1,
        jitter_relative=0.0,
        jitter_max=0.0,
    )

    assert diagnostics["raw_global_span"] == np.float32(0.0483)
    assert diagnostics["scale_factor"] == 1.0
    assert diagnostics["final_jitter_scale"] == 0.0
    assert diagnostics["final_global_span"] == diagnostics["raw_global_span"]
    np.testing.assert_allclose(
        np.linalg.norm(compact[:, None] - compact[None, :], axis=2),
        np.linalg.norm(values[:, None] - values[None, :], axis=2),
        rtol=2e-6,
        atol=1e-8,
    )


def test_compact_spectral_embedding_shrinks_large_layout_uniformly() -> None:
    values = np.asarray(
        [[-10.0, -2.0], [0.0, 0.0], [10.0, 2.0]],
        dtype=np.float32,
    )

    compact, diagnostics = compact_spectral_embedding(
        np,
        values.copy(),
        np.random.RandomState(13),
        max_span=0.1,
        jitter_relative=0.0,
        jitter_max=0.0,
    )

    assert diagnostics["scale_factor"] == 0.005
    assert np.isclose(diagnostics["pre_jitter_global_span"], 0.1)
    assert np.isclose(float(compact.max() - compact.min()), 0.1)


def test_compact_spectral_embedding_uses_seeded_degenerate_fallback() -> None:
    values = np.ones((8, 2), dtype=np.float32)

    first, first_diagnostics = compact_spectral_embedding(
        np,
        values.copy(),
        np.random.RandomState(17),
        max_span=0.1,
        jitter_relative=0.0,
        jitter_max=0.0,
    )
    second, second_diagnostics = compact_spectral_embedding(
        np,
        values.copy(),
        np.random.RandomState(17),
        max_span=0.1,
        jitter_relative=0.0,
        jitter_max=0.0,
    )

    np.testing.assert_array_equal(first, second)
    assert first_diagnostics["degenerate_fallback"]
    assert second_diagnostics["degenerate_fallback"]
    assert float(first.max() - first.min()) <= 0.1
