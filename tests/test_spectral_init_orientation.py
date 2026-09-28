"""Spectral initialization with and without the deterministic axis reflection."""
import numpy as np
import pytest

from ibumap import IBUMAP
from ibumap.config import InitializationConfig


def initialization(x, policy):
    model = IBUMAP(algorithm="umap", device="cpu", deterministic=True, random_state=42, n_neighbors=10,
                   n_jobs=1, n_epochs=50, init="spectral", spectral_scale_policy=policy)
    bundle = model.prepare_fixed_inputs(x, device="cpu", host_output=True)
    return np.asarray(bundle.init_embedding), bundle.diagnostics["initialization"]


def test_unoriented_differs_from_default_by_axis_reflection_only():
    reflected_any = False
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = np.concatenate([rng.normal(0, 1, (120, 5)), rng.normal(4, .5, (30, 5))]).astype(np.float32)
        oriented, diag_oriented = initialization(x, "legacy_box10")
        unoriented, diag_unoriented = initialization(x, "legacy_box10_unoriented")
        assert diag_unoriented["init_spectral_scale_policy"] == "legacy_box10_unoriented"
        assert diag_unoriented["init_axis_orientation_reflected"] == [False, False]
        for axis, reflected in enumerate(diag_oriented["init_axis_orientation_reflected"]):
            if reflected:
                # low + high = 10 before the jitter, so the two differ by twice the jitter.
                assert np.max(np.abs(oriented[:, axis] + unoriented[:, axis] - 10)) < 1e-2
                reflected_any = True
            else:
                assert np.array_equal(oriented[:, axis], unoriented[:, axis])
    assert reflected_any


def test_unknown_policy_is_rejected():
    assert InitializationConfig(spectral_scale_policy="legacy_box10_unoriented").spectral_scale_policy \
        == "legacy_box10_unoriented"
    with pytest.raises(ValueError, match="spectral_scale_policy"):
        InitializationConfig(spectral_scale_policy="box10")
