import unittest
import csv
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numba
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

umap_module = types.ModuleType("umap")
umap_umap_module = types.ModuleType("umap.umap_")
umap_layouts_module = types.ModuleType("umap.layouts")
umap_utils_module = types.ModuleType("umap.utils")
umap_spectral_module = types.ModuleType("umap.spectral")
umap_distances_module = types.ModuleType("umap.distances")
pynndescent_module = types.ModuleType("pynndescent")
pynndescent_distances_module = types.ModuleType("pynndescent.distances")
pynndescent_sparse_module = types.ModuleType("pynndescent.sparse")
umap_module.__path__ = []
pynndescent_module.__path__ = []


def _fake_fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds,
    knn_indices,
    knn_dists,
    angular_rp_forest,
    set_op_mix_ratio,
    local_connectivity,
    apply_set_operations,
    verbose,
    densmap_or_output_dens,
):
    rows = np.repeat(np.arange(X.shape[0]), knn_indices.shape[1])
    cols = np.asarray(knn_indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(knn_dists, dtype=np.float32).reshape(-1)
    weights = np.exp(-distances).astype(np.float32)
    graph = sparse.coo_matrix((weights, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    graph = graph.maximum(graph.T).tocsr().astype(np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph, None, None, None


def _fake_nearest_neighbors(*args, **kwargs):
    raise AssertionError("nearest_neighbors should not be called in fixed-input tests")


def _fake_spectral_layout(data, graph, n_components, random_state, metric, metric_kwds):
    return random_state.normal(size=(graph.shape[0], n_components)).astype(np.float32)


def _fake_clip(value):
    if value > 4.0:
        return 4.0
    if value < -4.0:
        return -4.0
    return value


def _fake_rdist(left, right):
    diff = left - right
    return np.sum(diff * diff)


def _fake_tau_rand_int(rng_state):
    rng_state[0] = (rng_state[0] * 1664525 + 1013904223) % 2147483647
    return rng_state[0]


umap_umap_module.fuzzy_simplicial_set = _fake_fuzzy_simplicial_set
umap_umap_module.nearest_neighbors = _fake_nearest_neighbors
umap_layouts_module.clip = numba.njit(_fake_clip, cache=False)
umap_layouts_module.rdist = numba.njit(_fake_rdist, cache=False)
umap_utils_module.tau_rand_int = numba.njit(_fake_tau_rand_int, cache=False)
umap_spectral_module.spectral_layout = _fake_spectral_layout
pynndescent_distances_module.named_distances = {"euclidean": None}
pynndescent_sparse_module.sparse_named_distances = {}
umap_module.umap_ = umap_umap_module
umap_module.layouts = umap_layouts_module
umap_module.utils = umap_utils_module
umap_module.spectral = umap_spectral_module
umap_module.distances = umap_distances_module
pynndescent_module.distances = pynndescent_distances_module
pynndescent_module.sparse = pynndescent_sparse_module

_FAKE_DEPENDENT_IBUMAP_PREFIXES = (
    "ibumap.graph",
    "ibumap.pipeline",
    "ibumap.optimizers",
    "ibumap.kernels",
)


def _clear_dependent_ibumap_modules():
    for name in list(sys.modules):
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _FAKE_DEPENDENT_IBUMAP_PREFIXES
        ):
            sys.modules.pop(name, None)


def _install_fake_umap_modules():
    for name in list(sys.modules):
        if name == "umap" or name.startswith("umap."):
            del sys.modules[name]
        elif name == "pynndescent" or name.startswith("pynndescent."):
            del sys.modules[name]
    _clear_dependent_ibumap_modules()
    sys.modules["umap"] = umap_module
    sys.modules["umap.umap_"] = umap_umap_module
    sys.modules["umap.layouts"] = umap_layouts_module
    sys.modules["umap.utils"] = umap_utils_module
    sys.modules["umap.spectral"] = umap_spectral_module
    sys.modules["umap.distances"] = umap_distances_module
    sys.modules["pynndescent"] = pynndescent_module
    sys.modules["pynndescent.distances"] = pynndescent_distances_module
    sys.modules["pynndescent.sparse"] = pynndescent_sparse_module


def _clear_fake_umap_modules():
    for name in list(sys.modules):
        if name == "umap" or name.startswith("umap."):
            del sys.modules[name]
        elif name == "pynndescent" or name.startswith("pynndescent."):
            del sys.modules[name]
    _clear_dependent_ibumap_modules()
    ibumap_module = sys.modules.get("ibumap")
    if ibumap_module is not None:
        for attr in ("graph", "pipeline", "optimizers", "kernels"):
            if hasattr(ibumap_module, attr):
                delattr(ibumap_module, attr)


from ibumap import IBUMAP
import pytest

# These tests keep the historical optimize_from_graph(X, graph, init) call, which
# CUDA still requires and CPU still accepts (with a FutureWarning).
pytestmark = pytest.mark.filterwarnings(
    r"ignore:optimize_from_graph\(X, fuzzy_graph, init_embedding\) is deprecated:FutureWarning"
)



N_SAMPLES = 64
N_FEATURES = 8
N_NEIGHBORS = 8
N_EPOCHS = 4


def _fixed_inputs():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="euclidean")
    nn.fit(X)
    knn_distances, knn_indices = nn.kneighbors(X)
    graph = _fake_fuzzy_simplicial_set(
        X,
        N_NEIGHBORS,
        123,
        "euclidean",
        {},
        knn_indices,
        knn_distances,
        False,
        1.0,
        1.0,
        True,
        False,
        False,
    )[0]
    init_embedding = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, 2)).astype(np.float32)
    return X, graph, init_embedding


def _gpu_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class IBUMAPNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.graph, cls.init_embedding = _fixed_inputs()

    def _run(self, device="cpu", **kwargs):
        time.sleep(0.1)
        _clear_fake_umap_modules()
        _install_fake_umap_modules()
        random_state = kwargs.pop("random_state", 123)
        try:
            model = IBUMAP(
                algorithm="ibumap",
                device=device,
                attraction_mode="true_loss",
                repulsion_mode="true_loss",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=random_state,
                deterministic=True,
                **kwargs,
            )
            return model.optimize_from_graph(self.X, self.graph, self.init_embedding)
        finally:
            _clear_fake_umap_modules()

    def assert_embedding_ok(self, embedding):
        self.assertEqual(embedding.shape, (N_SAMPLES, 2))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())

    def test_default_and_seed_only_paths_run(self):
        baseline = self._run()
        seed_only = self._run(noise_seed=999)

        self.assert_embedding_ok(baseline)
        self.assert_embedding_ok(seed_only)
        np.testing.assert_allclose(baseline, seed_only, rtol=0.0, atol=1e-6)

    def test_explicit_ibfft_kernel_defaults_preserve_output(self):
        baseline = self._run()
        explicit_defaults = self._run(
            ibfft_kernel_clip=4.0,
            umap_epsilon=1e-3,
            repulsion_clip_norm=4.0,
            repulsion_clip_with_alpha=True,
            total_update_clip_norm=None,
            total_update_clip_with_alpha=False,
            local_exact_repulsion=False,
            local_density_pressure=False,
            ibfft_kernel_subsample_mode=None,
            ibfft_kernel_subsample_radius_cells=2,
            ibfft_kernel_subsample_points=4,
            attraction_degree_damping=True,
            attraction_degree_damping_mode="weighted_degree",
            attraction_degree_damping_ref="p99",
            attraction_degree_damping_power=0.5,
            attraction_degree_damping_min_scale=None,
        )

        np.testing.assert_allclose(baseline, explicit_defaults, rtol=0.0, atol=1e-6)

    def test_custom_ibfft_kernel_parameters_reach_kernel_construction(self):
        time.sleep(0.1)
        _clear_fake_umap_modules()
        _install_fake_umap_modules()
        captured: list[tuple[float, float, str, int, int]] = []
        try:
            model = IBUMAP(
                algorithm="ibumap",
                device="cpu",
                attraction_mode="true_loss",
                repulsion_mode="true_loss",
                n_neighbors=N_NEIGHBORS,
                n_epochs=N_EPOCHS,
                random_state=123,
                deterministic=True,
                ibfft_kernel_clip=4.0,
                umap_epsilon=1e-5,
                ibfft_kernel_subsample_mode="near_origin",
                ibfft_kernel_subsample_radius_cells=3,
                ibfft_kernel_subsample_points=8,
            )
            model._ensure_pipeline(require_graph_backend=False)
            from ibumap.optimizers import ibumap_optimizer

            original = ibumap_optimizer.ibFFT_repulsive_sampling

            def recording_ibfft(*args, **kwargs):
                captured.append(
                    (
                        float(kwargs["ibfft_kernel_clip"]),
                        float(kwargs["umap_epsilon"]),
                        kwargs["ibfft_kernel_subsample_mode"],
                        int(kwargs["ibfft_kernel_subsample_radius_cells"]),
                        int(kwargs["ibfft_kernel_subsample_points"]),
                    )
                )
                return original(*args, **kwargs)

            with patch.object(
                ibumap_optimizer,
                "ibFFT_repulsive_sampling",
                recording_ibfft,
            ):
                embedding = model.optimize_from_graph(self.X, self.graph, self.init_embedding)
        finally:
            _clear_fake_umap_modules()

        self.assert_embedding_ok(embedding)
        self.assertEqual(len(captured), N_EPOCHS)
        self.assertTrue(
            all(values == (4.0, 1e-5, "near_origin", 3, 8) for values in captured)
        )

    def test_ibfft_kernel_parameters_are_written_to_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "diagnostics.csv"
            embedding = self._run(
                ibfft_kernel_clip=2.0,
                umap_epsilon=1e-4,
                ibfft_kernel_subsample_mode="near_origin",
                ibfft_kernel_subsample_radius_cells=1,
                ibfft_kernel_subsample_points=2,
                diagnostics_path=str(diagnostics_path),
            )
            self.assert_embedding_ok(embedding)
            with diagnostics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), N_EPOCHS)
        self.assertTrue(all(float(row["ibfft_kernel_clip"]) == 2.0 for row in rows))
        self.assertTrue(all(float(row["umap_epsilon"]) == 1e-4 for row in rows))
        self.assertTrue(
            all(row["ibfft_kernel_subsample_mode"] == "near_origin" for row in rows)
        )
        self.assertTrue(
            all(int(row["ibfft_kernel_subsample_radius_cells"]) == 1 for row in rows)
        )
        self.assertTrue(
            all(int(row["ibfft_kernel_subsample_points"]) == 2 for row in rows)
        )

    def test_multicriterion_topk_and_grid_diagnostics_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "diagnostics.csv"
            topk_path = Path(tmpdir) / "topk.csv"
            embedding = self._run(
                diagnostics_path=str(diagnostics_path),
                diagnostics_topk_path=str(topk_path),
                diagnostics_topk=2,
                diagnostics_labels=["a"] * self.X.shape[0],
                diagnostics_point_ids=np.arange(self.X.shape[0]),
            )
            self.assert_embedding_ok(embedding)
            with diagnostics_path.open(newline="") as handle:
                diagnostic_rows = list(csv.DictReader(handle))
            with topk_path.open(newline="") as handle:
                topk_rows = list(csv.DictReader(handle))

        self.assertEqual(len(diagnostic_rows), N_EPOCHS)
        self.assertTrue({"robust_bbox_area", "radial_p999", "grid_cell_size", "grid_h"}.issubset(diagnostic_rows[0]))
        self.assertEqual(len(topk_rows), N_EPOCHS * 2 * 5)
        self.assertEqual(
            {row["criterion"] for row in topk_rows},
            {"total_update_norm", "attr_norm", "repl_post_clip_norm", "radius", "delta_radius"},
        )
        self.assertTrue({"previous_radius", "cos_attr_repl", "cos_total_radial_direction"}.issubset(topk_rows[0]))

    def test_disabled_local_repulsion_preserves_default_output(self):
        baseline = self._run()
        explicit_disabled = self._run(
            local_exact_repulsion=False,
            local_density_pressure=False,
        )

        np.testing.assert_allclose(baseline, explicit_disabled, rtol=0.0, atol=1e-6)

    def test_local_repulsion_modes_run(self):
        variants = (
            {"local_exact_repulsion": True, "local_exact_radius_factor": 2.0},
            {
                "local_density_pressure": True,
                "local_density_pressure_min_count": 1,
            },
            {
                "local_exact_repulsion": True,
                "local_exact_radius_factor": 2.0,
                "local_density_pressure": True,
                "local_density_pressure_min_count": 1,
            },
        )
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                self.assert_embedding_ok(self._run(**kwargs))

    def test_local_repulsion_diagnostics_fields_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "diagnostics.csv"
            embedding = self._run(
                local_exact_repulsion=True,
                local_exact_radius_factor=2.0,
                local_density_pressure=True,
                local_density_pressure_min_count=1,
                diagnostics_path=str(diagnostics_path),
            )
            self.assert_embedding_ok(embedding)
            with diagnostics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        expected_fields = {
            "local_exact_enabled",
            "local_exact_weight",
            "local_exact_k",
            "local_exact_radius",
            "local_exact_active_count",
            "local_exact_active_pct",
            "local_exact_force_max",
            "local_exact_force_mean",
            "local_exact_force_p95",
            "local_exact_force_p99",
            "local_exact_force_p999",
            "local_density_pressure_enabled",
            "local_density_pressure_weight",
            "local_density_pressure_active_count",
            "local_density_pressure_active_pct",
            "local_density_pressure_force_max",
            "local_density_pressure_force_mean",
            "local_density_pressure_force_p95",
            "local_density_pressure_force_p99",
            "local_density_pressure_force_p999",
        }
        self.assertEqual(len(rows), N_EPOCHS)
        self.assertTrue(expected_fields.issubset(rows[0]))
        self.assertEqual(int(rows[0]["local_exact_enabled"]), 1)
        self.assertEqual(int(rows[0]["local_density_pressure_enabled"]), 1)

    def test_local_exact_runtime_diagnostics_and_early_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            timing_path = Path(tmpdir) / "runtime.csv"
            baseline = self._run(
                local_exact_repulsion=True,
                local_exact_radius_factor=2.0,
                local_exact_end_frac=0.5,
                local_exact_min_cell_count=1,
            )
            embedding = self._run(
                local_exact_repulsion=True,
                local_exact_radius_factor=2.0,
                local_exact_end_frac=0.5,
                local_exact_min_cell_count=1,
                local_exact_timing_sample_size=8,
                diagnostics_timing_path=str(timing_path),
            )
            self.assert_embedding_ok(embedding)
            with timing_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        np.testing.assert_allclose(baseline, embedding, rtol=0.0, atol=1e-6)
        self.assertEqual(len(rows), N_EPOCHS)
        self.assertEqual([int(row["local_exact_ran"]) for row in rows], [1, 1, 0, 0])
        self.assertGreater(float(rows[0]["local_exact_total_time_s"]), 0.0)
        self.assertGreaterEqual(int(rows[0]["local_exact_pair_count"]), 0)
        self.assertIn("local_candidate_time_s", rows[0])
        self.assertIn("cell_occupancy_p99", rows[0])

    def test_independent_partial_clip_ranges_are_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "diagnostics.csv"
            embedding = self._run(
                repulsion_clip_norm=4.0,
                repulsion_clip_with_alpha=True,
                repulsion_clip_epoch_range=(2, 4),
                total_update_clip_norm=4.0,
                total_update_clip_with_alpha=True,
                total_update_clip_epoch_range=(0, 2),
                diagnostics_path=str(diagnostics_path),
            )
            self.assert_embedding_ok(embedding)
            with diagnostics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), N_EPOCHS)
        self.assertEqual(
            [int(row["repulsion_clip_enabled_this_epoch"]) for row in rows],
            [0, 0, 1, 1],
        )
        self.assertEqual(
            [int(row["total_update_clip_enabled_this_epoch"]) for row in rows],
            [1, 1, 0, 0],
        )
        self.assertTrue(np.isnan(float(rows[0]["repulsion_clip_effective_norm"])))
        self.assertTrue(np.isnan(float(rows[3]["total_update_clip_effective_norm"])))
        self.assertEqual(float(rows[0]["repulsion_clip_active_pct"]), 0.0)
        self.assertEqual(float(rows[3]["total_update_clip_active_pct"]), 0.0)

    def test_force_and_embedding_noise_modes_run(self):
        for noise_mode in ("force", "embedding"):
            with self.subTest(noise_mode=noise_mode):
                embedding = self._run(
                    noise_mode=noise_mode,
                    noise_scale=0.01,
                    noise_decay="linear",
                    noise_until_epoch=3,
                    noise_seed=7,
                )
                self.assert_embedding_ok(embedding)

    def test_hybrid_early_noisy_late_deterministic_runs(self):
        embedding = self._run(
            noise_mode="force",
            noise_scale=0.01,
            noise_decay="linear",
            noise_seed=11,
            hybrid_mode="early_noisy_late_deterministic",
            hybrid_switch_epoch=2,
        )
        self.assert_embedding_ok(embedding)

    def test_same_noise_seed_is_reproducible(self):
        kwargs = {
            "noise_mode": "force",
            "noise_scale": 0.01,
            "noise_decay": "constant",
            "noise_seed": 13,
        }
        emb1 = self._run(**kwargs)
        emb2 = self._run(**kwargs)

        np.testing.assert_allclose(emb1, emb2, rtol=0.0, atol=1e-6)

    def test_root_random_state_seeds_noise_when_noise_seed_is_unset(self):
        kwargs = {
            "noise_mode": "force",
            "noise_scale": 0.01,
            "noise_decay": "constant",
        }
        first = self._run(random_state=29, **kwargs)
        repeated = self._run(random_state=29, **kwargs)
        different = self._run(random_state=31, **kwargs)

        np.testing.assert_array_equal(first, repeated)
        self.assertGreater(float(np.max(np.abs(first - different))), 1e-6)

    def test_different_noise_seed_changes_embedding(self):
        kwargs = {
            "noise_mode": "force",
            "noise_scale": 0.01,
            "noise_decay": "constant",
        }
        emb1 = self._run(noise_seed=17, **kwargs)
        emb2 = self._run(noise_seed=19, **kwargs)

        self.assertGreater(float(np.max(np.abs(emb1 - emb2))), 1e-6)

    @unittest.skipUnless(_gpu_available(), "cupy/CUDA GPU is unavailable")
    def test_gpu_force_noise_smoke(self):
        embedding = self._run(
            device="cuda",
            noise_mode="force",
            noise_scale=0.01,
            noise_seed=23,
            hybrid_mode="early_noisy_late_deterministic",
            hybrid_switch_epoch=2,
        )
        self.assert_embedding_ok(embedding)


if __name__ == "__main__":
    unittest.main()
