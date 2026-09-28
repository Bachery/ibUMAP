from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
from ibumap.config import ConstraintConfig, NoiseConfig, NumericsConfig
from ibumap.pipeline.cpu_pipeline import CPUPipeline
from ibumap.pipeline.gpu_pipeline import GPUPipeline
import pytest

# These tests keep the historical optimize_from_graph(X, graph, init) call, which
# CUDA still requires and CPU still accepts (with a FutureWarning).
pytestmark = pytest.mark.filterwarnings(
    r"ignore:optimize_from_graph\(X, fuzzy_graph, init_embedding\) is deprecated:FutureWarning"
)



class ApiValidationTest(unittest.TestCase):
    def test_invalid_algorithm_rejected(self):
        with self.assertRaises(ValueError):
            IBUMAP(algorithm="invalid")

    def test_invalid_attraction_for_umap_rejected(self):
        with self.assertRaises(ValueError):
            IBUMAP(algorithm="umap", attraction_mode="true_loss")

    def test_invalid_repulsion_for_umap_rejected(self):
        with self.assertRaises(ValueError):
            IBUMAP(algorithm="umap", repulsion_mode="sampling")

    def test_repulsion_mode_default_is_true_loss(self):
        model = IBUMAP()
        self.assertEqual(model.runtime.algorithm, "ibumap")
        self.assertEqual(model.runtime.repulsion_mode, "true_loss")

    def test_ibumap_requires_two_components(self):
        for n_components in (1, 3):
            with self.subTest(n_components=n_components):
                with self.assertRaisesRegex(ValueError, "ibumap.*n_components=2"):
                    IBUMAP(algorithm="ibumap", n_components=n_components)

        model = IBUMAP(algorithm="umap", n_components=3)
        self.assertEqual(model.n_components, 3)

    def test_deprecated_device_alias_warns_and_is_canonicalized(self):
        with self.assertWarnsRegex(FutureWarning, "replaced by device='cuda'"):
            model = IBUMAP(device="gpu")

        self.assertEqual(model.runtime.device, "cuda")
        report = model.explain_effective_config()
        self.assertEqual(report["canonical_device"], "cuda")
        self.assertEqual(report["execution_path"], "ibumap_cuda")
        self.assertEqual(report["deprecated_device_aliases_used"], ["gpu"])

        model = IBUMAP()
        with self.assertWarnsRegex(FutureWarning, "replaced by device='cuda'"):
            model.set_runtime(device="gpu")
        self.assertEqual(model.runtime.device, "cuda")
        self.assertEqual(
            model.explain_effective_config()["deprecated_device_aliases_used"],
            ["gpu"],
        )

    def test_effective_config_reports_cpu_umap_auxiliary_path(self):
        model = IBUMAP(algorithm="umap", device="cpu")
        report = model.explain_effective_config()

        self.assertEqual(report["execution_path"], "cpu_umap_async")
        self.assertIn("umap_sgd_update_mode", report["used_parameters"])
        self.assertIn("fft_config", report["ignored_parameters"])
        self.assertFalse(report["diagnostics"]["enabled"])

    def test_umap_sgd_update_mode_api_validation(self):
        model = IBUMAP(algorithm="umap")
        self.assertEqual(model.runtime.umap_sgd_update_mode, "asynchronous")

        with self.assertRaisesRegex(ValueError, "umap_sgd_update_mode"):
            IBUMAP(algorithm="umap", umap_sgd_update_mode="invalid")
        with self.assertRaisesRegex(ValueError, "only applies to algorithm='umap'"):
            IBUMAP(umap_sgd_update_mode="synchronous")
        with self.assertRaisesRegex(NotImplementedError, "device='cpu'"):
            IBUMAP(
                algorithm="umap",
                device="cuda",
                umap_sgd_update_mode="synchronous",
            )

    def test_umap_sgd_update_mode_set_runtime_validation(self):
        model = IBUMAP(algorithm="umap")
        model.set_runtime(umap_sgd_update_mode="synchronous")
        self.assertEqual(model.runtime.umap_sgd_update_mode, "synchronous")

        with self.assertRaisesRegex(NotImplementedError, "device='cpu'"):
            model.set_runtime(device="cuda")
        with self.assertRaisesRegex(ValueError, "only applies to algorithm='umap'"):
            model.set_runtime(algorithm="ibumap")

    def test_rng_lifecycle_api_and_runtime_validation(self):
        model = IBUMAP(algorithm="umap", rng_lifecycle="shared_rng")
        self.assertEqual(model.config.cpu_umap.rng_lifecycle, "shared_rng")
        self.assertEqual(model.runtime.rng_lifecycle, "shared_rng")

        model.set_runtime(rng_lifecycle="independent")
        self.assertEqual(model.runtime.rng_lifecycle, "independent")
        with self.assertRaisesRegex(ValueError, "rng_lifecycle"):
            model.set_runtime(rng_lifecycle="invalid")

        with self.assertRaisesRegex(ValueError, "only applies to algorithm='umap'"):
            IBUMAP(rng_lifecycle="shared_rng")
        with self.assertRaisesRegex(NotImplementedError, "device='cpu'"):
            IBUMAP(
                algorithm="umap",
                device="cuda",
                rng_lifecycle="shared_rng",
            )

    def test_noise_constructor_params_are_recorded(self):
        model = IBUMAP(
            noise_mode="force",
            noise_scale=0.01,
            noise_decay="exponential",
            noise_until_epoch=10,
            noise_seed=123,
            hybrid_mode="early_noisy_late_deterministic",
            hybrid_switch_epoch=5,
        )
        self.assertEqual(model.runtime.noise.mode, "force")
        self.assertEqual(model.runtime.noise.scale, 0.01)
        self.assertEqual(model.runtime.noise.decay, "exponential")
        self.assertEqual(model.runtime.noise.until_epoch, 10)
        self.assertEqual(model.runtime.noise.seed, 123)
        self.assertEqual(model.runtime.noise.hybrid_mode, "early_noisy_late_deterministic")
        self.assertEqual(model.runtime.noise.hybrid_switch_epoch, 5)

    def test_noise_config_and_flat_params_conflict(self):
        model = IBUMAP(
            noise_config=NoiseConfig(mode="force", scale=0.01),
            noise_mode="force",
        )
        self.assertEqual(model.config.ibumap.experimental.noise.mode, "force")

        with self.assertRaises(ValueError):
            IBUMAP(
                noise_config=NoiseConfig(mode="force", scale=0.01),
                noise_mode="embedding",
            )

    def test_clip_epoch_ranges_are_recorded(self):
        model = IBUMAP(
            n_epochs=200,
            repulsion_clip_norm=4.0,
            repulsion_clip_epoch_range=[0, 100],
            total_update_clip_norm=4.0,
            total_update_clip_epoch_range=(100, 200),
        )
        self.assertEqual(model.repulsion_clip_epoch_range, (0, 100))
        self.assertEqual(model.total_update_clip_epoch_range, (100, 200))

    def test_stabilized_repulsion_defaults_are_recorded(self):
        model = IBUMAP()

        self.assertEqual(model.ibfft_kernel_clip, 4.0)
        self.assertEqual(model.umap_epsilon, 1e-3)
        self.assertEqual(model.repulsion_clip_norm, 4.0)
        self.assertTrue(model.repulsion_clip_with_alpha)
        self.assertIsNone(model.ibfft_kernel_subsample_mode)
        self.assertEqual(model.ibfft_kernel_subsample_radius_cells, 2)
        self.assertEqual(model.ibfft_kernel_subsample_points, 4)
        self.assertEqual(model.runtime.numerics.epsilon, 1e-3)
        self.assertIsNone(model.total_update_clip_norm)
        self.assertFalse(model.total_update_clip_with_alpha)
        self.assertTrue(model.attraction_degree_damping)
        self.assertEqual(model.attraction_degree_damping_mode, "weighted_degree")
        self.assertEqual(model.attraction_degree_damping_ref, "p99")
        self.assertEqual(model.attraction_degree_damping_power, 0.5)
        self.assertIsNone(model.attraction_degree_damping_min_scale)
        self.assertFalse(model.local_exact_repulsion)
        self.assertFalse(model.local_density_pressure)

    def test_custom_ibfft_kernel_parameters_are_accepted(self):
        model = IBUMAP(ibfft_kernel_clip=4.0, umap_epsilon=1e-5)

        self.assertEqual(model.ibfft_kernel_clip, 4.0)
        self.assertEqual(model.umap_epsilon, 1e-5)
        self.assertEqual(model.runtime.numerics.epsilon, 1e-5)

    def test_umap_epsilon_inherits_custom_numerics_config(self):
        model = IBUMAP(numerics_config=NumericsConfig(epsilon=0.02))

        self.assertEqual(model.umap_epsilon, 0.02)
        self.assertEqual(model.runtime.numerics.epsilon, 0.02)

    def test_float64_numerics_cpu_preserves_core_arrays(self):
        rng = np.random.RandomState(4)
        n_samples = 12
        X = rng.normal(size=(n_samples, 4)).astype(np.float64)
        init = rng.normal(size=(n_samples, 2)).astype(np.float64)
        rows = np.arange(n_samples, dtype=np.int64)
        cols = (rows + 1) % n_samples
        graph = sparse.csr_matrix(
            (
                np.ones(n_samples * 2, dtype=np.float64),
                (
                    np.concatenate([rows, cols]),
                    np.concatenate([cols, rows]),
                ),
            ),
            shape=(n_samples, n_samples),
        )

        model = IBUMAP(
            numerics_config=NumericsConfig(dtype="float64"),
            n_epochs=2,
            random_state=4,
            deterministic=True,
        )
        embedding = model.optimize_from_graph(X, graph, init)

        self.assertEqual(model._raw_data.dtype, np.float64)
        self.assertEqual(model.graph_.data.dtype, np.float64)
        self.assertEqual(model.init_embedding_.dtype, np.float64)
        self.assertEqual(embedding.dtype, np.float64)

    def test_float64_numerics_cuda_is_explicitly_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "CPU only"):
            IBUMAP(
                device="cuda",
                numerics_config=NumericsConfig(dtype="float64"),
            )

    def test_float64_numerics_umap_paths_are_explicitly_rejected(self):
        for algorithm in ("umap", "hybrid"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "require dtype='float32'",
                ):
                    IBUMAP(
                        algorithm=algorithm,
                        numerics_config=NumericsConfig(dtype="float64"),
                    )

    def test_custom_ibfft_kernel_subsampling_parameters_are_accepted(self):
        model = IBUMAP(
            ibfft_kernel_subsample_mode="near_origin",
            ibfft_kernel_subsample_radius_cells=3,
            ibfft_kernel_subsample_points=8,
        )

        self.assertEqual(model.ibfft_kernel_subsample_mode, "near_origin")
        self.assertEqual(model.ibfft_kernel_subsample_radius_cells, 3)
        self.assertEqual(model.ibfft_kernel_subsample_points, 8)

    def test_invalid_ibfft_kernel_parameters_are_rejected(self):
        for value in (0.0, -1.0):
            with self.subTest(parameter="ibfft_kernel_clip", value=value):
                with self.assertRaises(ValueError):
                    IBUMAP(ibfft_kernel_clip=value)
            with self.subTest(parameter="umap_epsilon", value=value):
                with self.assertRaises(ValueError):
                    IBUMAP(umap_epsilon=value)

        invalid_kwargs = (
            {"ibfft_kernel_subsample_mode": "all"},
            {"ibfft_kernel_subsample_radius_cells": -1},
            {"ibfft_kernel_subsample_radius_cells": 1.5},
            {"ibfft_kernel_subsample_radius_cells": True},
            {"ibfft_kernel_subsample_points": 0},
            {"ibfft_kernel_subsample_points": 1.5},
            {"ibfft_kernel_subsample_points": False},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    IBUMAP(**kwargs)

    def test_ibfft_kernel_subsampling_gpu_request_is_explicitly_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "supports CPU only"):
            IBUMAP(device="cuda", ibfft_kernel_subsample_mode="near_origin")

        model = IBUMAP(ibfft_kernel_subsample_mode="near_origin")
        with self.assertRaisesRegex(NotImplementedError, "supports CPU only"):
            model.set_runtime(device="cuda")

    def test_gpu_soft_constraints_are_explicitly_rejected(self):
        soft = ConstraintConfig(mode="soft")
        with self.assertRaisesRegex(NotImplementedError, "soft constraints.*CPU only"):
            IBUMAP(device="cuda", constraints=soft)

        model = IBUMAP(constraints=soft)
        with self.assertRaisesRegex(NotImplementedError, "soft constraints.*CPU only"):
            model.set_runtime(device="cuda")

        gpu_model = IBUMAP(device="cuda")
        with self.assertRaisesRegex(NotImplementedError, "soft constraints.*CPU only"):
            gpu_model.set_constraint_config(soft)

    def test_invalid_clip_epoch_ranges_are_rejected(self):
        invalid_values = (
            10,
            "0,100",
            (0,),
            (0, 10, 20),
            (-1, 10),
            (10, 10),
            (11, 10),
            (0.0, 10),
            (False, 10),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    IBUMAP(repulsion_clip_epoch_range=value)

    def test_attraction_degree_damping_params_are_validated(self):
        model = IBUMAP(
            attraction_degree_damping=True,
            attraction_degree_damping_mode="degree",
            attraction_degree_damping_ref="manual",
            attraction_degree_damping_ref_value=3.0,
            attraction_degree_damping_power=0.75,
            attraction_degree_damping_min_scale=0.2,
        )
        self.assertTrue(model.attraction_degree_damping)
        self.assertEqual(model.attraction_degree_damping_ref_value, 3.0)

        invalid_kwargs = (
            {"attraction_degree_damping_mode": "weighted"},
            {"attraction_degree_damping_ref": "max"},
            {"attraction_degree_damping_ref": "manual"},
            {
                "attraction_degree_damping_ref": "manual",
                "attraction_degree_damping_ref_value": 0.0,
            },
            {"attraction_degree_damping_ref_value": -1.0},
            {"attraction_degree_damping_power": 0.0},
            {"attraction_degree_damping_power": -0.5},
            {"attraction_degree_damping_min_scale": 0.0},
            {"attraction_degree_damping_min_scale": 1.1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    IBUMAP(**kwargs)

        with self.assertRaisesRegex(ValueError, "algorithm='ibumap'"):
            IBUMAP(algorithm="umap", attraction_degree_damping=True)

        model = IBUMAP(attraction_degree_damping=True)
        with self.assertRaisesRegex(ValueError, "algorithm='ibumap'"):
            model.set_runtime(algorithm="umap")

    def test_attraction_degree_damping_default_depends_on_algorithm_and_device(self):
        for device, pipeline_type in (("cpu", CPUPipeline), ("cuda", GPUPipeline)):
            with self.subTest(algorithm="ibumap", device=device):
                model = IBUMAP(algorithm="ibumap", device=device)
                self.assertTrue(model.attraction_degree_damping)
                self.assertEqual(
                    model.attraction_degree_damping_mode, "weighted_degree"
                )
                self.assertEqual(model.attraction_degree_damping_ref, "p99")
                self.assertEqual(model.attraction_degree_damping_power, 0.5)
                self.assertIsNone(model.attraction_degree_damping_min_scale)
                params = pipeline_type(model)._legacy_params(n_vertices=3)
                self.assertTrue(params["attraction_degree_damping"])
                self.assertEqual(
                    params["attraction_degree_damping_mode"], "weighted_degree"
                )
                self.assertEqual(params["attraction_degree_damping_ref"], "p99")
                self.assertEqual(params["attraction_degree_damping_power"], 0.5)

        model = IBUMAP(algorithm="umap")
        self.assertFalse(model.attraction_degree_damping)

        model = IBUMAP(
            algorithm="ibumap", attraction_degree_damping=False
        )
        self.assertFalse(model.attraction_degree_damping)

    def test_clip_epoch_range_cannot_exceed_explicit_n_epochs(self):
        with self.assertRaisesRegex(ValueError, "must not exceed n_epochs=20"):
            IBUMAP(n_epochs=20, total_update_clip_epoch_range=(0, 21))

    def test_local_repulsion_params_are_recorded(self):
        model = IBUMAP(
            local_exact_repulsion=True,
            local_exact_k=6,
            local_exact_radius_factor=0.75,
            local_exact_weight=0.2,
            local_exact_every=2,
            local_exact_clip=3.0,
            local_exact_symmetric=False,
            local_exact_end_frac=0.5,
            local_exact_min_cell_count=3,
            local_exact_density_include_neighbor_cells=True,
            local_exact_timing_sample_size=64,
            local_density_pressure=True,
            local_density_pressure_weight=0.08,
            local_density_pressure_every=3,
            local_density_pressure_min_count=2,
            local_density_pressure_clip=2.5,
            local_density_pressure_power=1.5,
        )
        self.assertTrue(model.local_exact_repulsion)
        self.assertEqual(model.local_exact_k, 6)
        self.assertEqual(model.local_exact_every, 2)
        self.assertFalse(model.local_exact_symmetric)
        self.assertEqual(model.local_exact_end_frac, 0.5)
        self.assertTrue(model.local_exact_density_filter)
        self.assertEqual(model.local_exact_min_cell_count, 3)
        self.assertTrue(model.local_exact_density_include_neighbor_cells)
        self.assertTrue(model.local_density_pressure)
        self.assertEqual(model.local_density_pressure_min_count, 2)
        self.assertEqual(model.local_density_pressure_every, 3)

    def test_invalid_local_repulsion_params_are_rejected(self):
        invalid_kwargs = (
            {"local_exact_k": 0},
            {"local_exact_radius_factor": 0.0},
            {"local_exact_weight": -0.1},
            {"local_exact_every": 0},
            {"local_exact_clip": 0.0},
            {"local_exact_start_epoch": -1},
            {"local_exact_end_epoch": 0},
            {"local_exact_start_epoch": 5, "local_exact_end_epoch": 5},
            {"local_exact_start_frac": -0.1},
            {"local_exact_end_frac": 1.1},
            {"local_exact_start_frac": 0.6, "local_exact_end_frac": 0.5},
            {"local_exact_start_epoch": 1, "local_exact_end_frac": 0.5},
            {"local_exact_min_cell_count": 0},
            {"local_exact_min_cell_count_quantile": -0.1},
            {
                "local_exact_min_cell_count": 4,
                "local_exact_min_cell_count_quantile": 0.75,
            },
            {"local_exact_timing_sample_size": 0},
            {"local_density_pressure_weight": -0.1},
            {"local_density_pressure_every": 0},
            {"local_density_pressure_min_count": 0},
            {"local_density_pressure_clip": 0.0},
            {"local_density_pressure_power": 0.0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    IBUMAP(**kwargs)

    def test_local_repulsion_gpu_request_is_explicitly_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "support CPU only"):
            IBUMAP(device="cuda", local_exact_repulsion=True)

        model = IBUMAP(local_density_pressure=True)
        with self.assertRaisesRegex(NotImplementedError, "support CPU only"):
            model.set_runtime(device="cuda")


if __name__ == "__main__":
    unittest.main()
