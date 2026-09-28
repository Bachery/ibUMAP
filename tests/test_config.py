from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.config import (
    AttractionScheduleConfig,
    ConstraintConfig,
    FFTConfig,
    NoiseConfig,
    NumericsConfig,
    EffectiveConfig,
)


class ConfigValidationTest(unittest.TestCase):
    def test_fft_config_combine_stages_default(self):
        self.assertFalse(FFTConfig().combine_stages)
        self.assertFalse(EffectiveConfig().fft.combine_stages)

    def test_fft_config_validation(self):
        with self.assertRaises(ValueError):
            FFTConfig(n_interpolation_points=0)

    def test_constraint_config_pair_validation(self):
        with self.assertRaises(ValueError):
            ConstraintConfig(known_points_indices=[0, 1], known_points_positions=None)

    def test_constraint_config_shape_validation(self):
        with self.assertRaises(ValueError):
            ConstraintConfig(
                known_points_indices=[0, 1],
                known_points_positions=np.array([[0.0, 0.0]], dtype=np.float32),
            )

    def test_numerics_dtype_validation(self):
        self.assertEqual(NumericsConfig(dtype="float64").dtype, "float64")
        with self.assertRaises(ValueError):
            NumericsConfig(dtype="float16")

    def test_numerics_default_epsilon(self):
        self.assertEqual(NumericsConfig().epsilon, 1e-3)

    def test_attraction_schedule_config_validation(self):
        default_cfg = AttractionScheduleConfig()
        self.assertEqual(default_cfg.kernel_mode, "auto")
        self.assertEqual(default_cfg.schedule_mode, "row_scan")
        self.assertEqual(default_cfg.warp_min_degree, 8)

        cfg = AttractionScheduleConfig(
            kernel_mode="warp_per_row",
            schedule_mode="active_edge_periodic",
            calendar_memory_limit_bytes=1024,
            warp_min_degree=4,
        )
        self.assertEqual(cfg.kernel_mode, "warp_per_row")
        self.assertEqual(cfg.schedule_mode, "active_edge_periodic")
        self.assertEqual(cfg.calendar_memory_limit_bytes, 1024)

        invalid_cases = [
            {"kernel_mode": "bad"},
            {"schedule_mode": "bad"},
            {"calendar_memory_limit_bytes": -1},
            {"warp_min_degree": 0},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AttractionScheduleConfig(**kwargs)

    def test_runtime_mode_validation(self):
        with self.assertRaises(ValueError):
            EffectiveConfig(algorithm="umap", attraction_mode="true_loss")

    def test_runtime_repulsion_mode_validation(self):
        with self.assertRaises(ValueError):
            EffectiveConfig(algorithm="umap", repulsion_mode="sampling")

    def test_runtime_repulsion_mode_default(self):
        cfg = EffectiveConfig()
        self.assertEqual(cfg.algorithm, "ibumap")
        self.assertEqual(cfg.repulsion_mode, "true_loss")

    def test_legacy_runtime_rejects_gpu_soft_constraints(self):
        with self.assertRaisesRegex(NotImplementedError, "soft constraints.*CPU only"):
            EffectiveConfig(
                device="cuda",
                constraint=ConstraintConfig(mode="soft"),
            )

    def test_deprecated_device_alias_is_canonicalized(self):
        with self.assertWarnsRegex(FutureWarning, "replaced by device='cuda'"):
            cfg = EffectiveConfig(device="gpu")
        self.assertEqual(cfg.device, "cuda")

    def test_umap_sgd_update_mode_default(self):
        cfg = EffectiveConfig(algorithm="umap")
        self.assertEqual(cfg.umap_sgd_update_mode, "asynchronous")

    def test_umap_sgd_update_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "umap_sgd_update_mode"):
            EffectiveConfig(algorithm="umap", umap_sgd_update_mode="invalid")

        with self.assertRaisesRegex(ValueError, "only applies to algorithm='umap'"):
            EffectiveConfig(
                algorithm="ibumap",
                umap_sgd_update_mode="synchronous",
            )

        with self.assertRaisesRegex(NotImplementedError, "device='cpu'"):
            EffectiveConfig(
                algorithm="umap",
                device="cuda",
                umap_sgd_update_mode="synchronous",
            )

    def test_rng_lifecycle_validation(self):
        self.assertEqual(EffectiveConfig(algorithm="umap").rng_lifecycle, "independent")
        self.assertEqual(
            EffectiveConfig(
                algorithm="umap",
                rng_lifecycle="shared_rng",
            ).rng_lifecycle,
            "shared_rng",
        )
        with self.assertRaisesRegex(ValueError, "rng_lifecycle"):
            EffectiveConfig(algorithm="umap", rng_lifecycle="invalid")
        with self.assertRaisesRegex(ValueError, "only applies to algorithm='umap'"):
            EffectiveConfig(rng_lifecycle="shared_rng")
        with self.assertRaisesRegex(NotImplementedError, "device='cpu'"):
            EffectiveConfig(
                algorithm="umap",
                device="cuda",
                rng_lifecycle="shared_rng",
            )

    def test_noise_config_validation(self):
        with self.assertRaises(ValueError):
            NoiseConfig(mode="bad")
        with self.assertRaises(ValueError):
            NoiseConfig(scale=-1.0)
        with self.assertRaises(ValueError):
            NoiseConfig(decay="bad")
        with self.assertRaises(ValueError):
            NoiseConfig(hybrid_mode="bad")

    def test_runtime_noise_default(self):
        cfg = EffectiveConfig()
        self.assertEqual(cfg.noise.mode, "none")
        self.assertEqual(cfg.noise.scale, 0.0)
        self.assertEqual(cfg.noise.hybrid_mode, "none")

    def test_noise_rejected_for_non_ibumap_algorithm(self):
        with self.assertRaises(ValueError):
            EffectiveConfig(algorithm="umap", noise=NoiseConfig(mode="force", scale=0.01))


if __name__ == "__main__":
    unittest.main()
