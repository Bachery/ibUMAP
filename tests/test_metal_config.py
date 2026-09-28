from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import (
    ConstraintConfig,
    FFTConfig,
    FFTStage,
    NoiseConfig,
    NumericsConfig,
    IBUMAP,
)
from ibumap.runtime import MetalCapabilityError, resolve_graph_backend


class MetalConfigTest(unittest.TestCase):
    def test_default_metal_contract_and_report(self):
        model = IBUMAP(device="metal")
        report = model.explain_effective_config()
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["registry_version"], "2026-07-21")
        self.assertEqual(model.runtime.device, "metal")
        self.assertEqual(report["execution_path"], "ibumap_metal")
        self.assertEqual(report["graph_device"], "cpu")
        self.assertEqual(report["init_device"], "cpu")
        self.assertEqual(report["optimizer_device"], "metal")
        self.assertEqual(report["metal_capabilities"]["backend"], "mlx")
        self.assertEqual(
            report["metal_capabilities"]["optimizer_status"], "available"
        )
        self.assertEqual(
            report["metal_capabilities"]["pipeline_status"],
            "hybrid_cpu_graph_init",
        )
        self.assertIn(
            "update_embedding",
            report["metal_capabilities"]["public_entrypoints"],
        )

    def test_metal_auto_graph_backend_resolves_to_cpu(self):
        self.assertEqual(resolve_graph_backend("metal", "auto"), "cpu")
        self.assertEqual(resolve_graph_backend("metal", "cpu"), "cpu")

    def test_rejects_cuml_graph_backend(self):
        with self.assertRaisesRegex(MetalCapabilityError, "graph_backend"):
            IBUMAP(device="metal", graph_backend="cuml")

    def test_rejects_unsupported_algorithm(self):
        with self.assertRaisesRegex(MetalCapabilityError, "ibumap"):
            IBUMAP(device="metal", algorithm="umap")

    def test_rejects_float64(self):
        with self.assertRaisesRegex(MetalCapabilityError, "float32"):
            IBUMAP(
                device="metal",
                numerics_config=NumericsConfig(dtype="float64"),
            )

    def test_accepts_higher_order_fft_and_schedule(self):
        fixed = IBUMAP(
            device="metal", fft_config=FFTConfig(n_interpolation_points=2)
        )
        self.assertEqual(fixed.runtime.fft.n_interpolation_points, 2)
        scheduled = IBUMAP(
            device="metal",
            fft_config=FFTConfig(
                interpolation_schedule=(FFTStage(0.0, 1), FFTStage(0.5, 2))
            ),
        )
        self.assertEqual(len(scheduled.runtime.fft.interpolation_schedule), 2)

    def test_rejects_deterministic_higher_order_fft(self):
        with self.assertRaisesRegex(MetalCapabilityError, "p=1"):
            IBUMAP(
                device="metal",
                deterministic=True,
                fft_config=FFTConfig(n_interpolation_points=2),
            )

    def test_rejects_unimplemented_p2m_mode(self):
        with self.assertRaisesRegex(MetalCapabilityError, "p2m_mode"):
            IBUMAP(device="metal", p2m_mode="serial")

    def test_deterministic_mode_uses_segmented_compatible_policy(self):
        model = IBUMAP(device="metal", deterministic=True)
        self.assertTrue(model.runtime.deterministic)
        with self.assertRaisesRegex((ValueError, MetalCapabilityError), "p2m_mode"):
            IBUMAP(device="metal", deterministic=True, p2m_mode="atomic")

    def test_accepts_noise(self):
        model = IBUMAP(
            device="metal",
            noise_config=NoiseConfig(mode="force", scale=0.01, seed=7),
        )
        self.assertEqual(model.runtime.noise.mode, "force")

    def test_accepts_active_hard_constraints(self):
        constraint = ConstraintConfig(
            mode="hard",
            known_points_indices=np.asarray([0]),
            known_points_positions=np.asarray([[0.0, 0.0]], dtype=np.float32),
        )
        model = IBUMAP(device="metal", constraint_config=constraint)
        self.assertEqual(model.runtime.constraint.mode, "hard")

    def test_rejects_soft_constraints(self):
        constraint = ConstraintConfig(
            mode="soft",
            known_points_indices=np.asarray([0]),
            known_points_positions=np.asarray([[0.0, 0.0]], dtype=np.float32),
        )
        with self.assertRaisesRegex(MetalCapabilityError, "soft constraints"):
            IBUMAP(device="metal", constraint_config=constraint)


if __name__ == "__main__":
    unittest.main()
