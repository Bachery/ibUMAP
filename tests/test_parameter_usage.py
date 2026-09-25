from pathlib import Path
import importlib
import inspect
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
from ibumap.parameter_usage import build_parameter_usage_registry
from ibumap.utils import prepare_known_point_arrays


class ParameterUsageRegistryTest(unittest.TestCase):
    def test_effective_config_handles_disabled_optional_clipping_group(self):
        model = IBUMAP(
            algorithm="umap",
            umap_sgd_update_mode="synchronous",
            repulsion_clip_norm=None,
            total_update_clip_norm=None,
        )

        report = model.explain_effective_config()
        used = report["used_parameters"]

        self.assertIsNone(used["repulsion_clip_norm"])
        self.assertIsNone(used["repulsion_clip_epoch_range"])
        self.assertIsNone(used["umap.repulsion_clipping.norm"])
        self.assertIsNone(used["umap.repulsion_clipping.with_alpha"])
        self.assertIsNone(used["umap.repulsion_clipping.epoch_range"])

    def test_registry_covers_constructor_and_required_fields(self):
        records = build_parameter_usage_registry()
        by_name = {record["name"]: record for record in records}
        constructor_names = {
            name
            for name in inspect.signature(IBUMAP.__init__).parameters
            if name != "self"
        }

        self.assertEqual(len(constructor_names), 122)
        self.assertTrue(constructor_names.issubset(by_name))
        self.assertIn("initialization.init", by_name)
        self.assertEqual(by_name["init"]["canonical_name"], "initialization.init")
        self.assertEqual(
            by_name["spectral_scale_policy"]["canonical_name"],
            "initialization.spectral_scale_policy",
        )
        self.assertEqual(by_name["n_neighbors"]["canonical_name"], "graph.n_neighbors")
        self.assertTrue(by_name["n_neighbors"]["is_legacy_alias"])
        self.assertEqual(by_name["fit.y"]["status"], "unused")
        self.assertEqual(by_name["deterministic"]["status"], "supported")
        self.assertEqual(by_name["rng_lifecycle"]["status"], "supported")
        self.assertEqual(
            by_name["rng_lifecycle"]["canonical_name"],
            "cpu_umap.rng_lifecycle",
        )
        self.assertEqual(
            by_name["NumericsConfig.deterministic"]["status"], "unused"
        )
        self.assertEqual(
            by_name["diagnostics_memory_path"]["canonical_name"],
            "diagnostics.memory_path",
        )
        self.assertEqual(
            by_name["diagnostics_memory_epoch_stride"]["canonical_name"],
            "diagnostics.memory_epoch_stride",
        )
        self.assertEqual(
            by_name["diagnostics_memory_sample_interval_ms"]["canonical_name"],
            "diagnostics.memory_sample_interval_ms",
        )
        self.assertEqual(
            by_name["attraction_kernel_mode"]["canonical_name"],
            "ibumap.experimental.attraction_schedule.kernel_mode",
        )
        self.assertEqual(
            by_name["fft_kernel_cache_policy"]["canonical_name"],
            "fft.kernel_cache_policy",
        )
        self.assertEqual(
            by_name["fft_kernel_cache_limit_bytes"]["canonical_name"],
            "fft.kernel_cache_limit_bytes",
        )
        self.assertEqual(
            by_name["fft_kernel_cache_max_entries"]["canonical_name"],
            "fft.kernel_cache_max_entries",
        )
        self.assertEqual(
            by_name["refinement_fraction"]["canonical_name"],
            "hybrid.refinement_fraction",
        )
        for record in records:
            for required in (
                "name",
                "default",
                "category",
                "used_by",
                "ignored_by",
                "affects_result_by_default",
                "affects_runtime_by_default",
                "status",
                "notes",
                "source_locations",
                "canonical_name",
                "legacy_names",
                "is_legacy_alias",
            ):
                self.assertIn(required, record)

    def test_gpu_clip_avoids_host_copy_without_diagnostics(self):
        optimizer = importlib.import_module("ibumap.optimizers.ibumap_optimizer")
        asnumpy = Mock(side_effect=lambda value: np.asarray(value))

        class FakeCupy:
            float32 = np.float32
            sqrt = staticmethod(np.sqrt)
            sum = staticmethod(np.sum)
            minimum = staticmethod(np.minimum)

        FakeCupy.asnumpy = asnumpy

        force = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        with patch.object(optimizer, "cupy", FakeCupy):
            result = optimizer._clip_force_norm_inplace(
                force, 1.0, True, return_norms=False
            )

        self.assertIsNone(result)
        asnumpy.assert_not_called()
        self.assertLessEqual(float(np.linalg.norm(force, axis=1).max()), 1.0 + 1e-6)

    def test_disabled_constraints_use_empty_payload_arrays(self):
        mask, positions, reverse = prepare_known_point_arrays(100, 2, None, None)
        self.assertEqual(mask.shape, (100,))
        self.assertEqual(positions.shape, (0, 2))
        self.assertEqual(reverse.shape, (0,))


if __name__ == "__main__":
    unittest.main()
