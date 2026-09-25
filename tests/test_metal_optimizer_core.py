from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.optimizers.metal_ibumap_optimizer import (
    MetalOptimizerParameters,
    _noise_scale_for_epoch,
    _host_degree_damping,
    _validate_core_inputs,
)
from ibumap.fft_schedule import FFTStage, resolve_fft_schedule


class MetalOptimizerCoreTest(unittest.TestCase):
    def test_noise_decay_and_hybrid_switch(self):
        params = MetalOptimizerParameters(
            a=1.0,
            b=1.0,
            noise_mode="force",
            noise_scale=0.2,
            noise_decay="linear",
            hybrid_mode="early_noisy_late_deterministic",
            hybrid_switch_epoch=2,
        )
        self.assertEqual(_noise_scale_for_epoch(params, 0, 10), 0.2)
        self.assertEqual(_noise_scale_for_epoch(params, 1, 10), 0.1)
        self.assertEqual(_noise_scale_for_epoch(params, 2, 10), 0.0)
    def test_deprecated_combine_schedule_matches_explicit_schedule(self):
        combined = resolve_fft_schedule(
            n_epochs=100,
            n_interpolation_points=1,
            combine_stages=True,
        )
        explicit = resolve_fft_schedule(
            n_epochs=100,
            n_interpolation_points=1,
            interpolation_schedule=(
                FFTStage(0.0, 1), FFTStage(0.90, 2), FFTStage(0.95, 3)
            ),
        )
        self.assertEqual(combined, explicit)
    def test_host_degree_damping_matches_cpu_prefix_semantics(self):
        edgesrc = np.asarray([0, 2, 3], dtype=np.int32)
        weights = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
        params = MetalOptimizerParameters(
            a=1.0, b=1.0, degree_damping_ref="manual",
            degree_damping_ref_value=2.0,
        )
        values, ref, scale = _host_degree_damping(
            edgesrc, weights, 2, params
        )
        np.testing.assert_array_equal(values, np.asarray([4.0, 2.0], np.float32))
        self.assertEqual(ref, 2.0)
        np.testing.assert_allclose(
            scale, np.asarray([np.sqrt(0.5), 1.0], np.float32)
        )

    def test_core_input_validation_precedes_runtime_import(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            _validate_core_inputs(
                np.zeros((2, 3), dtype=np.float32),
                np.asarray([0, 0, 0]),
                np.empty(0),
                np.empty(0),
                np.ones(2),
                np.empty(0),
                2,
            )


if __name__ == "__main__":
    unittest.main()
