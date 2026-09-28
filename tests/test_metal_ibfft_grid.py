from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.kernels.metal.ibfft import (
    ALLOWED_N_BOXES_PER_DIM,
    fft_convolve,
    p2m_p1_atomic,
    resolve_grid_spec,
    scale_repulsive_force,
)


class MetalIbfftGridTest(unittest.TestCase):
    def test_grid_policy_matches_frozen_cpu_formula(self):
        bounds = np.asarray([-2.0, 3.0, 0.0], dtype=np.float32)
        grid = resolve_grid_spec(
            bounds,
            100,
            intervals_per_integer=1.0,
            min_num_intervals=50,
            n_boxes_per_dim=1.0,
        )
        estimate = int(
            min(
                np.sqrt(16 * 100),
                max(np.sqrt(4 * 100 / np.log(100)), 50, 5.0),
            )
        )
        expected = int(ALLOWED_N_BOXES_PER_DIM[ALLOWED_N_BOXES_PER_DIM > estimate][0])
        self.assertEqual(grid.n_boxes_per_dim, expected)
        self.assertEqual(grid.interpolation_dim, expected)
        self.assertEqual(grid.fft_dim, expected * 2)
        self.assertGreaterEqual(
            grid.box_width * grid.n_boxes_per_dim,
            grid.span,
        )

    def test_p_greater_than_one_only_changes_interpolation_shape(self):
        grid = resolve_grid_spec(
            np.asarray([-1.0, 1.0, 0.0], dtype=np.float32),
            32,
            n_interpolation_points=3,
        )
        self.assertEqual(grid.interpolation_dim, 3 * grid.n_boxes_per_dim)
        self.assertEqual(grid.fft_dim, 6 * grid.n_boxes_per_dim)

    def test_rejects_nonfinite_and_degenerate_bounds(self):
        with self.assertRaises(FloatingPointError):
            resolve_grid_spec(np.asarray([0, 1, 2], dtype=np.float32), 8)
        with self.assertRaisesRegex(ValueError, "span"):
            resolve_grid_spec(np.asarray([1, 1, 0], dtype=np.float32), 8)

    def test_rejects_single_point(self):
        with self.assertRaisesRegex(ValueError, "two points"):
            resolve_grid_spec(np.asarray([0, 1, 0], dtype=np.float32), 1)

    def test_p2m_p1_contract_uses_padded_three_channel_shape(self):
        grid = resolve_grid_spec(np.asarray([0, 1, 0], dtype=np.float32), 4)
        calls = []

        class FakeMX:
            float32 = np.float32

            @staticmethod
            def array(value):
                return np.asarray(value)

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return [np.zeros(kwargs["output_shapes"][0], dtype=np.float32)]

        result = p2m_p1_atomic(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.int32),
            grid,
            mx=FakeMX(),
            kernel=fake_kernel,
        )
        self.assertEqual(result.shape, (3, grid.fft_dim, grid.fft_dim))
        self.assertEqual(calls[0]["grid"], (4, 1, 1))
        self.assertEqual(calls[0]["init_value"], 0)

    def test_fft_convolution_rejects_shape_mismatch_before_launch(self):
        grid = resolve_grid_spec(np.asarray([0, 1, 0], dtype=np.float32), 4)
        with self.assertRaisesRegex(ValueError, "mat_w"):
            fft_convolve(
                np.zeros((2, 4, 4), dtype=np.float32),
                np.zeros((grid.fft_dim, grid.fft_dim), dtype=np.float32),
                grid,
                mx=object(),
            )

    def test_repulsion_scaling_preserves_cpu_operation_order(self):
        raw = np.asarray([[3.0, 4.0], [1.0, -2.0]], dtype=np.float32)
        effects = np.asarray([2.0, 0.5], dtype=np.float32)
        actual = scale_repulsive_force(
            raw,
            alpha=np.float32(0.25),
            neg_effects=effects,
            clip_norm=1.0,
            mx=np,
        )
        expected = raw * np.float32(0.25)
        expected *= effects[:, None]
        norms = np.linalg.norm(expected, axis=1)
        expected *= np.minimum(1.0, 1.0 / (norms + 1e-12))[:, None]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
