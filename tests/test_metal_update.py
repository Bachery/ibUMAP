from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.kernels.metal.update import alpha_for_epoch, apply_optimizer_update


class MetalUpdateTest(unittest.TestCase):
    def test_alpha_matches_existing_end_of_epoch_schedule(self):
        self.assertEqual(alpha_for_epoch(1.0, 0, 10), 1.0)
        self.assertEqual(alpha_for_epoch(1.0, 1, 10), 1.0)
        self.assertAlmostEqual(alpha_for_epoch(1.0, 2, 10), 0.9)
        self.assertAlmostEqual(alpha_for_epoch(1.0, 9, 10), 0.2)

    def test_unfused_update(self):
        embedding = np.ones((2, 2), dtype=np.float32)
        attraction = np.asarray([[1, -1], [2, -2]], dtype=np.float32)
        repulsion = np.asarray([[0.5, 0.25], [-0.5, 1]], dtype=np.float32)
        actual = apply_optimizer_update(
            embedding, attraction, repulsion, mx=np
        )
        np.testing.assert_array_equal(actual, embedding + attraction + repulsion)

    def test_update_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "attraction_force"):
            apply_optimizer_update(
                np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((2, 2)), mx=np
            )

    def test_total_clip_and_hard_mask(self):
        embedding = np.zeros((2, 2), dtype=np.float32)
        attraction = np.asarray([[3, 4], [6, 8]], dtype=np.float32)
        actual = apply_optimizer_update(
            embedding,
            attraction,
            np.zeros_like(attraction),
            total_clip_norm=2.0,
            known_mask=np.asarray([True, False]),
            mx=np,
        )
        np.testing.assert_array_equal(actual[0], embedding[0])
        self.assertAlmostEqual(float(np.linalg.norm(actual[1])), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
