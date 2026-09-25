from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.kernels.metal.attraction import degree_damping_scale


class MetalAttractionTest(unittest.TestCase):
    def test_degree_damping_matches_host_formula(self):
        degrees = np.asarray([0.0, 1.0, 4.0, 16.0], dtype=np.float32)
        actual = degree_damping_scale(
            degrees,
            degree_ref=4.0,
            power=0.5,
            min_scale=0.4,
            mx=np,
        )
        expected = np.asarray([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_degree_damping_validates_parameters(self):
        with self.assertRaisesRegex(ValueError, "degree_ref"):
            degree_damping_scale(np.ones(2), degree_ref=0, power=0.5, min_scale=None, mx=np)
        with self.assertRaisesRegex(ValueError, "power"):
            degree_damping_scale(np.ones(2), degree_ref=1, power=0, min_scale=None, mx=np)
        with self.assertRaisesRegex(ValueError, "min_scale"):
            degree_damping_scale(np.ones(2), degree_ref=1, power=1, min_scale=2, mx=np)


if __name__ == "__main__":
    unittest.main()
