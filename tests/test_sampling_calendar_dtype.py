from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from umap.umap_ import make_epochs_per_sample as umap_make_epochs_per_sample

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.utils import make_epochs_per_sample, make_epochs_per_sample_gpu


class SamplingCalendarDtypeTest(unittest.TestCase):
    def setUp(self):
        self.weights = np.asarray(
            [1.0, 0.9, 0.7, 0.3, 0.1, 0.0],
            dtype=np.float32,
        )

    def test_host_default_matches_umap_learn_float64_calendar(self):
        actual = make_epochs_per_sample(self.weights, 200)
        expected = umap_make_epochs_per_sample(self.weights, 200)

        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_explicit_float64_uses_float64_arithmetic(self):
        actual = make_epochs_per_sample(
            self.weights,
            200,
            dtype=np.float64,
        )
        expected = umap_make_epochs_per_sample(self.weights, 200)

        np.testing.assert_array_equal(actual, expected)

    def test_accelerator_host_staging_can_remain_float32(self):
        actual = make_epochs_per_sample(
            self.weights,
            200,
            dtype=np.float32,
        )

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual[-1], np.float32(-1.0))

    def test_gpu_calendar_remains_float32(self):
        fake_cupy = SimpleNamespace(
            float32=np.float32,
            maximum=np.maximum,
            ones=np.ones,
        )
        with patch.dict(sys.modules, {"cupy": fake_cupy}):
            actual = make_epochs_per_sample_gpu(self.weights, 200)

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual[-1], np.float32(-1.0))


if __name__ == "__main__":
    unittest.main()
