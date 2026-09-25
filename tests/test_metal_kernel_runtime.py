from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.kernels.metal.runtime import (
    as_metal_array,
    memory_snapshot,
    timed_call,
    to_numpy,
)


class _FakeArray:
    def __init__(self, value, dtype=None):
        self.value = np.asarray(value, dtype=dtype)

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class _FakeMX:
    float32 = np.float32

    def __init__(self):
        self.eval_count = 0
        self.sync_count = 0
        self.get_active_memory = lambda: 11
        self.get_cache_memory = lambda: 22
        self.get_peak_memory = lambda: 33

    def array(self, value, dtype=None):
        return _FakeArray(value, dtype=dtype)

    def eval(self, *values):
        self.eval_count += len(values)

    def synchronize(self):
        self.sync_count += 1


class MetalKernelRuntimeTest(unittest.TestCase):
    def test_array_and_host_conversion_use_explicit_boundary(self):
        mx = _FakeMX()
        value = as_metal_array([1, 2, 3], dtype=mx.float32, mx=mx)
        result = to_numpy(value, mx=mx)
        np.testing.assert_array_equal(result, np.asarray([1, 2, 3], dtype=np.float32))
        self.assertEqual(mx.eval_count, 1)
        self.assertEqual(mx.sync_count, 1)

    def test_timed_call_synchronizes_before_and_after(self):
        mx = _FakeMX()
        result, timing = timed_call(
            lambda: mx.array([4.0], dtype=mx.float32),
            mx=mx,
        )
        self.assertIsInstance(result, _FakeArray)
        self.assertGreaterEqual(timing.wall_time_s, 0.0)
        self.assertEqual(mx.eval_count, 1)
        self.assertEqual(mx.sync_count, 2)

    def test_memory_snapshot(self):
        self.assertEqual(
            memory_snapshot(mx=_FakeMX()),
            {"active_bytes": 11, "cache_bytes": 22, "peak_bytes": 33},
        )


if __name__ == "__main__":
    unittest.main()
