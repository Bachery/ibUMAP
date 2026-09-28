from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.runtime.device import DeviceError, ensure_gpu_runtime


class GpuPolicyTest(unittest.TestCase):
    def test_gpu_requires_modules(self):
        from ibumap import runtime

        with patch.object(runtime.device, "_has_module", lambda _: False):
            with self.assertRaises(DeviceError):
                ensure_gpu_runtime("cuml")


if __name__ == "__main__":
    unittest.main()
