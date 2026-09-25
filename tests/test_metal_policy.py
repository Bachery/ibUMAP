from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
from ibumap.runtime.metal import MetalDeviceError, ensure_metal_runtime


class MetalPolicyTest(unittest.TestCase):
    def test_requires_macos(self):
        with patch("ibumap.runtime.metal.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(MetalDeviceError, "requires macOS"):
                ensure_metal_runtime()

    def test_requires_arm64_python(self):
        with (
            patch("ibumap.runtime.metal.platform.system", return_value="Darwin"),
            patch("ibumap.runtime.metal.platform.machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(MetalDeviceError, "arm64"):
                ensure_metal_runtime()

    def test_requires_optional_mlx_dependency(self):
        with (
            patch("ibumap.runtime.metal.platform.system", return_value="Darwin"),
            patch("ibumap.runtime.metal.platform.machine", return_value="arm64"),
            patch("ibumap.runtime.metal.importlib.util.find_spec", return_value=None),
        ):
            with self.assertRaisesRegex(MetalDeviceError, r"ibumap\[metal\]"):
                ensure_metal_runtime()

    def test_returns_runtime_info_for_available_device(self):
        fake_mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
        with (
            patch("ibumap.runtime.metal.platform.system", return_value="Darwin"),
            patch("ibumap.runtime.metal.platform.machine", return_value="arm64"),
            patch("ibumap.runtime.metal.importlib.util.find_spec", return_value=object()),
            patch("ibumap.runtime.metal._mlx_version", return_value="0.32.0"),
            patch("ibumap.runtime.metal.importlib.import_module", return_value=fake_mx),
        ):
            info = ensure_metal_runtime()
        self.assertTrue(info.metal_available)
        self.assertEqual(info.backend, "mlx")
        self.assertEqual(info.backend_version, "0.32.0")

    def test_pipeline_does_not_silently_fallback(self):
        model = IBUMAP(device="metal")
        error = MetalDeviceError("no Metal device")
        with patch("ibumap.api.ensure_metal_runtime", side_effect=error):
            with self.assertRaisesRegex(MetalDeviceError, "no Metal device"):
                model._select_pipeline()


if __name__ == "__main__":
    unittest.main()
