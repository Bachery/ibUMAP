from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class MetalOptionalImportTest(unittest.TestCase):
    def test_cpu_import_does_not_import_mlx(self):
        script = textwrap.dedent(
            f"""
            import importlib.abc
            import os
            import sys

            sys.path.insert(0, {str(SRC)!r})
            os.environ["IBUMAP_DISABLE_JIT_WARMUP"] = "1"

            class RejectMLX(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "mlx" or fullname.startswith("mlx."):
                        raise RuntimeError("MLX import attempted")
                    return None

            sys.meta_path.insert(0, RejectMLX())
            import ibumap
            model = ibumap.IBUMAP(device="cpu")
            assert model.runtime.device == "cpu"
            assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
