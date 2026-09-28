from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.runtime.workspace import MetalWorkspace


class MetalWorkspaceTest(unittest.TestCase):
    def test_kernel_and_fft_lru_stats_and_clear(self):
        workspace = MetalWorkspace(fft_kernel_cache_max_entries=2)
        first = workspace.get_kernel("k", lambda: object())
        self.assertIs(first, workspace.get_kernel("k", lambda: object()))
        workspace.store_fft_kernel("a", 1)
        workspace.store_fft_kernel("b", 2)
        self.assertEqual(workspace.get_fft_kernel("a"), 1)
        workspace.store_fft_kernel("c", 3)
        self.assertIsNone(workspace.get_fft_kernel("b"))
        snapshot = workspace.snapshot()
        self.assertEqual(snapshot["kernel_hits"], 1)
        self.assertGreater(snapshot["fft_kernel_hits"], 0)
        workspace.clear()
        self.assertEqual(workspace.snapshot()["kernel_entries"], 0)
        self.assertEqual(workspace.snapshot()["fft_kernel_entries"], 0)


if __name__ == "__main__":
    unittest.main()
