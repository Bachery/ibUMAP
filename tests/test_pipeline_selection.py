from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
class PipelineSelectionTest(unittest.TestCase):
    def test_cpu_pipeline_selection(self):
        pipeline = IBUMAP(device="cpu")._select_pipeline()
        self.assertEqual(type(pipeline).__name__, "CPUPipeline")
        self.assertEqual(type(pipeline).__module__, "ibumap.pipeline.cpu_pipeline")

    def test_cuda_pipeline_selection_remains_independent(self):
        model = IBUMAP(device="cuda")
        with patch("ibumap.api.ensure_gpu_runtime", return_value=None):
            pipeline = model._select_pipeline()
        self.assertEqual(type(pipeline).__name__, "GPUPipeline")
        self.assertEqual(type(pipeline).__module__, "ibumap.pipeline.gpu_pipeline")

    def test_metal_pipeline_selection_is_independent(self):
        model = IBUMAP(device="metal")
        with patch("ibumap.api.ensure_metal_runtime", return_value=None):
            pipeline = model._select_pipeline()
        self.assertEqual(type(pipeline).__name__, "MetalPipeline")
        self.assertEqual(type(pipeline).__module__, "ibumap.pipeline.metal_pipeline")
        self.assertEqual(model.runtime.graph_backend, "cpu")


if __name__ == "__main__":
    unittest.main()
