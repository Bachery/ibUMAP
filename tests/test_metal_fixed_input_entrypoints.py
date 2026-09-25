from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP
class MetalFixedInputEntrypointsTest(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        self.init = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        self.graph = csr_matrix(
            np.asarray(
                [
                    [0.0, 1.0, 0.8, 0.0],
                    [1.0, 0.0, 0.0, 0.8],
                    [0.8, 0.0, 0.0, 1.0],
                    [0.0, 0.8, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        self.model = IBUMAP(
            device="metal",
            n_neighbors=2,
            n_epochs=5,
            init=self.init,
            random_state=7,
        )

    def test_preparation_preserves_host_artifacts(self):
        prepared = self.model.prepare_fixed_inputs_from_graph(self.X, self.graph)
        self.assertEqual(prepared.device, "metal")
        self.assertTrue(prepared.host_output)
        self.assertTrue(isspmatrix_csr(prepared.optimizer_graph))
        self.assertIsInstance(prepared.init_embedding, np.ndarray)
        self.assertEqual(prepared.effective_config["preparation"]["device"], "metal")

    def test_metal_host_output_false_is_explicitly_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "host_output=False"):
            self.model.prepare_fixed_inputs_from_graph(
                self.X,
                self.graph,
                host_output=False,
            )

    def test_prepared_graph_dispatches_to_metal_optimizer_boundary(self):
        prepared = self.model.prepare_fixed_inputs_from_graph(self.X, self.graph)
        original_init = prepared.init_embedding.copy()

        def fake_optimizer(init_embedding, **kwargs):
            self.assertTrue(isspmatrix_csr(prepared.optimizer_graph))
            self.assertEqual(kwargs["model"].runtime.device, "metal")
            return init_embedding + np.float32(0.25), {"metal_fake_time": 0.001}

        with (
            patch("ibumap.api.ensure_metal_runtime", return_value=None),
            patch(
                "ibumap.optimizers.metal_ibumap_optimizer.optimize_ibumap_metal",
                side_effect=fake_optimizer,
            ) as optimizer,
        ):
            result = self.model.optimize_from_prepared_graph(
                self.X,
                prepared.optimizer_graph,
                prepared.init_embedding,
            )

        optimizer.assert_called_once()
        self.assertEqual(type(self.model._pipeline).__name__, "MetalPipeline")
        self.assertEqual(
            type(self.model._pipeline).__module__,
            "ibumap.pipeline.metal_pipeline",
        )
        self.assertEqual(self.model.runtime.graph_backend, "cpu")
        np.testing.assert_allclose(result, original_init + 0.25)
        np.testing.assert_array_equal(prepared.init_embedding, original_init)
        self.model.clear_workspace()

    def test_stage_three_optimizer_boundary_is_enabled(self):
        prepared = self.model.prepare_fixed_inputs_from_graph(self.X, self.graph)
        expected = prepared.init_embedding + np.float32(0.125)

        class CoreResult:
            embedding = expected
            timings = {"metal_optimizer_time_s": 0.01}

        with (
            patch("ibumap.api.ensure_metal_runtime", return_value=None),
            patch(
                "ibumap.optimizers.metal_ibumap_optimizer.run_metal_optimizer_core",
                return_value=CoreResult(),
            ) as core,
        ):
            actual = self.model.optimize_from_prepared_graph(
                self.X,
                prepared.optimizer_graph,
                prepared.init_embedding,
            )
        core.assert_called_once()
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
