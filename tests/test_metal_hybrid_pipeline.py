from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap import IBUMAP


class MetalHybridPipelineTest(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
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
        self.init = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )

    def test_update_reuses_graph_and_workspace_without_mutating_inputs(self):
        model = IBUMAP(
            device="metal",
            n_neighbors=2,
            n_epochs=5,
            init=self.init,
            deterministic=True,
            random_state=7,
        )
        prepared = model.prepare_fixed_inputs_from_graph(self.X, self.graph)
        X_before = self.X.copy()
        graph_before = prepared.optimizer_graph.copy()
        init_before = prepared.init_embedding.copy()
        update_init = (prepared.init_embedding * np.float32(0.5)).copy()
        update_before = update_init.copy()
        workspace_ids = []
        calendar_dtypes = []

        def fake_optimizer(init_embedding, **kwargs):
            workspace = kwargs["workspace"]
            workspace_ids.append(id(workspace))
            calendar_dtypes.append(kwargs["epochs_per_sample"].dtype)
            workspace.get_kernel("stage4", lambda: object())
            cached = workspace.get_fft_kernel("stage4")
            if cached is None:
                workspace.store_fft_kernel("stage4", object())
            return np.asarray(init_embedding) + np.float32(0.25), {
                "metal_optimizer_time_s": 0.0,
            }

        with (
            patch("ibumap.api.ensure_metal_runtime", return_value=None),
            patch(
                "ibumap.optimizers.metal_ibumap_optimizer.optimize_ibumap_metal",
                side_effect=fake_optimizer,
            ),
        ):
            model.optimize_from_prepared_graph(
                self.X,
                prepared.optimizer_graph,
                prepared.init_embedding,
            )
            state_graph = model._pipeline.state.graph
            result = model.update_embedding(init=update_init, n_epochs=5)

        self.assertIs(model._pipeline.state.graph, state_graph)
        self.assertEqual(len(set(workspace_ids)), 1)
        self.assertEqual(calendar_dtypes, [np.dtype(np.float32), np.dtype(np.float32)])
        np.testing.assert_array_equal(result, update_before + np.float32(0.25))
        np.testing.assert_array_equal(self.X, X_before)
        np.testing.assert_array_equal(prepared.init_embedding, init_before)
        np.testing.assert_array_equal(update_init, update_before)
        np.testing.assert_array_equal(
            prepared.optimizer_graph.indptr, graph_before.indptr
        )
        np.testing.assert_array_equal(
            prepared.optimizer_graph.indices, graph_before.indices
        )
        np.testing.assert_array_equal(prepared.optimizer_graph.data, graph_before.data)

        costs = model.get_time_costs()
        self.assertIn("metal_epochs_per_sample_time_s", costs)
        self.assertIn("metal_pipeline_overhead_time_s", costs)
        self.assertGreaterEqual(costs["embd_opt_time"], 0.0)
        snapshot = model._pipeline.workspace.snapshot()
        self.assertGreater(snapshot["kernel_hits"], 0)
        self.assertGreater(snapshot["fft_kernel_hits"], 0)

        model.clear_workspace()
        cleared = model._pipeline.workspace.snapshot()
        self.assertEqual(cleared["kernel_entries"], 0)
        self.assertEqual(cleared["fft_kernel_entries"], 0)


if __name__ == "__main__":
    unittest.main()
