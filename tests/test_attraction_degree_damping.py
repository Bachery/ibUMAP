import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from ibumap import IBUMAP
from ibumap.optimizers.ibumap_optimizer import (
    _compute_attraction_degree_damping_scale,
)
import pytest

# These tests keep the historical optimize_from_graph(X, graph, init) call, which
# CUDA still requires and CPU still accepts (with a FutureWarning).
pytestmark = pytest.mark.filterwarnings(
    r"ignore:optimize_from_graph\(X, fuzzy_graph, init_embedding\) is deprecated:FutureWarning"
)



class AttractionDegreeDampingScaleTest(unittest.TestCase):
    def setUp(self):
        # CSR source rows: weighted degrees [5, 4, 1, 0], degrees [2, 1, 1, 0].
        self.edgesrc = np.array([0, 2, 3, 4, 4], dtype=np.int32)
        self.weights = np.array([4.0, 1.0, 4.0, 1.0], dtype=np.float32)

    def test_weighted_degree_manual_scale(self):
        degrees, degree_ref, scale = _compute_attraction_degree_damping_scale(
            self.edgesrc,
            self.weights,
            4,
            mode="weighted_degree",
            ref="manual",
            ref_value=1.0,
            power=0.5,
        )

        np.testing.assert_allclose(degrees, [5.0, 4.0, 1.0, 0.0])
        self.assertEqual(degree_ref, 1.0)
        np.testing.assert_allclose(
            scale,
            [np.sqrt(1.0 / 5.0), 0.5, 1.0, 1.0],
            rtol=1e-6,
        )

    def test_unweighted_degree_scale_and_floor(self):
        degrees, _, scale = _compute_attraction_degree_damping_scale(
            self.edgesrc,
            self.weights,
            4,
            mode="degree",
            ref="manual",
            ref_value=1.0,
            power=0.5,
            min_scale=0.8,
        )

        np.testing.assert_array_equal(degrees, [2.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(scale, [0.8, 1.0, 1.0, 1.0])


class AttractionDegreeDampingSmokeTest(unittest.TestCase):
    def test_cpu_smoke_and_diagnostics(self):
        rng = np.random.RandomState(7)
        n = 20
        X = rng.normal(size=(n, 4)).astype(np.float32)
        init = rng.normal(size=(n, 2)).astype(np.float32)

        # Symmetric ring plus a weighted hub gives a non-trivial p95 reference.
        graph = sparse.lil_matrix((n, n), dtype=np.float32)
        for i in range(n):
            j = (i + 1) % n
            graph[i, j] = graph[j, i] = 0.5
        for j in range(1, n):
            graph[0, j] = graph[j, 0] = 1.0
        graph = graph.tocsr()

        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics_path = Path(tmpdir) / "damping.csv"
            model = IBUMAP(
                algorithm="ibumap",
                device="cpu",
                n_epochs=2,
                random_state=7,
                deterministic=True,
                attraction_degree_damping=True,
                attraction_degree_damping_ref="p95",
                attraction_degree_damping_power=0.5,
                diagnostics_path=str(diagnostics_path),
            )
            embedding = model.optimize_from_graph(X, graph, init)

            self.assertEqual(embedding.shape, (n, 2))
            self.assertTrue(np.isfinite(embedding).all())
            with diagnostics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        required = {
            "attraction_degree_damping_enabled",
            "attraction_degree_damping_degree_ref",
            "attraction_degree_damping_degree_p99",
            "attraction_degree_damping_scale_min",
            "attraction_degree_damping_active_count",
            "attraction_degree_damping_active_pct",
            "attr_norm_pre_degree_damping_max",
            "attr_norm_post_degree_damping_max",
        }
        self.assertTrue(required.issubset(rows[0]))
        self.assertEqual(rows[0]["attraction_degree_damping_enabled"], "1")
        self.assertGreater(
            float(rows[0]["attraction_degree_damping_active_count"]), 0.0
        )
        self.assertLessEqual(
            float(rows[0]["attr_norm_post_degree_damping_max"]),
            float(rows[0]["attr_norm_pre_degree_damping_max"]) + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
