import unittest

import numpy as np

from ibumap.optimizers.ibumap_optimizer import UMAP_AttrForce_sampling


class SoftConstraintSamplingTest(unittest.TestCase):
    def _run(self, next_sample):
        embedding = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        attr_force = np.zeros_like(embedding)
        edgesrc = np.array([0, 2, 2], dtype=np.int32)
        # Self edges make the attraction contribution zero, isolating the pull.
        edgetgt = np.array([0, 0], dtype=np.int32)
        epochs_per_sample = np.ones(2, dtype=np.float64)
        epoch_of_next_sample = np.full(2, next_sample, dtype=np.float64)
        known_mask = np.array([True, False])
        known_positions = np.array([[2.0, 3.0]], dtype=np.float32)
        reverse = np.array([0, -1], dtype=np.int64)

        UMAP_AttrForce_sampling(
            attr_force,
            embedding,
            edgesrc,
            edgetgt,
            2,
            1.0,
            1.0,
            1.0,
            0,
            epochs_per_sample,
            epoch_of_next_sample,
            known_mask,
            known_positions,
            reverse,
            True,
            0.1,
        )
        return attr_force

    def test_pull_runs_once_when_no_edges_are_sampled(self):
        force = self._run(next_sample=1.0)
        np.testing.assert_allclose(force[0], [0.2, 0.3], rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(force[1], [0.0, 0.0])

    def test_pull_runs_once_when_multiple_edges_are_sampled(self):
        force = self._run(next_sample=0.0)
        np.testing.assert_allclose(force[0], [0.2, 0.3], rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(force[1], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
