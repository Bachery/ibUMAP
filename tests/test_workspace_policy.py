import unittest

import numpy as np

from ibumap.kernels.cpu import ibfft


def _kwargs(probabilities):
    return {
        "n_interpolation_points": 1,
        "intervals_per_integer": 1.0,
        "min_num_intervals": 32,
        "gamma": 2.0,
        "paraFactor": 2.0,
        "probabilities": probabilities,
        "n_boxes_per_dim": 1.0,
        "kernel_method": "UMAP_kernel",
        "umap_a": 1.57,
        "umap_b": 0.895,
        "umap_gamma": 1.0,
        "umap_epsilon": 1e-3,
        "ibfft_kernel_clip": 4.0,
    }


class CPUWorkspacePolicyTest(unittest.TestCase):
    def setUp(self):
        self.embedding = np.random.default_rng(7).uniform(
            0.0, 10.0, size=(128, 2)
        ).astype(np.float32)

    def _active_count(self, seed, probabilities):
        return int(
            np.count_nonzero(
                np.random.RandomState(seed).random_sample(self.embedding.shape[0])
                <= probabilities
            )
        )

    def test_fast_policy_reserves_full_n_point_buffers(self):
        workspace = {}
        probabilities = np.full(self.embedding.shape[0], 0.25)

        ibfft.ibFFT_repulsive_sampling(
            self.embedding,
            _workspace=workspace,
            workspace_policy="fast",
            random_state=np.random.RandomState(11),
            **_kwargs(probabilities),
        )

        self.assertNotIn("ChargesQij", workspace)
        self.assertEqual(workspace["box_idx"].shape[0], self.embedding.shape[0])
        self.assertEqual(workspace["neg_f"].shape[0], self.embedding.shape[0])

    def test_minimal_policy_tracks_exact_active_capacity(self):
        workspace = {}
        first_probabilities = np.full(self.embedding.shape[0], 0.25)
        second_probabilities = np.full(self.embedding.shape[0], 0.75)
        third_probabilities = np.full(self.embedding.shape[0], 0.10)
        expected = []

        for seed, probabilities in (
            (11, first_probabilities),
            (12, second_probabilities),
            (13, third_probabilities),
        ):
            expected.append(self._active_count(seed, probabilities))
            ibfft.ibFFT_repulsive_sampling(
                self.embedding,
                _workspace=workspace,
                workspace_policy="minimal",
                random_state=np.random.RandomState(seed),
                **_kwargs(probabilities),
            )
            self.assertNotIn("ChargesQij", workspace)
            self.assertEqual(workspace["box_idx"].shape[0], expected[-1])
            self.assertEqual(workspace["neg_f"].shape[0], expected[-1])

        self.assertLess(workspace["box_idx"].shape[0], self.embedding.shape[0])


if __name__ == "__main__":
    unittest.main()
