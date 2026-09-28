from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from umap.layouts import optimize_layout_euclidean as umap_learn_optimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.optimizers.umap_optimizer import (
    _make_rng_state_per_sample,
    umap_optimize_layout_euclidean,
)
from ibumap.optimizers import umap_optimizer


class UmapRngProtocolTest(unittest.TestCase):
    def setUp(self):
        self.embedding = np.asarray(
            [
                [0.25, 1.0],
                [1.5, 0.5],
                [3.25, 2.0],
                [4.75, 3.5],
            ],
            dtype=np.float32,
        )
        self.rng_state = np.asarray(
            [-538846105, 1273642420, 1935803229],
            dtype=np.int64,
        )

    def test_per_sample_state_matches_umap_learn_construction(self):
        original_rng_state = self.rng_state.copy()
        actual = _make_rng_state_per_sample(self.embedding, self.rng_state)
        expected = np.full(
            (self.embedding.shape[0], len(self.rng_state)),
            self.rng_state,
            dtype=np.int64,
        ) + self.embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

        self.assertEqual(actual.dtype, np.int64)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.rng_state, original_rng_state)

    def test_serial_optimizer_matches_umap_learn_with_identical_inputs(self):
        head = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        tail = np.asarray([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.int32)
        epochs_per_sample = np.asarray(
            [1.0, 1.5, 1.0, 2.0, 1.5, 1.0, 2.0, 1.0],
            dtype=np.float64,
        )
        n_epochs = 6
        negative_sample_rate = 3.0
        a = 1.5769434601962196
        b = 0.8950608781227859
        gamma = 1.0
        initial_alpha = 1.0

        expected = self.embedding.copy()
        expected = umap_learn_optimize(
            expected,
            expected,
            head,
            tail,
            n_epochs,
            self.embedding.shape[0],
            epochs_per_sample,
            a,
            b,
            self.rng_state.copy(),
            gamma=gamma,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            parallel=False,
            verbose=False,
            tqdm_kwds={"disable": True},
            move_other=True,
        )

        actual_rng_state = self.rng_state.copy()
        actual, _ = umap_optimize_layout_euclidean(
            self.embedding.copy(),
            head,
            tail,
            n_epochs,
            self.embedding.shape[0],
            self.embedding.shape[1],
            actual_rng_state,
            epochs_per_sample,
            epochs_per_sample / negative_sample_rate,
            parallel=False,
            verbose=False,
            tqdm_kwds={"disable": True},
            move_other=True,
            fft_params={
                "umap_initial_alpha": initial_alpha,
                "umap_a": a,
                "umap_b": b,
                "umap_gamma": gamma,
                "umap_epsilon": 0.001,
                "whether_known_points": np.zeros(
                    self.embedding.shape[0], dtype=np.bool_
                ),
                "known_points_positions": np.empty((0, 2), dtype=np.float32),
                "known_points_reverse_index": np.empty(0, dtype=np.int64),
                "soft_constraint": False,
                "constraint_weight": 0.1,
                "attr_gauss": False,
                "repl_gauss": False,
                "gauss_sigma": 1.0,
            },
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_rng_state, self.rng_state)

    def test_synchronous_optimizer_consumes_the_state_for_each_head(self):
        head = np.asarray([0, 1], dtype=np.int32)
        tail = np.asarray([1, 2], dtype=np.int32)
        epochs_per_sample = np.ones(2, dtype=np.float64)
        epochs_per_negative_sample = np.full(2, 0.5, dtype=np.float64)
        expected_states = _make_rng_state_per_sample(
            self.embedding,
            self.rng_state,
        )
        observed_states = []

        def record_state(state):
            observed_states.append(state.copy())
            state[0] += 1
            return 3

        with patch.object(
            umap_optimizer.numba,
            "njit",
            side_effect=lambda function, **kwargs: function,
        ), patch.object(
            umap_optimizer,
            "tau_rand_int",
            side_effect=record_state,
        ):
            umap_optimizer.umap_optimize_layout_euclidean_synchronous(
                self.embedding.copy(),
                head,
                tail,
                2,
                self.embedding.shape[0],
                self.embedding.shape[1],
                self.rng_state.copy(),
                epochs_per_sample,
                epochs_per_negative_sample,
                verbose=False,
                tqdm_kwds={"disable": True},
                move_other=True,
                fft_params={
                    "umap_initial_alpha": 1.0,
                    "umap_a": 1.5769434601962196,
                    "umap_b": 0.8950608781227859,
                    "umap_gamma": 1.0,
                    "umap_epsilon": 0.001,
                    "whether_known_points": np.zeros(
                        self.embedding.shape[0], dtype=np.bool_
                    ),
                    "known_points_positions": np.empty((0, 2), dtype=np.float32),
                    "known_points_reverse_index": np.empty(0, dtype=np.int64),
                    "soft_constraint": False,
                    "constraint_weight": 0.1,
                    "attr_gauss": False,
                    "repl_gauss": False,
                    "gauss_sigma": 1.0,
                    "repulsion_clip_epoch_range": None,
                },
            )

        self.assertEqual(len(observed_states), 2)
        np.testing.assert_array_equal(observed_states[0], expected_states[0])
        np.testing.assert_array_equal(observed_states[1], expected_states[1])


if __name__ == "__main__":
    unittest.main()
