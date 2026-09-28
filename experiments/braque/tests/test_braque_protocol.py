"""Small correctness checks for label semantics, display alignment and integrity checks."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

# Other experiments also have a scripts/_common.py; import this experiment's copy.
sys.modules.pop("_common", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import completed, label_metrics, match_labels, rigid_align, safe_name, seal, verify, write_json  # noqa: E402


class ProtocolChecks(unittest.TestCase):
    def test_label_permutation_is_not_a_change(self):
        a = np.array([-1, 0, 0, 1, 1])
        b = np.array([-1, 9, 9, 5, 5])
        mapped, _ = match_labels(a, b)
        np.testing.assert_array_equal(mapped, a)
        self.assertEqual(label_metrics(a, b)["assignment_disagreement"], 0)

    def test_noise_is_reserved_and_changes_partition_exactly(self):
        a = np.array([0, 0, 0, 0, 1, 1, -1, -1])
        b = np.array([2, 2, 3, 3, 4, -1, 4, -1])
        m = label_metrics(a, b)
        self.assertEqual(m["noise_status_disagreement"], 2 / 8)
        self.assertEqual(m["assignment_disagreement"], 4 / 8)
        self.assertEqual(m["cluster_to_cluster_disagreement"] + m["noise_status_disagreement"], m["assignment_disagreement"])
        self.assertEqual(m["cluster_to_noise_count"], 1)
        self.assertEqual(m["noise_to_cluster_count"], 1)

    def test_all_noise_and_no_shared_nonnoise(self):
        a = np.full(5, -1)
        b = np.array([0, 0, 1, 1, 1])
        m = label_metrics(a, b)
        self.assertEqual(m["assignment_disagreement"], 1)
        self.assertIsNone(m["ari_both_nonnoise"])
        self.assertEqual(label_metrics(a, a)["assignment_disagreement"], 0)

    def test_rigid_alignment_preserves_scale_and_pairwise_distances(self):
        a = np.array([[0., 0.], [2., 0.], [0., 1.], [3., 4.]])
        rotation = np.array([[0., 1.], [1., 0.]])  # reflection allowed
        b = a @ rotation + [9., -3.]
        aligned, _, _ = rigid_align(a, b)
        np.testing.assert_allclose(aligned, a, atol=1e-12)
        doubled, _, _ = rigid_align(a, 2 * b)
        np.testing.assert_allclose(pdist(doubled), pdist(2 * b), atol=1e-12)
        self.assertFalse(np.allclose(doubled, a))

    def test_tampered_artifact_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("original")
            seal(root, ["a.txt"])
            verify(root)
            (root / "a.txt").write_text("changed")
            with self.assertRaises(ValueError):
                verify(root)

    def test_directory_traversal_rejected(self):
        for name in ("../old_results", "/tmp/old_results", "..", "a/b"):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_interrupted_unsealed_attempt_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "runs/umap_seeded/repeat_000/attempt_001"
            attempt.mkdir(parents=True)
            write_json(attempt / "run.json", {"status": "completed", "contract_sha256": "abc",
                                              "profile": "umap_seeded", "repeat": 0})
            self.assertIsNone(completed(root, "umap_seeded", 0, "abc"))
            seal(attempt, ["run.json"])
            self.assertIsNotNone(completed(root, "umap_seeded", 0, "abc"))


if __name__ == "__main__":
    unittest.main()
