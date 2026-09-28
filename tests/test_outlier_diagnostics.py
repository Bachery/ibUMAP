import csv
import io
import unittest

import numpy as np

from ibumap.optimizers.ibumap_optimizer import (
    _embedding_bbox,
    _topk_fieldnames,
    _write_topk_rows,
)


class OutlierDiagnosticsTests(unittest.TestCase):
    def test_embedding_bbox_includes_robust_and_radial_fields(self):
        embedding = np.array([[0, 0], [1, 1], [2, 2], [100, 0]], dtype=np.float32)
        row = _embedding_bbox(embedding, False)
        self.assertGreater(row["bbox_area_expansion_ratio"], 1.0)
        self.assertIn("robust_bbox_area", row)
        self.assertIn("radial_p999", row)

    def test_topk_writes_each_diagnostic_criterion(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_topk_fieldnames())
        writer.writeheader()
        attr = np.array([[1, 0], [0, 2], [0, 0]], dtype=np.float32)
        repl = np.array([[0, 1], [1, 0], [3, 0]], dtype=np.float32)
        radii = _write_topk_rows(
            writer, 1, 0.5, attr, repl, np.array([1, 1, 3]), attr + repl, attr + repl,
            np.array([[0, 0], [1, 0], [10, 0]], dtype=np.float32),
            np.array([0.0, 1.0, 9.0]), np.array([2, 3, 1]), np.array([50, 100, 0]),
            ["a", "b", "c"], [0, 1, 2], 1, None, False,
        )
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            {row["criterion"] for row in rows},
            {"total_update_norm", "attr_norm", "repl_post_clip_norm", "radius", "delta_radius"},
        )
        self.assertEqual(radii.shape, (3,))
        self.assertIn("cos_total_radial_direction", rows[0])
        self.assertEqual(rows[0]["attraction_degree_damping_scale"], "1.0")


if __name__ == "__main__":
    unittest.main()
