from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.optimizers.ibumap_optimizer import (
    _clip_enabled_for_epoch,
    _resolve_repulsion_clip_norm,
    _resolve_total_update_clip_norm,
    _validate_clip_epoch_range_for_epochs,
)


class ClipEpochScheduleTest(unittest.TestCase):
    def test_no_range_keeps_full_run_clipping(self):
        params = {
            "repulsion_clip_norm": 4.0,
            "repulsion_clip_with_alpha": True,
            "repulsion_clip_epoch_range": None,
        }
        self.assertEqual(_resolve_repulsion_clip_norm(params, 0.5, 0, 200), 2.0)
        self.assertEqual(_resolve_repulsion_clip_norm(params, 0.5, 199, 200), 2.0)

    def test_first_half_boundaries(self):
        params = {
            "total_update_clip_norm": 4.0,
            "total_update_clip_with_alpha": True,
            "total_update_clip_epoch_range": (0, 100),
        }
        self.assertEqual(_resolve_total_update_clip_norm(params, 0.5, 0, 200), 2.0)
        self.assertEqual(_resolve_total_update_clip_norm(params, 0.5, 99, 200), 2.0)
        self.assertIsNone(_resolve_total_update_clip_norm(params, 0.5, 100, 200))

    def test_last_half_boundaries(self):
        params = {
            "repulsion_clip_norm": 4.0,
            "repulsion_clip_with_alpha": False,
            "repulsion_clip_epoch_range": (100, 200),
        }
        self.assertIsNone(_resolve_repulsion_clip_norm(params, 0.5, 99, 200))
        self.assertEqual(_resolve_repulsion_clip_norm(params, 0.5, 100, 200), 4.0)
        self.assertEqual(_resolve_repulsion_clip_norm(params, 0.5, 199, 200), 4.0)

    def test_schedules_are_independent(self):
        params = {
            "repulsion_clip_norm": 4.0,
            "repulsion_clip_epoch_range": None,
            "total_update_clip_norm": 4.0,
            "total_update_clip_epoch_range": (100, 200),
        }
        self.assertEqual(_resolve_repulsion_clip_norm(params, 1.0, 50, 200), 4.0)
        self.assertIsNone(_resolve_total_update_clip_norm(params, 1.0, 50, 200))
        self.assertEqual(_resolve_repulsion_clip_norm(params, 1.0, 150, 200), 4.0)
        self.assertEqual(_resolve_total_update_clip_norm(params, 1.0, 150, 200), 4.0)

    def test_range_does_not_enable_missing_norm(self):
        self.assertFalse(_clip_enabled_for_epoch(None, (0, 100), 50))
        params = {
            "repulsion_clip_norm": None,
            "repulsion_clip_epoch_range": (0, 100),
        }
        self.assertIsNone(_resolve_repulsion_clip_norm(params, 1.0, 50, 200))

    def test_resolved_epoch_validation(self):
        self.assertEqual(
            _validate_clip_epoch_range_for_epochs((0, 200), 200, "clip_range"),
            (0, 200),
        )
        with self.assertRaisesRegex(ValueError, "resolved n_epochs"):
            _validate_clip_epoch_range_for_epochs((100, 201), 200, "clip_range")


if __name__ == "__main__":
    unittest.main()
