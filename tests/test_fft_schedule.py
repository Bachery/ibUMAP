import unittest

from ibumap import FFTConfig, FFTStage
from ibumap.fft_schedule import (
    fft_stage_index_for_epoch,
    format_resolved_fft_schedule,
    resolve_fft_schedule,
    validate_fft_schedule_execution,
)


class FFTScheduleTest(unittest.TestCase):
    def test_combine_stages_resolves_to_persistent_coarse_to_fine_stages(self):
        stages = resolve_fft_schedule(
            n_epochs=200,
            n_interpolation_points=1,
            combine_stages=True,
        )

        self.assertEqual(
            [
                (stage.start_epoch, stage.end_epoch, stage.n_interpolation_points)
                for stage in stages
            ],
            [(0, 180, 1), (180, 190, 2), (190, 200, 3)],
        )
        orders = [
            stages[fft_stage_index_for_epoch(stages, epoch)].n_interpolation_points
            for epoch in range(200)
        ]
        self.assertEqual(orders, [1] * 180 + [2] * 10 + [3] * 10)
        self.assertEqual(
            format_resolved_fft_schedule(stages),
            "0:180:p1,180:190:p2,190:200:p3",
        )

    def test_fixed_order_is_one_full_length_stage(self):
        stages = resolve_fft_schedule(
            n_epochs=17,
            n_interpolation_points=2,
        )

        self.assertEqual(len(stages), 1)
        self.assertEqual(
            (stages[0].start_epoch, stages[0].end_epoch), (0, 17)
        )
        self.assertEqual(stages[0].n_interpolation_points, 2)

    def test_explicit_schedule_is_normalized_by_fft_config(self):
        config = FFTConfig(
            interpolation_schedule=(
                (0.0, 1),
                {"start_fraction": 0.75, "n_interpolation_points": 2},
                FFTStage(0.90, 4),
            )
        )

        self.assertIsInstance(config.interpolation_schedule, tuple)
        self.assertTrue(
            all(isinstance(stage, FFTStage) for stage in config.interpolation_schedule)
        )
        stages = resolve_fft_schedule(
            n_epochs=100,
            n_interpolation_points=config.n_interpolation_points,
            interpolation_schedule=config.interpolation_schedule,
        )
        self.assertEqual(
            [stage.n_interpolation_points for stage in stages], [1, 2, 4]
        )
        self.assertEqual([stage.start_epoch for stage in stages], [0, 75, 90])

    def test_invalid_or_ambiguous_schedules_fail_early(self):
        invalid_schedules = (
            (),
            ((0.1, 1),),
            ((0.0, 1), (0.5, 1)),
            ((0.0, 2), (0.5, 1)),
            ((0.0, 1), (0.5, 2), (0.5, 3)),
        )
        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule):
                with self.assertRaises(ValueError):
                    FFTConfig(interpolation_schedule=schedule)

        with self.assertRaisesRegex(ValueError, "not both"):
            FFTConfig(
                combine_stages=True,
                interpolation_schedule=((0.0, 1), (0.5, 2)),
            )
        with self.assertRaisesRegex(ValueError, "requires n_interpolation_points=1"):
            FFTConfig(n_interpolation_points=2, combine_stages=True)
        with self.assertRaisesRegex(ValueError, "too small"):
            resolve_fft_schedule(
                n_epochs=19,
                n_interpolation_points=1,
                combine_stages=True,
            )

    def test_higher_order_p2m_incompatibilities_are_preflighted(self):
        stages = resolve_fft_schedule(
            n_epochs=20,
            n_interpolation_points=1,
            combine_stages=True,
        )

        validate_fft_schedule_execution(stages, device="cpu", p2m_mode="auto")
        validate_fft_schedule_execution(stages, device="cpu", p2m_mode="serial")
        validate_fft_schedule_execution(stages, device="cuda", p2m_mode="segmented")
        with self.assertRaisesRegex(NotImplementedError, "p>1 on CPU"):
            validate_fft_schedule_execution(
                stages, device="cpu", p2m_mode="segmented"
            )
        with self.assertRaisesRegex(NotImplementedError, "p>1 on CUDA"):
            validate_fft_schedule_execution(
                stages, device="cuda", p2m_mode="block_atomic"
            )
        validate_fft_schedule_execution(stages, device="metal", p2m_mode="auto")
        validate_fft_schedule_execution(stages, device="metal", p2m_mode="atomic")
        with self.assertRaisesRegex(NotImplementedError, "Metal"):
            validate_fft_schedule_execution(
                stages, device="metal", p2m_mode="segmented"
            )


if __name__ == "__main__":
    unittest.main()
