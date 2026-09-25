from pathlib import Path
import unittest

import numpy as np

from ibumap.optimizers.ibumap_optimizer import (
    UMAP_AttrForce,
    UMAP_AttrForce_sampling,
    UMAP_AttrForce_sampling_calendar,
    UMAP_AttrForce_sampling_periodic,
    _build_sampling_attraction_calendar,
    _build_sampling_attraction_periodic,
    _resolve_attraction_kernel_mode,
    _resolve_attraction_schedule_mode,
    _zero_calendar_attr_rows,
    _warp_per_row_launch,
)


ROOT = Path(__file__).resolve().parents[1]


def _between(source, start, end):
    return source.split(start, 1)[1].split(end, 1)[0]


def _old_gradient(distance_squared, a, b):
    coefficient = -2.0 * a * b * pow(distance_squared, b - 1.0)
    return coefficient / (a * pow(distance_squared, b) + 1.0)


def _reference_force(embedding, edgesrc, edgetgt, a, b, alpha, weights=None):
    result = np.zeros_like(embedding)
    for i in range(embedding.shape[0]):
        for k in range(edgesrc[i], edgesrc[i + 1]):
            j = edgetgt[k]
            move = embedding[i] - embedding[j]
            distance_squared = float(np.dot(move, move))
            coefficient = (
                _old_gradient(distance_squared, a, b)
                if distance_squared > 0.0
                else 0.0
            )
            if weights is not None:
                coefficient *= weights[k]
            result[i] += alpha * np.clip(coefficient * move, -4.0, 4.0)
    return result


class AttractionKernelOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.embedding = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-0.5, 0.25]],
            dtype=np.float32,
        )
        self.edgesrc = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
        self.edgetgt = np.asarray([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.int32)
        self.weights = np.asarray(
            [0.25, 1.0, 0.5, 0.9, 0.75, 0.4, 0.6, 0.8], dtype=np.float32
        )
        self.a = 1.576943460
        self.b = 0.895060879
        self.alpha = 0.7
        self.known_mask = np.zeros(self.embedding.shape[0], dtype=np.bool_)
        self.known_positions = np.empty((0, 2), dtype=np.float32)
        self.known_reverse = np.full(self.embedding.shape[0], -1, dtype=np.int64)

    def test_true_loss_matches_double_power_reference(self):
        force = np.full_like(self.embedding, 123.0)
        UMAP_AttrForce(
            force,
            self.embedding,
            self.edgesrc,
            self.edgetgt,
            self.weights,
            self.embedding.shape[0],
            2,
            self.a,
            self.b,
            self.alpha,
            self.known_mask,
            self.known_positions,
            self.known_reverse,
        )

        expected = _reference_force(
            self.embedding,
            self.edgesrc,
            self.edgetgt,
            self.a,
            self.b,
            self.alpha,
            self.weights,
        )
        np.testing.assert_allclose(force, expected, rtol=2e-6, atol=2e-7)

    def test_sampling_matches_double_power_reference(self):
        force = np.full_like(self.embedding, 123.0)
        epochs_per_sample = np.ones(self.edgetgt.shape[0], dtype=np.float64)
        epoch_of_next_sample = np.zeros(self.edgetgt.shape[0], dtype=np.float64)
        UMAP_AttrForce_sampling(
            force,
            self.embedding,
            self.edgesrc,
            self.edgetgt,
            self.embedding.shape[0],
            self.a,
            self.b,
            self.alpha,
            0,
            epochs_per_sample,
            epoch_of_next_sample,
            self.known_mask,
            self.known_positions,
            self.known_reverse,
        )

        expected = _reference_force(
            self.embedding,
            self.edgesrc,
            self.edgetgt,
            self.a,
            self.b,
            self.alpha,
        )
        np.testing.assert_allclose(force, expected, rtol=2e-6, atol=2e-7)

    def test_sampling_calendar_matches_row_scan_schedule(self):
        n_epochs = 6
        epochs_per_sample = np.asarray(
            [1.0, 2.5, 0.5, 3.0, 10.0, 1.5, 2.0, 4.0],
            dtype=np.float32,
        )
        (
            epoch_row_offsets,
            calendar_rows,
            calendar_edge_offsets,
            calendar_edges,
            stats,
        ) = _build_sampling_attraction_calendar(
            self.edgesrc,
            epochs_per_sample,
            n_epochs,
            memory_limit_bytes=1024 * 1024,
        )
        self.assertGreater(stats["events"], 0)
        self.assertGreater(stats["active_rows"], 0)
        for key in (
            "build_count_time",
            "build_fill_time",
            "build_sort_time",
            "build_group_time",
        ):
            self.assertIn(key, stats)
            self.assertGreaterEqual(stats[key], 0.0)

        row_force = np.full_like(self.embedding, np.nan)
        calendar_force = np.zeros_like(self.embedding)
        epoch_of_next_sample = epochs_per_sample.copy()
        for epoch in range(n_epochs):
            UMAP_AttrForce_sampling(
                row_force,
                self.embedding,
                self.edgesrc,
                self.edgetgt,
                self.embedding.shape[0],
                self.a,
                self.b,
                self.alpha,
                epoch,
                epochs_per_sample,
                epoch_of_next_sample,
                self.known_mask,
                self.known_positions,
                self.known_reverse,
            )
            if epoch > 0:
                _zero_calendar_attr_rows(
                    calendar_force,
                    calendar_rows,
                    epoch_row_offsets[epoch - 1],
                    epoch_row_offsets[epoch],
                )
            UMAP_AttrForce_sampling_calendar(
                calendar_force,
                self.embedding,
                calendar_rows,
                calendar_edge_offsets,
                calendar_edges,
                self.edgetgt,
                epoch_row_offsets[epoch],
                epoch_row_offsets[epoch + 1],
                self.a,
                self.b,
                self.alpha,
            )
            np.testing.assert_allclose(calendar_force, row_force, rtol=2e-6, atol=2e-7)

    def test_sampling_periodic_matches_row_scan_schedule(self):
        n_epochs = 6
        epochs_per_sample = np.asarray(
            [1.0, 2.5, 0.5, 3.0, 10.0, 1.5, 2.0, 4.0],
            dtype=np.float32,
        )
        (
            bucket_offsets,
            bucket_edges,
            bucket_edge_sources,
            bucket_due_epochs,
            stats,
        ) = _build_sampling_attraction_periodic(
            self.edgesrc,
            epochs_per_sample,
            n_epochs,
            memory_limit_bytes=1024 * 1024,
        )
        self.assertGreater(stats["events"], 0)
        self.assertGreater(stats["build_bucket_count"], 0)
        self.assertEqual(stats["build_sort_time"], 0.0)
        self.assertGreaterEqual(stats["build_schedule_time"], 0.0)

        row_force = np.full_like(self.embedding, np.nan)
        periodic_force = np.zeros_like(self.embedding)
        row_epoch_marks = np.zeros(self.embedding.shape[0], dtype=np.int32)
        epoch_of_next_sample = epochs_per_sample.copy()
        total_events = 0
        total_active_rows = 0
        for epoch in range(n_epochs):
            UMAP_AttrForce_sampling(
                row_force,
                self.embedding,
                self.edgesrc,
                self.edgetgt,
                self.embedding.shape[0],
                self.a,
                self.b,
                self.alpha,
                epoch,
                epochs_per_sample,
                epoch_of_next_sample,
                self.known_mask,
                self.known_positions,
                self.known_reverse,
            )
            periodic_force.fill(0.0)
            active_events, active_rows = UMAP_AttrForce_sampling_periodic(
                periodic_force,
                self.embedding,
                bucket_offsets,
                bucket_edges,
                bucket_edge_sources,
                bucket_due_epochs,
                row_epoch_marks,
                self.edgetgt,
                epoch,
                self.a,
                self.b,
                self.alpha,
            )
            total_events += int(active_events)
            total_active_rows += int(active_rows)
            np.testing.assert_allclose(periodic_force, row_force, rtol=2e-6, atol=2e-7)
        self.assertEqual(total_events, stats["events"])
        self.assertGreater(total_active_rows, 0)

    def test_periodic_build_avoids_full_event_sort(self):
        n_vertices = 32
        degree = 4
        edge_count = n_vertices * degree
        edgesrc = np.arange(0, edge_count + 1, degree, dtype=np.int32)
        intervals = np.asarray([4.0, 8.0, 16.0, 32.0], dtype=np.float32)
        epochs_per_sample = np.resize(intervals, edge_count).astype(np.float32)
        n_epochs = 64

        *_, calendar_stats = _build_sampling_attraction_calendar(
            edgesrc,
            epochs_per_sample,
            n_epochs,
            memory_limit_bytes=1024 * 1024,
        )
        *_, periodic_stats = _build_sampling_attraction_periodic(
            edgesrc,
            epochs_per_sample,
            n_epochs,
            memory_limit_bytes=1024 * 1024,
        )

        self.assertEqual(periodic_stats["events"], calendar_stats["events"])
        self.assertEqual(periodic_stats["build_sort_time"], 0.0)
        self.assertLess(
            periodic_stats["estimated_bytes"],
            calendar_stats["estimated_bytes"],
        )
        self.assertEqual(periodic_stats["build_bucket_count"], len(intervals))

    def test_sampling_calendar_memory_guard(self):
        epochs_per_sample = np.ones(self.edgetgt.shape[0], dtype=np.float32)
        with self.assertRaises(MemoryError):
            _build_sampling_attraction_calendar(
                self.edgesrc,
                epochs_per_sample,
                n_epochs=4,
                memory_limit_bytes=1,
            )

    def test_cpu_and_cuda_kernels_each_use_one_power_per_active_edge(self):
        cpu_source = (
            ROOT / "src" / "ibumap" / "optimizers" / "ibumap_optimizer.py"
        ).read_text(encoding="utf-8")
        exact_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce(",
            "def UMAP_AttrForce_sampling(",
        )
        sampling_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling(",
            "def _count_sampling_calendar_events",
        )
        calendar_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling_calendar(",
            "def UMAP_AttrForce_sampling_periodic(",
        )
        periodic_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling_periodic(",
            "def UMAP_ReplForce(",
        )
        for source in (exact_cpu, sampling_cpu, calendar_cpu, periodic_cpu):
            self.assertEqual(source.count("pow(dist_squared"), 1)
            self.assertIn("distance_power_b / dist_squared", source)

        cuda_source = (
            ROOT / "src" / "ibumap" / "kernels" / "gpu" / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        exact_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_cu(",
            "void UMAP_AttrForce_sampling_cu(",
        )
        sampling_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_sampling_cu(",
            "__inline__ __device__ float ibumap_warp_sum",
        )
        exact_warp_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_warp_per_row_cu(",
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
        )
        sampling_warp_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
            "void UMAP_ApplyForce_cu(",
        )
        for source in (exact_cuda, sampling_cuda, exact_warp_cuda, sampling_warp_cuda):
            self.assertEqual(source.count("powf(dsqare"), 1)
            self.assertIn("distance_power_b / dsqare", source)

    def test_kernels_overwrite_rows_and_optimizer_skips_buffer_clear(self):
        cpu_source = (
            ROOT / "src" / "ibumap" / "optimizers" / "ibumap_optimizer.py"
        ).read_text(encoding="utf-8")
        exact_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce(",
            "def UMAP_AttrForce_sampling(",
        )
        sampling_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling(",
            "def _count_sampling_calendar_events",
        )
        calendar_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling_calendar(",
            "def UMAP_AttrForce_sampling_periodic(",
        )
        periodic_cpu = _between(
            cpu_source,
            "def UMAP_AttrForce_sampling_periodic(",
            "def UMAP_ReplForce(",
        )
        attraction_loop = cpu_source.split("### Attractive force", 1)[1].split(
            "### Repulsive force", 1
        )[0]
        for source in (exact_cpu, sampling_cpu):
            self.assertIn("attr_force[i][0] = temp0", source)
            self.assertIn("attr_force[i][1] = temp1", source)
        self.assertIn("attr_force[i][0] = temp0", calendar_cpu)
        self.assertIn("attr_force[i][1] = temp1", calendar_cpu)
        self.assertIn("attr_force[i][0] += ", periodic_cpu)
        self.assertIn("attr_force[i][1] += ", periodic_cpu)
        row_scan_branch = attraction_loop.split(
            'elif attraction_schedule_mode == "active_edge_periodic":', 1
        )[1].split("else:", 1)[1].split("attraction_thread_per_row_epochs += 1", 1)[0]
        self.assertNotIn("attr_force.fill", row_scan_branch)

        cuda_source = (
            ROOT / "src" / "ibumap" / "kernels" / "gpu" / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        exact_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_cu(",
            "void UMAP_AttrForce_sampling_cu(",
        )
        sampling_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_sampling_cu(",
            "__inline__ __device__ float ibumap_warp_sum",
        )
        exact_warp_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_warp_per_row_cu(",
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
        )
        sampling_warp_cuda = _between(
            cuda_source,
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
            "void UMAP_ApplyForce_cu(",
        )
        for source in (exact_cuda, sampling_cuda, exact_warp_cuda, sampling_warp_cuda):
            self.assertIn("attr_force[2 * i] = temp0", source)
            self.assertIn("attr_force[2 * i + 1] = temp1", source)

    def test_cuda_warp_per_row_kernels_use_warp_reduction_without_atomics(self):
        cuda_source = (
            ROOT / "src" / "ibumap" / "kernels" / "gpu" / "RawCudafloat.cu"
        ).read_text(encoding="utf-8")
        helper = _between(
            cuda_source,
            "__inline__ __device__ float ibumap_warp_sum",
            "void UMAP_AttrForce_warp_per_row_cu(",
        )
        exact_warp = _between(
            cuda_source,
            "void UMAP_AttrForce_warp_per_row_cu(",
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
        )
        sampling_warp = _between(
            cuda_source,
            "void UMAP_AttrForce_sampling_warp_per_row_cu(",
            "void UMAP_ApplyForce_cu(",
        )

        self.assertIn("__shfl_down_sync", helper)
        for source in (exact_warp, sampling_warp):
            self.assertIn("const int lane = threadIdx.x & 31", source)
            self.assertIn("const int warp_in_block = threadIdx.x >> 5", source)
            self.assertIn("if (lane == 0)", source)
            self.assertNotIn("atomicAdd", source)

    def test_attraction_kernel_mode_resolution(self):
        self.assertEqual(_resolve_attraction_schedule_mode("auto"), "row_scan")
        self.assertEqual(
            _resolve_attraction_schedule_mode("active_edge_periodic"),
            "active_edge_periodic",
        )
        self.assertEqual(
            _resolve_attraction_kernel_mode(
                "auto", is_gpu=True, average_degree=15.0, warp_min_degree=8
            ),
            "warp_per_row",
        )
        self.assertEqual(
            _resolve_attraction_kernel_mode(
                "auto", is_gpu=True, average_degree=4.0, warp_min_degree=8
            ),
            "thread_per_row",
        )
        self.assertEqual(
            _resolve_attraction_kernel_mode(
                "auto", is_gpu=False, average_degree=15.0, warp_min_degree=8
            ),
            "thread_per_row",
        )
        with self.assertRaises(NotImplementedError):
            _resolve_attraction_kernel_mode(
                "warp_per_row", is_gpu=False, average_degree=15.0, warp_min_degree=8
            )
        with self.assertRaises(ValueError):
            _resolve_attraction_schedule_mode("bad")

        grid, block = _warp_per_row_launch(17, threads_per_block=128)
        self.assertEqual(block, (128,))
        self.assertEqual(grid, (5,))


if __name__ == "__main__":
    unittest.main()
