import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibumap.runtime.memory import StageMemoryRecorder


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class MemoryDiagnosticsTest(unittest.TestCase):
    def test_stage_memory_recorder_writes_jsonl_and_honors_epoch_stride(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            recorder = StageMemoryRecorder(
                path,
                run_id="unit",
                entrypoint="test",
                algorithm="ibumap",
                device="cpu",
                epoch_stride=2,
            )
            try:
                recorder.record("optimizer_attraction", "begin", epoch=0)
                recorder.record("optimizer_attraction", "begin", epoch=1)
                recorder.record("optimizer_attraction", "begin", epoch=2)
            finally:
                recorder.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            epoch_rows = [
                row
                for row in rows
                if row["stage"] == "optimizer_attraction"
                and row["event"] == "begin"
            ]
            self.assertEqual([row["epoch"] for row in epoch_rows], [0, 2])
            self.assertTrue(all("cpu" in row for row in rows))
            self.assertTrue(all("ru_maxrss_bytes" in row["cpu"] for row in rows))

    def test_periodic_sampler_records_active_stage_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            recorder = StageMemoryRecorder(
                path,
                run_id="periodic",
                entrypoint="test",
                algorithm="ibumap",
                device="cpu",
                sample_interval_ms=5,
            )
            recorder.record("graph", "begin")
            time.sleep(0.05)
            recorder.record("graph", "end")
            recorder.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            periodic = [
                row
                for row in rows
                if row["stage"] == "periodic" and row["event"] == "sample"
            ]
            self.assertGreaterEqual(len(periodic), 1)
            self.assertTrue(
                any("graph" in row.get("metadata", {}).get("active_stages", []) for row in periodic)
            )
            self.assertIsNotNone(recorder._sampler_thread)
            self.assertFalse(recorder._sampler_thread.is_alive())

    def test_periodic_sampler_is_not_created_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = StageMemoryRecorder(
                Path(tmpdir) / "memory.jsonl",
                run_id="stage-only",
                entrypoint="test",
                algorithm="ibumap",
                device="cpu",
            )
            try:
                self.assertIsNone(recorder._sampler_stop)
                self.assertIsNone(recorder._sampler_thread)
            finally:
                recorder.close()

    def test_periodic_sampler_rejects_nonpositive_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "sample_interval_ms"):
                StageMemoryRecorder(
                    Path(tmpdir) / "memory.jsonl",
                    run_id="invalid",
                    entrypoint="test",
                    algorithm="ibumap",
                    device="cpu",
                    sample_interval_ms=0,
                )

    @unittest.skipUnless(_cuda_available(), "cupy/CUDA GPU is unavailable")
    def test_cuda_periodic_sampler_reads_gpu_metrics_from_background_thread(self):
        import cupy as cp

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            recorder = StageMemoryRecorder(
                path,
                run_id="cuda-periodic",
                entrypoint="test",
                algorithm="ibumap",
                device="cuda",
                sample_interval_ms=5,
            )
            allocation = cp.ones(1024, dtype=cp.float32)
            try:
                recorder.record("graph", "begin")
                time.sleep(0.05)
                recorder.record("graph", "end")
            finally:
                del allocation
                recorder.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            periodic = [row for row in rows if row["stage"] == "periodic"]
            self.assertGreaterEqual(len(periodic), 1)
            self.assertTrue(
                any("cuda_total_bytes" in row.get("gpu", {}) for row in periodic)
            )
            self.assertFalse(recorder._sampler_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
