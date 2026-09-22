from __future__ import annotations

import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Optional


def _ru_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _read_kb_map(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except OSError:
        return {}
    return values


def _cpu_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {"ru_maxrss_bytes": _ru_maxrss_bytes()}
    smaps = _read_kb_map(Path("/proc/self/smaps_rollup"))
    if smaps:
        payload.update(
            {
                "rss_bytes": smaps.get("Rss"),
                "pss_bytes": smaps.get("Pss"),
                "private_clean_bytes": smaps.get("Private_Clean"),
                "private_dirty_bytes": smaps.get("Private_Dirty"),
                "shared_clean_bytes": smaps.get("Shared_Clean"),
                "shared_dirty_bytes": smaps.get("Shared_Dirty"),
            }
        )
        return payload

    status = _read_kb_map(Path("/proc/self/status"))
    if status:
        payload.update(
            {
                "rss_bytes": status.get("VmRSS"),
                "hwm_bytes": status.get("VmHWM"),
            }
        )
    return payload


class StageMemoryRecorder:
    """Write stage-boundary and optional periodic process-memory snapshots.

    The recorder is only instantiated when memory diagnostics are enabled. Keeping
    disabled paths as ``None`` avoids /proc reads, CUDA queries, file writes, and
    optional NVML imports on normal runs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        entrypoint: str,
        algorithm: str,
        device: str,
        epoch_stride: int = 1,
        sample_interval_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import threading

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.entrypoint = str(entrypoint)
        self.algorithm = str(algorithm)
        self.device = "cuda" if str(device) == "gpu" else str(device)
        self.epoch_stride = max(1, int(epoch_stride))
        if sample_interval_ms is not None:
            import math

            if isinstance(sample_interval_ms, bool):
                raise ValueError("sample_interval_ms must be positive and finite when set")
            sample_interval_ms = float(sample_interval_ms)
            if not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0:
                raise ValueError("sample_interval_ms must be positive and finite when set")
        self.sample_interval_ms = sample_interval_ms
        self.pid = os.getpid()
        self._started_perf = time.perf_counter()
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._active_stages: list[str] = []
        self._sampler_stop = threading.Event() if sample_interval_ms is not None else None
        self._sampler_thread: Optional[Any] = None
        self._sampler_error: Optional[str] = None
        self._cupy = None
        self._cuda_device_id = None
        self._nvml = None
        self._nvml_handle = None
        self._mlx = None
        if self.device == "cuda":
            try:
                import cupy as cp  # type: ignore

                self._cupy = cp
                self._cuda_device_id = int(cp.cuda.runtime.getDevice())
            except Exception:
                self._cupy = None
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                device_id = self._cuda_device_id or 0
                self._nvml = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception:
                self._nvml = None
                self._nvml_handle = None
        elif self.device == "metal":
            try:
                import mlx.core as mx  # type: ignore

                self._mlx = mx
            except Exception:
                self._mlx = None
        self.record(
            "diagnostics",
            "start",
            metadata={
                "metadata": dict(metadata or {}),
                "epoch_stride": self.epoch_stride,
                "sample_interval_ms": self.sample_interval_ms,
            },
        )
        if self._sampler_stop is not None:
            self._sampler_thread = threading.Thread(
                target=self._sample_periodically,
                name=f"stage-memory-{self.run_id}",
                daemon=True,
            )
            self._sampler_thread.start()

    def should_record_epoch(self, epoch: int | None) -> bool:
        return epoch is None or int(epoch) % self.epoch_stride == 0

    def _gpu_snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        cp = self._cupy
        if cp is not None:
            try:
                device_id = self._cuda_device_id or 0
                with cp.cuda.Device(device_id):
                    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                    pool = cp.get_default_memory_pool()
                    pinned = cp.get_default_pinned_memory_pool()
                    payload.update(
                        {
                            "cuda_free_bytes": int(free_bytes),
                            "cuda_total_bytes": int(total_bytes),
                            "cupy_pool_used_bytes": int(pool.used_bytes()),
                            "cupy_pool_total_bytes": int(pool.total_bytes()),
                            "cupy_pinned_pool_free_blocks": int(pinned.n_free_blocks()),
                        }
                    )
            except Exception as exc:
                payload["cuda_error"] = f"{type(exc).__name__}: {exc}"

        nvml = self._nvml
        if nvml is not None and self._nvml_handle is not None:
            try:
                for proc in nvml.nvmlDeviceGetComputeRunningProcesses(
                    self._nvml_handle
                ):
                    if int(proc.pid) == self.pid:
                        payload["nvml_process_used_bytes"] = int(proc.usedGpuMemory)
                        break
            except Exception as exc:
                payload["nvml_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def _metal_snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        mx = self._mlx
        if mx is None:
            return payload
        try:
            payload.update(
                {
                    "mlx_active_memory_bytes": int(mx.get_active_memory()),
                    "mlx_cache_memory_bytes": int(mx.get_cache_memory()),
                    "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                }
            )
        except Exception as exc:
            payload["metal_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "cpu": _cpu_snapshot(),
        }
        if self.device == "cuda":
            payload["gpu"] = self._gpu_snapshot()
        elif self.device == "metal":
            payload["metal"] = self._metal_snapshot()
        return payload

    def record(
        self,
        stage: str,
        event: str,
        *,
        epoch: int | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.should_record_epoch(epoch):
            return
        with self._lock:
            self._record_locked(
                stage,
                event,
                epoch=epoch,
                metadata=metadata,
            )

    def _record_locked(
        self,
        stage: str,
        event: str,
        *,
        epoch: int | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        stage = str(stage)
        event = str(event)
        if event == "begin":
            self._active_stages.append(stage)
        row_metadata = dict(metadata or {})
        if stage == "periodic":
            row_metadata["active_stages"] = list(self._active_stages)
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "entrypoint": self.entrypoint,
            "algorithm": self.algorithm,
            "device": self.device,
            "pid": self.pid,
            "stage": stage,
            "event": event,
            "epoch": None if epoch is None else int(epoch),
            "time_unix_s": time.time(),
            "time_since_start_s": time.perf_counter() - self._started_perf,
        }
        if row_metadata:
            row["metadata"] = row_metadata
        row.update(self.snapshot())
        self._file.write(json.dumps(row, sort_keys=True) + "\n")
        if event in {"end", "error"}:
            for index in range(len(self._active_stages) - 1, -1, -1):
                if self._active_stages[index] == stage:
                    del self._active_stages[index]
                    break

    def _sample_periodically(self) -> None:
        stop = self._sampler_stop
        if stop is None or self.sample_interval_ms is None:
            return
        interval_s = self.sample_interval_ms / 1000.0
        while not stop.wait(interval_s):
            try:
                self.record("periodic", "sample")
            except Exception as exc:
                self._sampler_error = f"{type(exc).__name__}: {exc}"
                stop.set()
                return

    def close(self) -> None:
        if self._sampler_stop is not None:
            self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join()
        with self._lock:
            if self._file.closed:
                return
            try:
                metadata = (
                    {"periodic_sampler_error": self._sampler_error}
                    if self._sampler_error is not None
                    else None
                )
                self._record_locked("diagnostics", "end", metadata=metadata)
            finally:
                self._file.close()
                if self._nvml is not None:
                    try:
                        self._nvml.nvmlShutdown()
                    except Exception:
                        pass
