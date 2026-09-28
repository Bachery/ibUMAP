"""Hybrid CPU graph/init plus MLX optimizer pipeline boundary."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

import numpy as np

from .cpu_pipeline import CPUPipeline
from ..runtime.workspace import MetalWorkspace
from ..utils import DEVICE_SAMPLING_DTYPE, make_epochs_per_sample


class MetalPipeline(CPUPipeline):
    """Keep graph/init on the host and dispatch optimization to MLX Metal."""

    def __init__(self, model: Any):
        super().__init__(model)
        self.workspace = MetalWorkspace()

    def _optimize(self, init_embedding) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        from ..optimizers.metal_ibumap_optimizer import optimize_ibumap_metal

        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("optimization", "begin")
        started = perf_counter()
        try:
            epochs_started = perf_counter()
            epochs_per_sample = make_epochs_per_sample(
                self.state.weights,
                self.model._resolved_n_epochs,
                dtype=DEVICE_SAMPLING_DTYPE,
            )
            epochs_per_sample_time = perf_counter() - epochs_started
            embedding, timings = optimize_ibumap_metal(
                np.asarray(init_embedding, dtype=np.float32, order="C"),
                edgesrc=self.state.edgesrc,
                edgetgt=self.state.edgetgt,
                weights=self.state.weights,
                degrees=self.state.degrees,
                epochs_per_sample=epochs_per_sample,
                n_epochs=self.model._resolved_n_epochs,
                model=self.model,
                workspace=self.workspace,
            )
            result = np.asarray(embedding, dtype=np.float32, order="C")
            expected_shape = (self.state.n_vertices, self.model.n_components)
            if result.shape != expected_shape:
                raise RuntimeError(
                    f"Metal optimizer returned shape {result.shape}, expected {expected_shape}"
                )
            self.model._time_costs.update(
                {str(key): float(value) for key, value in timings.items()}
            )
            embd_opt_time = perf_counter() - started
            self.model._time_costs[
                "metal_epochs_per_sample_time_s"
            ] = epochs_per_sample_time
            self.model._time_costs["embd_opt_time"] = embd_opt_time
            self.model._time_costs["metal_pipeline_overhead_time_s"] = max(
                0.0,
                embd_opt_time
                - float(self.model._time_costs.get("metal_optimizer_time_s", 0.0)),
            )
            if recorder is not None:
                recorder.record(
                    "optimization",
                    "end",
                    metadata={"shape": tuple(int(value) for value in result.shape)},
                )
            self._finalize_workspace()
            return result
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "optimization",
                    "error",
                    metadata={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            raise

    def update(self, X, init, n_epochs: Optional[int]) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Pipeline state is empty; call fit first")
        if n_epochs is not None:
            self.model._resolved_n_epochs = int(n_epochs)
        if init is None:
            init_embedding = np.asarray(self.model.embedding_, dtype=np.float32).copy()
        elif isinstance(init, str):
            init_embedding = self._initialize_embedding(X, init)
        else:
            init_embedding = self._external_init_embedding(init)
        return self._optimize(init_embedding)
