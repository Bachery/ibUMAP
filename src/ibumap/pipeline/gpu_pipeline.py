from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state

from ..config import EffectiveConfig
from ..graph import build_cpu_graph_from_knn, build_gpu_graph_cuml
from ..init import initialize_embedding_gpu
from ..kernels.gpu.cupy_sparse import deterministic_csr_row_sum_cupy
from ..optimizers.ibumap_optimizer import umap_true_loss_optimization
from ..optimizers.tfdp_optimizer import tfdp_optimization
from ..runtime.device import ensure_cupy_runtime, ensure_gpu_runtime
from ..runtime.workspace import GPUWorkspace
from ..utils import (
    _resolve_preprocess_epochs,
    derive_random_seed,
    make_epochs_per_sample_gpu,
    preprocess_graph_csr,
)


@dataclass
class GPUPipelineState:
    graph: Any
    head: Optional[Any]
    tail: Optional[Any]
    edgesrc: Any
    edgetgt: Any
    weights: Any
    degrees: Optional[Any]
    n_vertices: int


class GPUPipeline:
    def __init__(self, model: Any):
        if model.numeric_dtype == np.dtype(np.float64):
            raise NotImplementedError(
                "dtype='float64' is currently supported on CPU only; "
                "CUDA kernels use float32"
            )
        self.model = model
        self.workspace = GPUWorkspace()
        self.state: Optional[GPUPipelineState] = None

    def _random_seed(self, stream: str):
        root = self.model.random_state
        if root is None and self.model.runtime.deterministic:
            root = 0
        return derive_random_seed(root, stream)

    def clear_workspace(self) -> None:
        self.workspace.clear()

    def _finalize_workspace(self) -> None:
        if self.model.config.runtime.workspace_policy == "minimal":
            self.workspace.clear()

    def _synchronize(self) -> None:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()

    def _legacy_params(self, n_vertices: int) -> Dict[str, Any]:
        cfg: EffectiveConfig = self.model.runtime
        constraint = cfg.constraint
        noise = cfg.noise
        attraction_schedule = self.model.config.ibumap.experimental.attraction_schedule

        if constraint.known_points_indices is None or constraint.known_points_positions is None:
            whether_known_points = np.empty(0, dtype=bool)
            reverse = np.empty(0, dtype=np.int64)
            known_points_positions = np.empty(
                (0, self.model.n_components), dtype=np.float32
            )
        else:
            whether_known_points = np.zeros(n_vertices, dtype=bool)
            reverse = np.full((n_vertices,), -1, dtype=np.int64)
            idx = np.asarray(constraint.known_points_indices, dtype=np.int64)
            pos = np.asarray(constraint.known_points_positions, dtype=np.float32)
            whether_known_points[idx] = True
            reverse[idx] = np.arange(idx.shape[0], dtype=np.int64)
            known_points_positions = pos

        return {
            "umap_a": self.model._a,
            "umap_b": self.model._b,
            "umap_gamma": self.model.repulsion_strength,
            "umap_initial_alpha": self.model.learning_rate,
            "umap_epsilon": cfg.numerics.epsilon,
            "ibfft_kernel_clip": self.model.ibfft_kernel_clip,
            "ibfft_kernel_subsample_mode": self.model.ibfft_kernel_subsample_mode,
            "ibfft_kernel_subsample_radius_cells": self.model.ibfft_kernel_subsample_radius_cells,
            "ibfft_kernel_subsample_points": self.model.ibfft_kernel_subsample_points,
            "umap_seperate_time_measure": False,
            "attr_gauss": False,
            "repl_gauss": False,
            "gauss_sigma": 1.0,
            "n_interpolation_points": cfg.fft.n_interpolation_points,
            "intervals_per_integer": cfg.fft.intervals_per_integer,
            "min_num_intervals": cfg.fft.min_num_intervals,
            "n_boxes_per_dim": cfg.fft.n_boxes_per_dim,
            "interpolation_schedule": cfg.fft.interpolation_schedule,
            "tfdp_alpha": cfg.tfdp.alpha,
            "tfdp_beta": cfg.tfdp.beta,
            "tfdp_gamma": cfg.tfdp.gamma,
            "tfdp_paraFactor": cfg.tfdp.para_factor,
            "tfdp_combine": cfg.fft.combine_stages,
            "soft_constraint": constraint.mode == "soft",
            "constraint_weight": constraint.weight,
            "known_points_indices": constraint.known_points_indices,
            "known_points_positions": known_points_positions,
            "whether_known_points": whether_known_points,
            "known_points_reverse_index": reverse,
            "noise_mode": noise.mode,
            "noise_scale": noise.scale,
            "noise_decay": noise.decay,
            "noise_until_epoch": noise.until_epoch,
            "noise_seed": noise.seed,
            "hybrid_mode": noise.hybrid_mode,
            "hybrid_switch_epoch": noise.hybrid_switch_epoch,
            "attraction_degree_damping": self.model.attraction_degree_damping,
            "attraction_degree_damping_mode": self.model.attraction_degree_damping_mode,
            "attraction_degree_damping_ref": self.model.attraction_degree_damping_ref,
            "attraction_degree_damping_ref_value": self.model.attraction_degree_damping_ref_value,
            "attraction_degree_damping_power": self.model.attraction_degree_damping_power,
            "attraction_degree_damping_min_scale": self.model.attraction_degree_damping_min_scale,
            "repulsion_clip_norm": self.model.repulsion_clip_norm,
            "repulsion_clip_with_alpha": self.model.repulsion_clip_with_alpha,
            "repulsion_clip_epoch_range": self.model.repulsion_clip_epoch_range,
            "diagnostics_path": self.model.diagnostics_path,
            "diagnostics_timing_path": self.model.diagnostics_timing_path,
            "diagnostics_thresholds": self.model.diagnostics_thresholds,
            "diagnostics_topk_path": self.model.diagnostics_topk_path,
            "diagnostics_topk": self.model.diagnostics_topk,
            "diagnostics_labels": self.model.diagnostics_labels,
            "diagnostics_point_ids": self.model.diagnostics_point_ids,
            "memory_recorder": getattr(self.model, "_memory_recorder", None),
            "total_update_clip_norm": self.model.total_update_clip_norm,
            "total_update_clip_with_alpha": self.model.total_update_clip_with_alpha,
            "total_update_clip_epoch_range": self.model.total_update_clip_epoch_range,
            "local_exact_repulsion": self.model.local_exact_repulsion,
            "local_exact_k": self.model.local_exact_k,
            "local_exact_radius_factor": self.model.local_exact_radius_factor,
            "local_exact_weight": self.model.local_exact_weight,
            "local_exact_every": self.model.local_exact_every,
            "local_exact_clip": self.model.local_exact_clip,
            "local_exact_symmetric": self.model.local_exact_symmetric,
            "local_exact_start_epoch": self.model.local_exact_start_epoch,
            "local_exact_end_epoch": self.model.local_exact_end_epoch,
            "local_exact_start_frac": self.model.local_exact_start_frac,
            "local_exact_end_frac": self.model.local_exact_end_frac,
            "local_exact_density_filter": self.model.local_exact_density_filter,
            "local_exact_min_cell_count": self.model.local_exact_min_cell_count,
            "local_exact_min_cell_count_quantile": self.model.local_exact_min_cell_count_quantile,
            "local_exact_density_include_neighbor_cells": self.model.local_exact_density_include_neighbor_cells,
            "local_exact_timing_sample_size": self.model.local_exact_timing_sample_size,
            "local_density_pressure": self.model.local_density_pressure,
            "local_density_pressure_weight": self.model.local_density_pressure_weight,
            "local_density_pressure_every": self.model.local_density_pressure_every,
            "local_density_pressure_min_count": self.model.local_density_pressure_min_count,
            "local_density_pressure_clip": self.model.local_density_pressure_clip,
            "local_density_pressure_power": self.model.local_density_pressure_power,
            "attraction_kernel_mode": attraction_schedule.kernel_mode,
            "attraction_schedule_mode": attraction_schedule.schedule_mode,
            "attraction_calendar_memory_limit_bytes": attraction_schedule.calendar_memory_limit_bytes,
            "attraction_warp_min_degree": attraction_schedule.warp_min_degree,
            "sampling_random_state": self._random_seed("sampling"),
            "noise_random_state": self._random_seed("noise"),
            "deterministic": self.model.runtime.deterministic,
            "p2m_mode": self.model.config.fft.p2m_mode,
            "fft_kernel_cache_policy": self.model.config.fft.kernel_cache_policy,
            "fft_kernel_cache_limit_bytes": self.model.config.fft.kernel_cache_limit_bytes,
            "fft_kernel_cache_max_entries": self.model.config.fft.kernel_cache_max_entries,
            "workspace_policy": self.model.config.runtime.workspace_policy,
            "workspace_limit_bytes": self.model.config.runtime.workspace_limit_bytes,
            "optimizer_workspace": self.workspace,
        }

    def _prepare_state(self, X):
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("graph", "begin")
        self._synchronize()
        t0 = time()
        try:
            graph = build_gpu_graph_cuml(
                X,
                n_neighbors=self.model.n_neighbors,
                metric=self.model.metric,
                random_state=self._random_seed("graph"),
                verbose=self.model.verbose,
                metric_kwds=self.model.metric_kwds,
                set_op_mix_ratio=self.model.set_op_mix_ratio,
                local_connectivity=self.model.local_connectivity,
            )
            self._synchronize()
            build_graph_time = time() - t0

            self.model._time_costs["build_graph_time"] = build_graph_time
            state = self._state_from_gpu_graph(graph)
            if recorder is not None:
                recorder.record(
                    "graph",
                    "end",
                    metadata={"nnz": int(state.graph.nnz), "n_vertices": int(state.n_vertices)},
                )
            return state
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "graph",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise

    def _prepare_state_from_knn(self, X, knn_indices, knn_distances):
        from cupyx.scipy import sparse as cupyx_sparse

        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("graph_from_knn", "begin")
        t0 = time()
        try:
            graph_cpu = build_cpu_graph_from_knn(
                X,
                n_neighbors=self.model.n_neighbors,
                metric=self.model.metric,
                metric_kwds=self.model.metric_kwds,
                random_state=self._random_seed("graph"),
                knn_indices=knn_indices,
                knn_dists=knn_distances,
                angular_rp_forest=self.model.angular_rp_forest,
                set_op_mix_ratio=self.model.set_op_mix_ratio,
                local_connectivity=self.model.local_connectivity,
                densmap_or_output_dens=False,
                verbose=self.model.verbose,
                input_distance_func=self.model.input_distance_func,
            )
            fuzzy_graph_time = time() - t0

            self._synchronize()
            transfer_t0 = time()
            graph_gpu = cupyx_sparse.csr_matrix(graph_cpu)
            self._synchronize()
            graph_to_gpu_time = time() - transfer_t0

            self.model._time_costs["fuzzy_graph_time"] = fuzzy_graph_time
            self.model._time_costs["graph_to_gpu_time"] = graph_to_gpu_time
            self.model._time_costs["build_graph_time"] = fuzzy_graph_time + graph_to_gpu_time
            state = self._state_from_gpu_graph(graph_gpu)
            if recorder is not None:
                recorder.record(
                    "graph_from_knn",
                    "end",
                    metadata={"nnz": int(state.graph.nnz), "n_vertices": int(state.n_vertices)},
                )
            return state
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "graph_from_knn",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise

    def _prepare_state_from_external_graph(self, fuzzy_graph):
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse

        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("graph_to_gpu", "begin")
        if not issparse(fuzzy_graph):
            raise TypeError("fuzzy_graph must be a scipy sparse matrix")
        preprocess_t0 = time()
        graph_cpu, n_epochs, degrees_cpu, preprocess_stats = preprocess_graph_csr(
            fuzzy_graph,
            self.model.n_epochs,
            dtype=np.float32,
            return_degrees=self.model.runtime.algorithm == "ibumap",
        )
        graph_preprocess_time = time() - preprocess_t0

        self._synchronize()
        transfer_t0 = time()
        graph_gpu = cupyx_sparse.csr_matrix(graph_cpu)
        degrees = (
            cp.asarray(degrees_cpu, dtype=cp.float32)
            if degrees_cpu is not None
            else None
        )
        self._synchronize()
        graph_to_gpu_time = time() - transfer_t0

        self.model._time_costs["build_graph_time"] = 0.0
        self.model._time_costs["fuzzy_graph_time"] = 0.0
        self.model._time_costs["graph_preprocess_time"] = graph_preprocess_time
        self.model._time_costs["graph_preprocess_input_nnz"] = int(
            preprocess_stats["input_nnz"]
        )
        self.model._time_costs["graph_preprocess_nnz"] = int(
            preprocess_stats["nnz"]
        )
        self.model._time_costs["graph_preprocess_removed_edges"] = int(
            preprocess_stats["removed_edges"]
        )
        self.model._time_costs["graph_preprocess_reused_csr"] = int(
            preprocess_stats["reused_csr"]
        )
        self.model._time_costs["graph_to_gpu_time"] = graph_to_gpu_time
        state = self._state_from_gpu_csr(graph_gpu, n_epochs, degrees)
        if recorder is not None:
            recorder.record(
                "graph_to_gpu",
                "end",
                metadata={"nnz": int(state.graph.nnz), "n_vertices": int(state.n_vertices)},
            )
        return state

    def _prepare_state_from_prepared_graph(self, optimizer_graph):
        """Move an already-preprocessed graph to CUDA without preprocessing it."""
        from cupyx.scipy import sparse as cupyx_sparse

        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("prepared_graph_to_gpu", "begin")

        self._synchronize()
        transfer_t0 = time()
        if issparse(optimizer_graph):
            if optimizer_graph.shape[0] != optimizer_graph.shape[1]:
                raise ValueError("optimizer_graph must be square")
            graph_gpu = cupyx_sparse.csr_matrix(optimizer_graph)
        elif cupyx_sparse.isspmatrix(optimizer_graph):
            if optimizer_graph.shape[0] != optimizer_graph.shape[1]:
                raise ValueError("optimizer_graph must be square")
            graph_gpu = optimizer_graph.tocsr()
        else:
            raise TypeError(
                "optimizer_graph must be a SciPy or CuPyX CSR matrix returned by "
                "prepare_fixed_inputs"
            )
        self._synchronize()
        graph_to_gpu_time = time() - transfer_t0

        n_epochs, _ = _resolve_preprocess_epochs(
            graph_gpu.shape[0], self.model.n_epochs
        )
        degrees = self._optimizer_degrees(graph_gpu)
        self.model._time_costs["build_graph_time"] = 0.0
        self.model._time_costs["fuzzy_graph_time"] = 0.0
        self.model._time_costs["graph_preprocess_time"] = 0.0
        self.model._time_costs["graph_preprocess_input_nnz"] = int(graph_gpu.nnz)
        self.model._time_costs["graph_preprocess_nnz"] = int(graph_gpu.nnz)
        self.model._time_costs["graph_preprocess_removed_edges"] = 0
        self.model._time_costs["graph_preprocess_reused_csr"] = 1
        self.model._time_costs["graph_preprocessing_bypassed"] = 1
        self.model._time_costs["graph_to_gpu_time"] = graph_to_gpu_time
        self.model._time_costs["prepared_graph_state_time"] = graph_to_gpu_time
        state = self._state_from_gpu_csr(graph_gpu, n_epochs, degrees)
        if recorder is not None:
            recorder.record(
                "prepared_graph_to_gpu",
                "end",
                metadata={"nnz": int(state.graph.nnz), "n_vertices": int(state.n_vertices)},
            )
        return state

    def _state_from_gpu_csr(
        self,
        graph_csr,
        n_epochs: int,
        degrees=None,
    ) -> GPUPipelineState:
        self.model._resolved_n_epochs = int(n_epochs)
        return GPUPipelineState(
            graph=graph_csr,
            head=None,
            tail=None,
            edgesrc=graph_csr.indptr,
            edgetgt=graph_csr.indices,
            weights=graph_csr.data,
            degrees=degrees,
            n_vertices=int(graph_csr.shape[0]),
        )

    def _state_from_gpu_graph(self, graph) -> GPUPipelineState:
        graph = graph.tocoo()
        graph.sum_duplicates()

        default_epochs = 500 if graph.shape[0] <= 10000 else 200
        n_epochs = default_epochs if self.model.n_epochs is None else int(self.model.n_epochs)
        divisor = default_epochs if n_epochs <= 10 else n_epochs
        if graph.nnz > 0:
            threshold = graph.data.max() / float(divisor)
            graph.data[graph.data < threshold] = 0.0
            graph.eliminate_zeros()

        self.model._resolved_n_epochs = n_epochs

        graph_csr = graph.tocsr()
        degrees = self._optimizer_degrees(graph_csr)

        return GPUPipelineState(
            graph=graph_csr,
            head=None,
            tail=None,
            edgesrc=graph_csr.indptr,
            edgetgt=graph_csr.indices,
            weights=graph_csr.data,
            degrees=degrees,
            n_vertices=int(graph_csr.shape[0]),
        )

    def _optimizer_degrees(self, graph_csr):
        """Build ibUMAP degree damping input under the runtime policy."""

        if self.model.runtime.algorithm != "ibumap":
            return None
        if self.model.runtime.deterministic:
            degrees = deterministic_csr_row_sum_cupy(graph_csr)
            self.model._time_costs["optimizer_degree_deterministic"] = 1
            return degrees

        import cupy as cp

        self.model._time_costs["optimizer_degree_deterministic"] = 0
        return graph_csr.sum(axis=1).reshape(-1).astype(cp.float32)

    def _external_init_embedding(self, init_embedding):
        import cupy as cp

        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("external_initialization_to_gpu", "begin")
        init_embedding_cpu = np.array(
            init_embedding,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        expected_shape = (self.state.n_vertices, self.model.n_components)
        if init_embedding_cpu.shape != expected_shape:
            raise ValueError(
                "init_embedding shape must be "
                f"{expected_shape}, got {init_embedding_cpu.shape}"
            )
        self.model.init_embedding_ = init_embedding_cpu.copy()

        self._synchronize()
        t0 = time()
        init_embedding_gpu = cp.asarray(init_embedding_cpu, dtype=cp.float32)
        self._synchronize()
        self.model._time_costs["init_to_gpu_time"] = time() - t0
        if recorder is not None:
            recorder.record(
                "external_initialization_to_gpu",
                "end",
                metadata={"shape": tuple(int(value) for value in init_embedding_gpu.shape)},
            )
        return init_embedding_gpu

    def _initialize_embedding(self, X, init):
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("initialization", "begin")
        random_state = check_random_state(self._random_seed("init"))
        init_timing: Dict[str, float] = {}
        init_diagnostics: Dict[str, Any] = {}
        init_cfg = self.model.config.initialization
        try:
            init_embedding = initialize_embedding_gpu(
                X,
                self.state.graph,
                self.model.n_components,
                init,
                random_state,
                self.model.metric,
                self.model.metric_kwds,
                deterministic=self.model.runtime.deterministic,
                spectral_method=init_cfg.spectral_method,
                spectral_tol=init_cfg.spectral_tol,
                spectral_maxiter=init_cfg.spectral_maxiter,
                spectral_ncv=init_cfg.spectral_ncv,
                spectral_auto_defaults=init_cfg.spectral_auto_defaults,
                spectral_scale_policy=init_cfg.resolve_spectral_scale_policy(
                    self.model.runtime.algorithm
                ),
                spectral_max_span=init_cfg.spectral_max_span,
                spectral_jitter_relative=init_cfg.spectral_jitter_relative,
                spectral_jitter_max=init_cfg.spectral_jitter_max,
                report_duplicate_ratio=init_cfg.report_duplicate_ratio,
                init_timing=init_timing,
                init_diagnostics=init_diagnostics,
            )
            self.model._time_costs.update(init_timing)
            self.model._init_diagnostics = dict(init_diagnostics)
            if recorder is not None:
                recorder.record(
                    "initialization",
                    "end",
                    metadata={
                        "init": str(init) if isinstance(init, str) else "array",
                        "shape": tuple(int(value) for value in init_embedding.shape),
                    },
                )
            return init_embedding
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "initialization",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise

    def _optimize_external(self, init_embedding):
        import cupy as cp

        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        if self.model.runtime.algorithm == "umap":
            raise NotImplementedError(
                "GPU algorithm='umap' does not expose a graph-to-embedding optimizer"
            )

        legacy = self._legacy_params(self.state.n_vertices)

        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("optimization", "begin")
        self._synchronize()
        emb_opt_t0 = time()
        try:
            epochs_per_sample = make_epochs_per_sample_gpu(
                self.state.weights,
                self.model._resolved_n_epochs,
            )
            if self.model.runtime.algorithm == "ibumap":
                embedding, opt_time = umap_true_loss_optimization(
                    init_embedding,
                    self.state.edgesrc,
                    self.state.edgetgt,
                    self.state.weights,
                    self.state.degrees,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    self.model.n_components,
                    epochs_per_sample,
                    self.model.negative_sample_rate,
                    self.model.runtime.attraction_mode,
                    self.model.runtime.repulsion_mode,
                    "cuda",
                    legacy,
                )
            elif self.model.runtime.algorithm == "tfdp":
                epochs_per_negative_sample = (
                    epochs_per_sample / self.model.negative_sample_rate
                )
                embedding, opt_time = tfdp_optimization(
                    init_embedding,
                    self.state.edgesrc,
                    self.state.edgetgt,
                    self.state.weights,
                    None,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    None,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    self.model.negative_sample_rate,
                    "tFDP_ibFFT_GPU",
                    legacy,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.model.runtime.algorithm}")

            self._synchronize()
            emb_opt_time = time() - emb_opt_t0

            self.model._time_costs.update(opt_time)
            self.model._time_costs["embd_opt_time"] = emb_opt_time
            result = np.asarray(embedding, dtype=np.float32)
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
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise

    def fit(self, X, init):
        ensure_gpu_runtime(self.model.runtime.graph_backend)
        if initialize_embedding_gpu is None:
            raise RuntimeError("GPU initializer is unavailable; ensure cupy dependencies are installed")

        import cupy as cp

        recorder = getattr(self.model, "_memory_recorder", None)
        fit_pre_t0 = time()
        self.state = self._prepare_state(X)
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        legacy = self._legacy_params(self.state.n_vertices)

        emb_init_t0 = time()
        init_embedding = self._initialize_embedding(X, init)
        self.model.init_embedding_ = cp.asnumpy(init_embedding)
        emb_init_time = time() - emb_init_t0
        random_state = check_random_state(self._random_seed("init"))

        emb_opt_t0 = time()
        if recorder is not None:
            recorder.record("optimization", "begin")
        epochs_per_sample = make_epochs_per_sample_gpu(
            self.state.weights,
            self.model._resolved_n_epochs,
        )
        try:
            if self.model.runtime.algorithm == "umap":
                from cuml.manifold import UMAP as CuMLUMAP

                if recorder is not None:
                    recorder.record("input_to_gpu", "begin")
                X_gpu = cp.asarray(X, dtype=cp.float32)
                if recorder is not None:
                    recorder.record(
                        "input_to_gpu",
                        "end",
                        metadata={"shape": tuple(int(value) for value in X_gpu.shape)},
                    )
                umap_gpu = CuMLUMAP(
                    n_neighbors=self.model.n_neighbors,
                    n_components=self.model.n_components,
                    n_epochs=self.model._resolved_n_epochs,
                    min_dist=self.model.min_dist,
                    repulsion_strength=self.model.repulsion_strength,
                    random_state=random_state.randint(0, 2**31 - 1),
                )
                embedding = umap_gpu.fit_transform(X_gpu)
                opt_time = {
                    "opt_prep_time": 0.0,
                    "attr_time": 0.0,
                    "repl_time": 0.0,
                    "appl_time": 0.0,
                }
            elif self.model.runtime.algorithm == "ibumap":
                embedding, opt_time = umap_true_loss_optimization(
                    init_embedding,
                    self.state.edgesrc,
                    self.state.edgetgt,
                    self.state.weights,
                    self.state.degrees,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    self.model.n_components,
                    epochs_per_sample,
                    self.model.negative_sample_rate,
                    self.model.runtime.attraction_mode,
                    self.model.runtime.repulsion_mode,
                    "cuda",
                    legacy,
                )
            elif self.model.runtime.algorithm == "tfdp":
                epochs_per_negative_sample = (
                    epochs_per_sample / self.model.negative_sample_rate
                )
                embedding, opt_time = tfdp_optimization(
                    init_embedding,
                    self.state.edgesrc,
                    self.state.edgetgt,
                    self.state.weights,
                    None,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    None,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    self.model.negative_sample_rate,
                    "tFDP_ibFFT_GPU",
                    legacy,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.model.runtime.algorithm}")
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "optimization",
                    "error",
                    metadata={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            raise

        emb_opt_time = time() - emb_opt_t0

        self.model._time_costs.update(opt_time)
        self.model._time_costs["embd_init_time"] = emb_init_time
        self.model._time_costs["embd_opt_time"] = emb_opt_time

        if hasattr(embedding, "get"):
            result = embedding.get().astype(np.float32, copy=False)
        elif hasattr(embedding, "to_numpy"):
            result = embedding.to_numpy().astype(np.float32, copy=False)
        else:
            result = cp.asnumpy(embedding).astype(np.float32, copy=False)
        if recorder is not None:
            recorder.record(
                "optimization",
                "end",
                metadata={"shape": tuple(int(value) for value in result.shape)},
            )
        self._finalize_workspace()
        return result

    def fit_from_knn(self, X, knn_indices, knn_distances, init):
        ensure_cupy_runtime()
        if initialize_embedding_gpu is None:
            raise RuntimeError("GPU initializer is unavailable; ensure cupy dependencies are installed")

        fit_pre_t0 = time()
        self.state = self._prepare_state_from_knn(X, knn_indices, knn_distances)
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        self._synchronize()
        emb_init_t0 = time()
        init_embedding = self._initialize_embedding(X, init)
        self._synchronize()
        import cupy as cp

        self.model.init_embedding_ = cp.asnumpy(init_embedding)
        self.model._time_costs["embd_init_time"] = time() - emb_init_t0

        return self._optimize_external(init_embedding)

    def optimize_from_graph(self, X, fuzzy_graph, init_embedding):
        ensure_cupy_runtime()

        fit_pre_t0 = time()
        self.state = self._prepare_state_from_external_graph(fuzzy_graph)
        if self.state.n_vertices != X.shape[0]:
            raise ValueError("fuzzy_graph shape must match X")
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        init_embedding = self._external_init_embedding(init_embedding)
        self.model._time_costs["embd_init_time"] = 0.0

        return self._optimize_external(init_embedding)

    def optimize_from_prepared_graph(
        self,
        X=None,
        optimizer_graph=None,
        init_embedding=None,
        rng_state=None,
    ):
        ensure_cupy_runtime()
        if rng_state is not None:
            raise ValueError(
                "rng_state is only supported by CPU algorithm='umap'"
            )

        self.state = self._prepare_state_from_prepared_graph(optimizer_graph)
        if X is not None and self.state.n_vertices != X.shape[0]:
            raise ValueError("optimizer_graph shape must match X")
        self.model._time_costs["fit_preprocess_time"] = self.model._time_costs[
            "prepared_graph_state_time"
        ]

        init_embedding = self._external_init_embedding(init_embedding)
        self.model._time_costs["embd_init_time"] = 0.0
        return self._optimize_external(init_embedding)

    def update(self, X, init, n_epochs: Optional[int]):
        if self.state is None:
            raise RuntimeError("Pipeline state is empty; call fit first")
        if initialize_embedding_gpu is None:
            raise RuntimeError("GPU initializer is unavailable; ensure cupy dependencies are installed")

        if n_epochs is not None:
            self.model._resolved_n_epochs = int(n_epochs)

        import cupy as cp

        X_gpu = (
            cp.asarray(X, dtype=cp.float32)
            if self.model.runtime.algorithm == "umap"
            else None
        )
        legacy = self._legacy_params(self.state.n_vertices)

        if init is None:
            self.model._init_diagnostics = {"init_method": "existing"}
            init_embedding = cp.asarray(self.model.embedding_, dtype=cp.float32)
        elif isinstance(init, str):
            init_embedding = self._initialize_embedding(X, init)
        else:
            self.model._init_diagnostics = {"init_method": "custom"}
            init_embedding = cp.asarray(init, dtype=cp.float32)

        epochs_per_sample = make_epochs_per_sample_gpu(
            self.state.weights,
            self.model._resolved_n_epochs,
        )
        if self.model.runtime.algorithm == "umap":
            from cuml.manifold import UMAP as CuMLUMAP
            random_state = check_random_state(self._random_seed("sampling"))
            umap_gpu = CuMLUMAP(
                n_neighbors=self.model.n_neighbors,
                n_components=self.model.n_components,
                n_epochs=self.model._resolved_n_epochs,
                min_dist=self.model.min_dist,
                repulsion_strength=self.model.repulsion_strength,
                random_state=random_state.randint(0, 2**31 - 1),
                init=init_embedding,
            )
            embedding = umap_gpu.fit_transform(X_gpu)
            opt_time = {
                "opt_prep_time": 0.0,
                "attr_time": 0.0,
                "repl_time": 0.0,
                "appl_time": 0.0,
            }
        elif self.model.runtime.algorithm == "ibumap":
            del X_gpu
            embedding, opt_time = umap_true_loss_optimization(
                init_embedding,
                self.state.edgesrc,
                self.state.edgetgt,
                self.state.weights,
                self.state.degrees,
                self.model._resolved_n_epochs,
                self.state.n_vertices,
                self.model.n_components,
                epochs_per_sample,
                self.model.negative_sample_rate,
                self.model.runtime.attraction_mode,
                self.model.runtime.repulsion_mode,
                "cuda",
                legacy,
            )
        else:
            epochs_per_negative_sample = (
                epochs_per_sample / self.model.negative_sample_rate
            )
            embedding, opt_time = tfdp_optimization(
                init_embedding,
                self.state.edgesrc,
                self.state.edgetgt,
                self.state.weights,
                None,
                self.model._resolved_n_epochs,
                self.state.n_vertices,
                None,
                epochs_per_sample,
                epochs_per_negative_sample,
                self.model.negative_sample_rate,
                "tFDP_ibFFT_GPU",
                legacy,
            )

        self.model._time_costs.update(opt_time)

        if hasattr(embedding, "get"):
            result = embedding.get().astype(np.float32, copy=False)
        elif hasattr(embedding, "to_numpy"):
            result = embedding.to_numpy().astype(np.float32, copy=False)
        else:
            result = cp.asnumpy(embedding).astype(np.float32, copy=False)
        self._finalize_workspace()
        return result
