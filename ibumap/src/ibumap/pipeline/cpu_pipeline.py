from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.sparse import isspmatrix_csr, issparse
from sklearn.utils import check_random_state

from ..config import UMAPFFTConfig
from ..graph import (
    build_cpu_graph,
    build_cpu_graph_from_knn,
    resolve_cpu_graph_thread_policy,
)
from ..init import initialize_embedding_cpu
from ..optimizers.ibumap_optimizer import umap_true_loss_optimization
from ..optimizers.tfdp_optimizer import tfdp_optimization
from ..optimizers.umap_optimizer import (
    umap_optimize_layout_euclidean,
    umap_optimize_layout_euclidean_synchronous,
)
from ..runtime.workspace import CPUWorkspace
from ..utils import (
    CPU_SAMPLING_DTYPE,
    cal_degrees,
    derive_random_seed,
    get_rng_state,
    make_epochs_per_sample,
    normalize_rng_state,
    prepare_known_point_arrays,
    preprocess_graph,
    preprocess_graph_csr,
    _resolve_preprocess_epochs,
)


@dataclass
class CPUPipelineState:
    graph: Any
    head: Optional[np.ndarray]
    tail: Optional[np.ndarray]
    edgesrc: np.ndarray
    edgetgt: np.ndarray
    weights: np.ndarray
    degrees: Optional[np.ndarray]
    n_vertices: int


class CPUPipeline:
    def __init__(self, model: Any):
        self.model = model
        self.workspace = CPUWorkspace()
        self.state: Optional[CPUPipelineState] = None
        self._shared_random_state: Optional[np.random.RandomState] = None

    def _random_seed(self, stream: str):
        root = self.model.random_state
        if root is None and self.model.runtime.deterministic:
            root = 0
        return derive_random_seed(root, stream)

    def _root_random_state(self):
        root = self.model.random_state
        if root is None and self.model.runtime.deterministic:
            root = 0
        return root

    @contextmanager
    def _rng_run(self):
        """Scope shared-RNG state to one public pipeline invocation."""
        lifecycle = getattr(self.model.runtime, "rng_lifecycle", "independent")
        if lifecycle != "shared_rng":
            yield
            return

        if self._shared_random_state is not None:
            yield
            return

        self._shared_random_state = check_random_state(self._root_random_state())
        try:
            yield
        finally:
            self._shared_random_state = None

    def _stage_random_source(self, stream: str):
        if self._shared_random_state is not None:
            return self._shared_random_state
        return self._random_seed(stream)

    def _stage_random_state(self, stream: str) -> np.random.RandomState:
        return check_random_state(self._stage_random_source(stream))

    def _sampling_rng_state(self, rng_state=None) -> np.ndarray:
        if rng_state is not None:
            return normalize_rng_state(rng_state)
        return get_rng_state(self._stage_random_state("sampling"))

    def _run_with_rng(self, operation, *args, **kwargs):
        with self._rng_run():
            return operation(*args, **kwargs)

    def clear_workspace(self) -> None:
        self.workspace.clear()

    def _finalize_workspace(self) -> None:
        if self.model.config.runtime.workspace_policy == "minimal":
            self.workspace.clear()

    @property
    def _dtype(self) -> np.dtype:
        return self.model.numeric_dtype

    def _legacy_params(self, n_vertices: int) -> Dict[str, Any]:
        cfg: UMAPFFTConfig = self.model.runtime
        constraint = cfg.constraint
        noise = cfg.noise
        attraction_schedule = self.model.config.ibumap.experimental.attraction_schedule
        known_positions = constraint.known_points_positions
        known_indices = constraint.known_points_indices

        (
            whether_known_points,
            known_points_positions,
            known_points_reverse_index,
        ) = prepare_known_point_arrays(
            n_vertices,
            self.model.n_components,
            known_indices,
            known_positions,
            dtype=self._dtype,
        )

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
            "known_points_indices": known_indices,
            "known_points_positions": known_points_positions,
            "whether_known_points": whether_known_points,
            "known_points_reverse_index": known_points_reverse_index,
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
            "numeric_dtype": self._dtype,
            "p2m_mode": self.model.config.fft.p2m_mode,
            "fft_kernel_cache_policy": self.model.config.fft.kernel_cache_policy,
            "fft_kernel_cache_limit_bytes": self.model.config.fft.kernel_cache_limit_bytes,
            "fft_kernel_cache_max_entries": self.model.config.fft.kernel_cache_max_entries,
            "deterministic": self.model.runtime.deterministic,
            "workspace_policy": self.model.config.runtime.workspace_policy,
            "workspace_limit_bytes": self.model.config.runtime.workspace_limit_bytes,
            "optimizer_workspace": self.workspace,
        }

    def _prepare_state(self, X) -> CPUPipelineState:
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("graph", "begin")
        t0 = time()
        try:
            graph_thread_policy = resolve_cpu_graph_thread_policy(
                self.model.n_jobs,
                deterministic=self.model.runtime.deterministic,
                algorithm=self.model.runtime.algorithm,
                device=self.model.runtime.device,
            )
            self.model._time_costs.update(graph_thread_policy.diagnostics())
            graph, _, _, _ = build_cpu_graph(
                X,
                n_neighbors=self.model.n_neighbors,
                metric=self.model.metric,
                metric_kwds=self.model.metric_kwds,
                random_state=self._stage_random_source("graph"),
                angular_rp_forest=self.model.angular_rp_forest,
                low_memory=self.model.low_memory,
                n_jobs=graph_thread_policy.effective_n_jobs,
                verbose=self.model.verbose,
                set_op_mix_ratio=self.model.set_op_mix_ratio,
                local_connectivity=self.model.local_connectivity,
                densmap_or_output_dens=False,
                input_distance_func=self.model.input_distance_func,
                sparse_data=False,
                preserve_edge_order=self.model.runtime.algorithm == "umap",
            )
            build_graph_time = time() - t0

            self.model._time_costs["build_graph_time"] = build_graph_time
            state = self._state_from_graph(graph)
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

    def _prepare_state_from_knn(self, X, knn_indices, knn_distances) -> CPUPipelineState:
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("graph_from_knn", "begin")
        t0 = time()
        try:
            graph = build_cpu_graph_from_knn(
                X,
                n_neighbors=self.model.n_neighbors,
                metric=self.model.metric,
                metric_kwds=self.model.metric_kwds,
                random_state=self._stage_random_source("graph"),
                knn_indices=knn_indices,
                knn_dists=knn_distances,
                angular_rp_forest=self.model.angular_rp_forest,
                set_op_mix_ratio=self.model.set_op_mix_ratio,
                local_connectivity=self.model.local_connectivity,
                densmap_or_output_dens=False,
                verbose=self.model.verbose,
                input_distance_func=self.model.input_distance_func,
                dtype=self._dtype,
                preserve_edge_order=self.model.runtime.algorithm == "umap",
            )
            fuzzy_graph_time = time() - t0
            self.model._time_costs["fuzzy_graph_time"] = fuzzy_graph_time
            self.model._time_costs["build_graph_time"] = fuzzy_graph_time
            state = self._state_from_graph(graph)
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

    def _state_from_graph(self, graph) -> CPUPipelineState:
        if not issparse(graph):
            raise TypeError("fuzzy_graph must be a scipy sparse matrix")
        if self.model.runtime.algorithm == "umap":
            return self._state_from_umap_graph(graph)

        preprocess_t0 = time()
        graph_csr, n_epochs, degrees, preprocess_stats = preprocess_graph_csr(
            graph,
            self.model.n_epochs,
            dtype=self._dtype,
            return_degrees=self.model.runtime.algorithm in ("ibumap", "hybrid"),
        )
        self.model._time_costs["graph_preprocess_time"] = time() - preprocess_t0
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
        self.model._resolved_n_epochs = n_epochs
        # CPU UMAP has already branched to its ordered COO protocol above.
        keep_coo_indices = self.model.runtime.algorithm == "hybrid"
        edgesrc = graph_csr.indptr
        edgetgt = graph_csr.indices
        weights = graph_csr.data
        head = (
            np.repeat(
                np.arange(graph_csr.shape[0], dtype=edgetgt.dtype),
                np.diff(edgesrc),
            )
            if keep_coo_indices
            else None
        )
        tail = edgetgt if keep_coo_indices else None

        return CPUPipelineState(
            graph=graph_csr,
            head=head,
            tail=tail,
            edgesrc=edgesrc,
            edgetgt=edgetgt,
            weights=weights,
            degrees=degrees,
            n_vertices=graph_csr.shape[0],
        )

    def _state_from_umap_graph(self, graph) -> CPUPipelineState:
        """Preserve umap-learn's ordered COO edge/weight protocol."""
        preprocess_t0 = time()
        graph_coo, n_epochs, preprocess_stats = preprocess_graph(
            graph,
            self.model.n_epochs,
            return_stats=True,
        )
        graph_csr = graph_coo.tocsr()
        self.model._time_costs["graph_preprocess_time"] = time() - preprocess_t0
        self.model._time_costs["graph_preprocess_input_nnz"] = int(
            preprocess_stats["input_nnz"]
        )
        self.model._time_costs["graph_preprocess_nnz"] = int(
            preprocess_stats["nnz"]
        )
        self.model._time_costs["graph_preprocess_removed_edges"] = int(
            preprocess_stats["removed_edges"]
        )
        self.model._time_costs["graph_preprocess_reused_csr"] = 0
        self.model._resolved_n_epochs = n_epochs

        return CPUPipelineState(
            graph=graph_coo,
            head=graph_coo.row,
            tail=graph_coo.col,
            edgesrc=graph_csr.indptr,
            edgetgt=graph_csr.indices,
            weights=graph_coo.data,
            degrees=None,
            n_vertices=graph_coo.shape[0],
        )

    def _state_from_prepared_graph(self, optimizer_graph) -> CPUPipelineState:
        """Create optimizer state without re-running fuzzy-graph preprocessing."""
        if not isspmatrix_csr(optimizer_graph):
            raise TypeError(
                "optimizer_graph must be a scipy CSR matrix returned by "
                "prepare_fixed_inputs"
            )
        if optimizer_graph.shape[0] != optimizer_graph.shape[1]:
            raise ValueError("optimizer_graph must be square")
        if optimizer_graph.data.dtype != self._dtype:
            raise ValueError(
                "optimizer_graph dtype must match the model numeric dtype; "
                "prepare it with this model configuration"
            )
        if not optimizer_graph.has_canonical_format or not optimizer_graph.has_sorted_indices:
            raise ValueError(
                "optimizer_graph must be canonical sorted CSR; use "
                "prepare_fixed_inputs instead of a raw fuzzy graph"
            )

        state_t0 = time()
        n_epochs, _ = _resolve_preprocess_epochs(
            optimizer_graph.shape[0], self.model.n_epochs
        )
        degrees = (
            cal_degrees(
                optimizer_graph.shape[0],
                optimizer_graph.indptr,
                optimizer_graph.data,
            )
            if self.model.runtime.algorithm in ("ibumap", "hybrid")
            else None
        )
        self.model._resolved_n_epochs = n_epochs
        self.model._time_costs["graph_preprocess_time"] = 0.0
        self.model._time_costs["graph_preprocess_input_nnz"] = int(optimizer_graph.nnz)
        self.model._time_costs["graph_preprocess_nnz"] = int(optimizer_graph.nnz)
        self.model._time_costs["graph_preprocess_removed_edges"] = 0
        self.model._time_costs["graph_preprocess_reused_csr"] = 1
        self.model._time_costs["graph_preprocessing_bypassed"] = 1
        self.model._time_costs["prepared_graph_state_time"] = time() - state_t0

        keep_coo_indices = self.model.runtime.algorithm in ("umap", "hybrid")
        edgesrc = optimizer_graph.indptr
        edgetgt = optimizer_graph.indices
        return CPUPipelineState(
            graph=optimizer_graph,
            head=(
                np.repeat(
                    np.arange(optimizer_graph.shape[0], dtype=edgetgt.dtype),
                    np.diff(edgesrc),
                )
                if keep_coo_indices
                else None
            ),
            tail=edgetgt if keep_coo_indices else None,
            edgesrc=edgesrc,
            edgetgt=edgetgt,
            weights=optimizer_graph.data,
            degrees=degrees,
            n_vertices=optimizer_graph.shape[0],
        )

    def _initialize_embedding(self, X, init):
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("initialization", "begin")
        random_state = self._stage_random_state("init")
        init_cfg = self.model.config.initialization
        init_timing: Dict[str, float] = {}
        init_diagnostics: Dict[str, Any] = {}
        try:
            init_embedding = initialize_embedding_cpu(
                X,
                self.state.graph,
                self.model.n_components,
                init,
                random_state,
                self.model.metric,
                self.model.metric_kwds,
                dtype=self._dtype,
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

    def _external_init_embedding(self, init_embedding) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("external_initialization", "begin")
        init_embedding = np.array(init_embedding, dtype=self._dtype, order="C", copy=True)
        expected_shape = (self.state.n_vertices, self.model.n_components)
        if init_embedding.shape != expected_shape:
            raise ValueError(
                "init_embedding shape must be "
                f"{expected_shape}, got {init_embedding.shape}"
            )
        if recorder is not None:
            recorder.record(
                "external_initialization",
                "end",
                metadata={"shape": tuple(int(value) for value in init_embedding.shape)},
            )
        return init_embedding

    def _optimize_hybrid(self, init_embedding, legacy):
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        if self.state.head is None or self.state.tail is None:
            raise RuntimeError("Hybrid optimization requires COO edge indices")
        if self.state.degrees is None:
            raise RuntimeError("Hybrid optimization requires graph degrees")

        ibumap_epochs, refinement_epochs = self.model.config.hybrid.split_epochs(
            self.model._resolved_n_epochs
        )

        ibumap_epochs_per_sample = make_epochs_per_sample(
            self.state.weights,
            ibumap_epochs,
            dtype=CPU_SAMPLING_DTYPE,
        )
        ibumap_start = time()
        embedding, ibumap_time_costs = umap_true_loss_optimization(
            init_embedding,
            self.state.edgesrc,
            self.state.edgetgt,
            self.state.weights,
            self.state.degrees,
            ibumap_epochs,
            self.state.n_vertices,
            self.model.n_components,
            ibumap_epochs_per_sample,
            self.model.negative_sample_rate,
            "sampling",
            "true_loss",
            "cpu",
            legacy,
        )
        ibumap_wall_time = time() - ibumap_start

        refinement_epochs_per_sample = make_epochs_per_sample(
            self.state.weights,
            refinement_epochs,
            dtype=CPU_SAMPLING_DTYPE,
        )
        refinement_epochs_per_negative_sample = (
            refinement_epochs_per_sample
            / self.model.config.hybrid.refinement_negative_sample_rate
        )
        refinement_random_state = self._stage_random_state("refinement")
        refinement_legacy = dict(legacy)
        refinement_legacy["umap_initial_alpha"] = (
            self.model.config.hybrid.refinement_learning_rate
        )

        refinement_start = time()
        embedding, refinement_time_costs = umap_optimize_layout_euclidean(
            embedding,
            self.state.head,
            self.state.tail,
            refinement_epochs,
            self.state.n_vertices,
            self.model.n_components,
            get_rng_state(refinement_random_state),
            refinement_epochs_per_sample,
            refinement_epochs_per_negative_sample,
            parallel=(not self.model.runtime.deterministic),
            verbose=self.model.verbose,
            tqdm_kwds=self.model.tqdm_kwds,
            move_other=True,
            drfft_params=refinement_legacy,
        )
        refinement_wall_time = time() - refinement_start

        time_costs = {
            f"hybrid_ibumap_{key}": value
            for key, value in ibumap_time_costs.items()
        }
        time_costs.update(
            {
                f"hybrid_refinement_{key}": value
                for key, value in refinement_time_costs.items()
            }
        )
        time_costs.update(
            {
                "hybrid_total_epochs": int(self.model._resolved_n_epochs),
                "hybrid_ibumap_epochs": int(ibumap_epochs),
                "hybrid_refinement_epochs": int(refinement_epochs),
                "hybrid_refinement_fraction": float(
                    self.model.config.hybrid.refinement_fraction
                ),
                "hybrid_ibumap_time": float(ibumap_wall_time),
                "hybrid_refinement_time": float(refinement_wall_time),
                "hybrid_optimization_time": float(
                    ibumap_wall_time + refinement_wall_time
                ),
                "opt_prep_time": float(
                    ibumap_time_costs.get("opt_prep_time", 0.0)
                    + refinement_time_costs.get("opt_prep_time", 0.0)
                ),
                "optimization_time": float(
                    ibumap_wall_time + refinement_wall_time
                ),
                "appl_time": float(
                    ibumap_time_costs.get("appl_time", 0.0)
                    + refinement_time_costs.get("appl_time", 0.0)
                ),
            }
        )
        return embedding, time_costs

    def _optimize(self, init_embedding, rng_state=None) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Pipeline state is empty")
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("optimization", "begin")
        legacy = self._legacy_params(self.state.n_vertices)
        emb_opt_t0 = time()
        try:
            if self.model.runtime.algorithm == "umap":
                epochs_per_sample = make_epochs_per_sample(
                    self.state.weights,
                    self.model._resolved_n_epochs,
                    dtype=CPU_SAMPLING_DTYPE,
                )
                epochs_per_negative_sample = (
                    epochs_per_sample / self.model.negative_sample_rate
                )
                rng_state = self._sampling_rng_state(rng_state)
                if self.model.runtime.umap_sgd_update_mode == "asynchronous":
                    # Original UMAP-style online SGD applies each sampled update immediately.
                    embedding, opt_time = umap_optimize_layout_euclidean(
                        init_embedding,
                        self.state.head,
                        self.state.tail,
                        self.model._resolved_n_epochs,
                        self.state.n_vertices,
                        self.model.n_components,
                        rng_state,
                        epochs_per_sample,
                        epochs_per_negative_sample,
                        parallel=(not self.model.runtime.deterministic),
                        verbose=self.model.verbose,
                        tqdm_kwds=self.model.tqdm_kwds,
                        move_other=True,
                        drfft_params=legacy,
                    )
                else:
                    # Diagnostic CPU mode accumulates sampled updates for one epoch.
                    embedding, opt_time = umap_optimize_layout_euclidean_synchronous(
                        init_embedding,
                        self.state.head,
                        self.state.tail,
                        self.model._resolved_n_epochs,
                        self.state.n_vertices,
                        self.model.n_components,
                        rng_state,
                        epochs_per_sample,
                        epochs_per_negative_sample,
                        verbose=self.model.verbose,
                        tqdm_kwds=self.model.tqdm_kwds,
                        move_other=True,
                        drfft_params=legacy,
                    )
            elif self.model.runtime.algorithm == "ibumap":
                epochs_per_sample = make_epochs_per_sample(
                    self.state.weights,
                    self.model._resolved_n_epochs,
                    dtype=CPU_SAMPLING_DTYPE,
                )
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
                    "cpu",
                    legacy,
                )
            elif self.model.runtime.algorithm == "hybrid":
                embedding, opt_time = self._optimize_hybrid(
                    init_embedding,
                    legacy,
                )
            elif self.model.runtime.algorithm == "tfdp":
                epochs_per_sample = make_epochs_per_sample(
                    self.state.weights,
                    self.model._resolved_n_epochs,
                    dtype=CPU_SAMPLING_DTYPE,
                )
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
                    "tFDP_ibFFT_CPU",
                    legacy,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.model.runtime.algorithm}")

            emb_opt_time = time() - emb_opt_t0

            self.model._time_costs.update(opt_time)
            self.model._time_costs["embd_opt_time"] = emb_opt_time
            result = embedding.astype(self._dtype, copy=False)
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

    def fit(self, X, init) -> np.ndarray:
        return self._run_with_rng(self._fit, X, init)

    def _fit(self, X, init) -> np.ndarray:
        fit_pre_t0 = time()
        self.state = self._prepare_state(X)
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        emb_init_t0 = time()
        init_embedding = self._initialize_embedding(X, init)
        self.model.init_embedding_ = init_embedding.copy()
        self.model._time_costs["embd_init_time"] = time() - emb_init_t0

        return self._optimize(init_embedding)

    def fit_from_knn(self, X, knn_indices, knn_distances, init) -> np.ndarray:
        return self._run_with_rng(
            self._fit_from_knn,
            X,
            knn_indices,
            knn_distances,
            init,
        )

    def _fit_from_knn(self, X, knn_indices, knn_distances, init) -> np.ndarray:
        fit_pre_t0 = time()
        self.state = self._prepare_state_from_knn(X, knn_indices, knn_distances)
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        emb_init_t0 = time()
        init_embedding = self._initialize_embedding(X, init)
        self.model.init_embedding_ = init_embedding.copy()
        self.model._time_costs["embd_init_time"] = time() - emb_init_t0

        return self._optimize(init_embedding)

    def optimize_from_graph(
        self,
        fuzzy_graph,
        init_embedding,
        X=None,
        rng_state=None,
    ) -> np.ndarray:
        return self._run_with_rng(
            self._optimize_from_graph,
            fuzzy_graph,
            init_embedding,
            X=X,
            rng_state=rng_state,
        )

    def _optimize_from_graph(
        self,
        fuzzy_graph,
        init_embedding,
        X=None,
        rng_state=None,
    ) -> np.ndarray:
        fit_pre_t0 = time()
        self.state = self._state_from_graph(fuzzy_graph)
        if X is not None and self.state.n_vertices != X.shape[0]:
            raise ValueError("fuzzy_graph shape must match X")
        self.model._time_costs["build_graph_time"] = 0.0
        self.model._time_costs["fuzzy_graph_time"] = 0.0
        self.model._time_costs["fit_preprocess_time"] = time() - fit_pre_t0

        init_embedding = self._external_init_embedding(init_embedding)
        self.model.init_embedding_ = init_embedding.copy()
        self.model._time_costs["embd_init_time"] = 0.0

        if rng_state is None:
            return self._optimize(init_embedding)
        return self._optimize(init_embedding, rng_state=rng_state)

    def optimize_from_prepared_graph(
        self,
        X=None,
        optimizer_graph=None,
        init_embedding=None,
        rng_state=None,
    ) -> np.ndarray:
        return self._run_with_rng(
            self._optimize_from_prepared_graph,
            X,
            optimizer_graph,
            init_embedding,
            rng_state=rng_state,
        )

    def _optimize_from_prepared_graph(
        self,
        X=None,
        optimizer_graph=None,
        init_embedding=None,
        rng_state=None,
    ) -> np.ndarray:
        recorder = getattr(self.model, "_memory_recorder", None)
        if recorder is not None:
            recorder.record("prepared_graph", "begin")
        self.state = self._state_from_prepared_graph(optimizer_graph)
        if X is not None and self.state.n_vertices != X.shape[0]:
            raise ValueError("optimizer_graph shape must match X")
        self.model._time_costs["build_graph_time"] = 0.0
        self.model._time_costs["fuzzy_graph_time"] = 0.0
        self.model._time_costs["fit_preprocess_time"] = self.model._time_costs[
            "prepared_graph_state_time"
        ]
        if recorder is not None:
            recorder.record(
                "prepared_graph",
                "end",
                metadata={
                    "nnz": int(self.state.graph.nnz),
                    "n_vertices": int(self.state.n_vertices),
                },
            )

        init_embedding = self._external_init_embedding(init_embedding)
        self.model.init_embedding_ = init_embedding.copy()
        self.model._time_costs["embd_init_time"] = 0.0
        if rng_state is None:
            return self._optimize(init_embedding)
        return self._optimize(init_embedding, rng_state=rng_state)

    def update(self, X, init, n_epochs: Optional[int]) -> np.ndarray:
        return self._run_with_rng(self._update, X, init, n_epochs)

    def _update(self, X, init, n_epochs: Optional[int]) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Pipeline state is empty; call fit first")

        if n_epochs is not None:
            self.model._resolved_n_epochs = int(n_epochs)

        legacy = self._legacy_params(self.state.n_vertices)

        if init is None:
            init_embedding = self.model.embedding_.copy()
        elif isinstance(init, str):
            random_state = self._stage_random_state("init")
            init_timing: Dict[str, float] = {}
            init_diagnostics: Dict[str, Any] = {}
            init_embedding = initialize_embedding_cpu(
                X,
                self.state.graph,
                self.model.n_components,
                init,
                random_state,
                self.model.metric,
                self.model.metric_kwds,
                dtype=self._dtype,
                spectral_method=self.model.config.initialization.spectral_method,
                spectral_tol=self.model.config.initialization.spectral_tol,
                spectral_maxiter=self.model.config.initialization.spectral_maxiter,
                spectral_ncv=self.model.config.initialization.spectral_ncv,
                spectral_auto_defaults=self.model.config.initialization.spectral_auto_defaults,
                spectral_scale_policy=self.model.config.initialization.resolve_spectral_scale_policy(
                    self.model.runtime.algorithm
                ),
                spectral_max_span=self.model.config.initialization.spectral_max_span,
                spectral_jitter_relative=self.model.config.initialization.spectral_jitter_relative,
                spectral_jitter_max=self.model.config.initialization.spectral_jitter_max,
                report_duplicate_ratio=self.model.config.initialization.report_duplicate_ratio,
                init_timing=init_timing,
                init_diagnostics=init_diagnostics,
            )
            self.model._time_costs.update(init_timing)
            self.model._init_diagnostics = dict(init_diagnostics)
        else:
            init_embedding = np.asarray(init, dtype=self._dtype)

        epochs_per_sample = make_epochs_per_sample(
            self.state.weights,
            self.model._resolved_n_epochs,
            dtype=CPU_SAMPLING_DTYPE,
        )
        if self.model.runtime.algorithm == "umap":
            epochs_per_negative_sample = (
                epochs_per_sample / self.model.negative_sample_rate
            )
            rng_state = self._sampling_rng_state()
            if self.model.runtime.umap_sgd_update_mode == "asynchronous":
                embedding, opt_time = umap_optimize_layout_euclidean(
                    init_embedding,
                    self.state.head,
                    self.state.tail,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    self.model.n_components,
                    rng_state,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    parallel=(not self.model.runtime.deterministic),
                    verbose=self.model.verbose,
                    tqdm_kwds=self.model.tqdm_kwds,
                    move_other=True,
                    drfft_params=legacy,
                )
            else:
                embedding, opt_time = umap_optimize_layout_euclidean_synchronous(
                    init_embedding,
                    self.state.head,
                    self.state.tail,
                    self.model._resolved_n_epochs,
                    self.state.n_vertices,
                    self.model.n_components,
                    rng_state,
                    epochs_per_sample,
                    epochs_per_negative_sample,
                    verbose=self.model.verbose,
                    tqdm_kwds=self.model.tqdm_kwds,
                    move_other=True,
                    drfft_params=legacy,
                )
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
                "cpu",
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
                "tFDP_ibFFT_CPU",
                legacy,
            )

        self.model._time_costs.update(opt_time)
        result = embedding.astype(self._dtype, copy=False)
        self._finalize_workspace()
        return result
