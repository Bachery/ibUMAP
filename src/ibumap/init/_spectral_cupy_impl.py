# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, the ibUMAP authors.
#
# Parts of this file are adapted from umap-learn (BSD-3-Clause,
# Copyright (c) 2017, Leland McInnes). See THIRD_PARTY_NOTICES.md.

import numpy as np
import cupy as cp
import warnings
from time import perf_counter
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import cupyx.scipy.sparse.csgraph
import cupyx.scipy.spatial.distance
from ._eigsh_cupy import (
	deterministic_csr_spmv_algorithm,
	eigsh_csr_alg1,
	eigsh_csr_alg2,
	fast_csr_spmv_algorithm,
)
from ._spectral_defaults import resolve_component_ncv, resolve_spectral_defaults
from ..kernels.gpu.cupy_sparse import (
	canonicalize_component_labels_cupy,
	deterministic_csr_row_sum_cupy,
)
# from time import time


_CSR_NORMALIZE_FLOAT32_KERNEL = cp.RawKernel(
	r"""
	extern "C" __global__
	void ibumap_csr_normalize_float32(
		const int n_rows,
		const int* indptr,
		const int* indices,
		const float* graph_data,
		const float* inverse_sqrt_degree,
		float* laplacian_data
	) {
		const int row = blockDim.x * blockIdx.x + threadIdx.x;
		if (row >= n_rows) {
			return;
		}
		const float left_scale = inverse_sqrt_degree[row];
		for (int position = indptr[row]; position < indptr[row + 1]; ++position) {
			const int column = indices[position];
			laplacian_data[position] = -(
				(left_scale * graph_data[position])
				* inverse_sqrt_degree[column]
			);
		}
	}
	""",
	"ibumap_csr_normalize_float32",
)


def _add_timing(timing, key, value):
	if timing is not None:
		timing[key] = float(timing.get(key, 0.0) + value)


def _set_diagnostic(diagnostics, key, value):
	if diagnostics is not None:
		diagnostics[key] = value


def _set_diagnostic_default(diagnostics, key, value):
	if diagnostics is not None and key not in diagnostics:
		diagnostics[key] = value


def _deterministic_normalized_laplacian_cupy(graph):
	"""Build a float32 normalized Laplacian in fixed CSR row order.

	CUDA spectral inputs are symmetric fuzzy graphs. For such graphs the CSR
	row sums equal the column sums used by the reference UMAP formulation.
	One CUDA thread processes each row sequentially, avoiding the atomic
	scatter used by CuPy's CSR ``sum(axis=0)``. Diagonal sparse products are
	replaced by direct per-edge scaling so no SpGEMM reduction is involved.
	"""

	graph = cupyx.scipy.sparse.csr_matrix(
		graph,
		dtype=cp.float32,
		copy=True,
	)
	graph.sum_duplicates()
	graph.sort_indices()
	if graph.indptr.dtype != cp.int32 or graph.indices.dtype != cp.int32:
		raise TypeError(
			"deterministic CUDA spectral initialization requires int32 CSR "
			"indices"
		)

	n_rows = int(graph.shape[0])
	threads = 256
	blocks = (n_rows + threads - 1) // threads
	degree = deterministic_csr_row_sum_cupy(graph)
	if not bool(cp.all(cp.isfinite(degree) & (degree > 0.0)).item()):
		raise FloatingPointError(
			"deterministic CUDA spectral graph contains a non-finite or "
			"non-positive degree"
		)

	sqrt_degree = cp.sqrt(degree)
	inverse_sqrt_degree = cp.reciprocal(sqrt_degree)
	laplacian = graph.copy()
	_CSR_NORMALIZE_FLOAT32_KERNEL(
		(blocks,),
		(threads,),
		(
			np.int32(n_rows),
			graph.indptr,
			graph.indices,
			graph.data,
			inverse_sqrt_degree,
			laplacian.data,
		),
	)
	laplacian.setdiag(
		laplacian.diagonal() + cp.float32(1.0)
	)
	laplacian.sum_duplicates()
	laplacian.sort_indices()
	return laplacian, sqrt_degree


def _build_normalized_laplacian_cupy(
	graph,
	*,
	deterministic,
	diagnostics=None,
):
	"""Build the CUDA normalized Laplacian with the requested reduction policy."""

	if deterministic:
		_set_diagnostic(
			diagnostics,
			"spectral_degree_reduction",
			"deterministic_csr_row_sum",
		)
		_set_diagnostic(
			diagnostics,
			"spectral_laplacian_construction",
			"deterministic_csr_direct_scaling",
		)
		_set_diagnostic(
			diagnostics,
			"spectral_symmetric_graph_assumed",
			True,
		)
		return _deterministic_normalized_laplacian_cupy(graph)

	_set_diagnostic(
		diagnostics,
		"spectral_degree_reduction",
		"cupy_csr_sum_axis0",
	)
	_set_diagnostic(
		diagnostics,
		"spectral_laplacian_construction",
		"cupy_sparse_diagonal_product",
	)
	sqrt_degree = cp.sqrt(
		cp.asarray(graph.sum(axis=0), dtype=cp.float32).squeeze()
	)
	identity = cupyx.scipy.sparse.identity(
		graph.shape[0],
		dtype=cp.float32,
	)
	degree_scale = cupyx.scipy.sparse.diags(1.0 / sqrt_degree, 0)
	graph = cupyx.scipy.sparse.csr_matrix(graph, dtype=cp.float32)
	laplacian = identity - degree_scale.dot(graph).dot(degree_scale)
	if not cupyx.scipy.sparse.issparse(laplacian):
		laplacian = cp.asarray(laplacian, dtype=cp.float32)
	return laplacian, sqrt_degree


def _random_component_layout(random_state, size, dim, data_range, center):
	data_range_value = float(
		data_range.item() if hasattr(data_range, "item") else data_range
	)
	if not np.isfinite(data_range_value) or data_range_value <= 0.0:
		raise FloatingPointError("component layout has a non-finite or non-positive range")
	if not bool(cp.isfinite(center).all().item()):
		raise FloatingPointError("component layout has a non-finite center")
	return (
		random_state.uniform(
			low=-data_range_value,
			high=data_range_value,
			size=(size, dim),
		).astype(cp.float32)
		+ center
	)

def component_layout_cupy(
	data,
	n_components,
	component_labels,
	dim,
	random_state,
	metric="euclidean",
	metric_kwds={},
):
	"""GPU version of umap.spectral.component_layout using CuPy
	"""
	# 如果没有数据，则返回随机猜测
	if data is None:
		if isinstance(random_state, int): random_state = cp.random.RandomState(random_state)
		return random_state.random(size=(n_components, dim)).astype(cp.float32) * cp.float32(10.0)
	
	if metric == "precomputed":
		# 处理预计算的距离矩阵
		distance_matrix = cp.zeros((n_components, n_components), dtype=cp.float32)
		linkage = metric_kwds.get("linkage", "average")
		if linkage == "average":	linkage_func = cp.mean
		elif linkage == "complete":	linkage_func = cp.max
		elif linkage == "single":	linkage_func = cp.min
		else:
			raise ValueError("Unrecognized linkage '%s'. Please choose from 'average', 'complete', or 'single'" % linkage)
		for c_i in range(n_components):
			mask_i = component_labels == c_i
			dm_i = data[mask_i]
			for c_j in range(c_i + 1, n_components):
				mask_j = component_labels == c_j
				dm_ij = dm_i[:, mask_j]
				dist = linkage_func(dm_ij)
				distance_matrix[c_i, c_j] = dist
				distance_matrix[c_j, c_i] = dist
	else:
		# 计算每个连通分量的质心
		component_centroids = cp.empty((n_components, data.shape[1]), dtype=cp.float32)
		for label in range(n_components):
			mask = component_labels == label
			component_centroids[label] = cp.mean(data[mask], axis=0)
		# 如果质心是稀疏矩阵，转换成稠密矩阵
		if cupyx.scipy.sparse.issparse(component_centroids):
			component_centroids = component_centroids.toarray()
		# 计算距离矩阵
		distance_matrix = cupyx.scipy.spatial.distance.cdist(
			component_centroids, component_centroids, metric=metric, **metric_kwds
		)

	# 计算亲和矩阵
	affinity_matrix = cp.exp(-(distance_matrix ** 2))
	# 谱嵌入：使用特征值分解
	eigenvalues, eigenvectors = cp.linalg.eigh(affinity_matrix)
	order = cp.argsort(eigenvalues)[::-1]	# 降序排列
	component_embedding = eigenvectors[:, order[:dim]]

	# 归一化
	component_embedding /= cp.max(component_embedding)

	return component_embedding


def multi_component_layout_cupy(
	data,
	graph,
	n_components,
	component_labels,
	dim,
	random_state,
	metric="euclidean",
	metric_kwds={},
	init="random",
	method=None,
	tol=0.0,
	maxiter=0,
	ncv=None,
	auto_defaults=False,
	timing=None,
	diagnostics=None,
	deterministic=False,
):
	"""GPU version of umap.spectral.multi_component_layout using CuPy
	"""

	# 初始化结果数组
	result = cp.empty((graph.shape[0], dim), dtype=cp.float32)

	# 计算质心嵌入
	if n_components > 2 * dim:
		feature_transfer_t0 = perf_counter()
		data = None if data is None else cp.asarray(data, dtype=cp.float32)
		feature_transfer_time = perf_counter() - feature_transfer_t0
		_add_timing(timing, "init_feature_transfer_time", feature_transfer_time)
		_add_timing(timing, "init_transfer_time", feature_transfer_time)
		_set_diagnostic(
			diagnostics,
			"spectral_feature_data_materialized",
			data is not None,
		)
		t0 = perf_counter()
		meta_embedding = component_layout_cupy(
			data,
			n_components,
			component_labels,
			dim,
			random_state,
			metric=metric,
			metric_kwds=metric_kwds,
		)
		_add_timing(timing, "init_component_layout_time", perf_counter() - t0)
	else:
		t0 = perf_counter()
		k = int(cp.ceil(n_components / 2.0))
		base = cp.hstack([
			cp.eye(k, dtype=cp.float32),
			cp.zeros((k, dim - k), dtype=cp.float32),
		])
		meta_embedding = cp.vstack([base, -base])[:n_components]
		_add_timing(timing, "init_component_layout_time", perf_counter() - t0)

	# 对每个连通分量进行嵌入
	for label in range(n_components):
		# 提取子图
		t0 = perf_counter()
		mask = component_labels == label
		component_graph = graph[mask, :][:, mask].tocoo()
		_add_timing(timing, "init_component_subgraph_time", perf_counter() - t0)
		# 计算距离并确定范围
		t0 = perf_counter()
		distances = cupyx.scipy.spatial.distance.cdist(meta_embedding[label].reshape(1, -1), meta_embedding, metric=metric)
		data_range = distances[distances > 0.0].min() / 2.0
		_add_timing(timing, "init_component_range_time", perf_counter() - t0)
		# eigsh requires enough room for k eigenpairs plus its Lanczos basis.
		k = dim + 1
		if component_graph.shape[0] <= k + 1:
			t0 = perf_counter()
			result[mask] = _random_component_layout(
				random_state,
				component_graph.shape[0],
				dim,
				data_range,
				meta_embedding[label],
			)
			_add_timing(timing, "init_component_random_time", perf_counter() - t0)
		else:
			component_embedding = _spectral_layout_cupy(
				data = None,
				graph = component_graph,
				dim = dim,
				random_state = random_state,
				metric = metric,
				metric_kwds = metric_kwds,
				init = init,
				method = method,
				tol = tol,
				maxiter = maxiter,
				ncv = ncv,
				auto_defaults = auto_defaults,
				deterministic = deterministic,
				timing = timing,
				diagnostics = diagnostics,
			)
			t0 = perf_counter()
			scale_denominator = cp.max(cp.abs(component_embedding))
			scale_denominator_value = float(scale_denominator.item())
			data_range_value = float(data_range.item())
			scaling_is_valid = (
				bool(cp.isfinite(component_embedding).all().item())
				and np.isfinite(scale_denominator_value)
				and scale_denominator_value > 0.0
				and np.isfinite(data_range_value)
				and data_range_value > 0.0
			)
			if scaling_is_valid:
				component_embedding *= data_range_value / scale_denominator_value
				scaling_is_valid = bool(cp.isfinite(component_embedding).all().item())
			if not scaling_is_valid:
				warnings.warn(
					"Component spectral scaling produced non-finite values; "
					"falling back to random component layout."
				)
				_set_diagnostic(diagnostics, "spectral_fallback", True)
				component_embedding = _random_component_layout(
					random_state,
					component_graph.shape[0],
					dim,
					data_range,
					cp.zeros(dim, dtype=cp.float32),
				)
			result[mask] = component_embedding + meta_embedding[label]
			if not bool(cp.isfinite(result[mask]).all().item()):
				raise FloatingPointError(
					f"component {label} layout is non-finite after scaling"
				)
			_add_timing(timing, "init_component_postprocess_time", perf_counter() - t0)

	return result


def _solve_eigsh_cupy(
	matrix,
	*,
	k,
	ncv,
	tol,
	maxiter,
	deterministic,
	matrix_is_canonical=False,
	diagnostics=None,
):
	"""Dispatch CuPy eigsh through the requested CSR SpMV policy."""

	initial_vector = cp.ones(matrix.shape[0], dtype=matrix.dtype)
	if deterministic:
		if matrix_is_canonical:
			if not cupyx.scipy.sparse.isspmatrix_csr(matrix):
				raise TypeError("canonical CUDA eigsh input must be a CSR matrix")
			csr_preparation = "upstream_canonical_reused"
			csr_solver_copy = False
		else:
			matrix = cupyx.scipy.sparse.csr_matrix(
				matrix,
				dtype=matrix.dtype,
				copy=True,
			)
			matrix.sum_duplicates()
			matrix.sort_indices()
			csr_preparation = "solver_canonical_copy"
			csr_solver_copy = True
		_set_diagnostic(
			diagnostics,
			"spectral_csr_preparation",
			csr_preparation,
		)
		_set_diagnostic(
			diagnostics,
			"spectral_csr_solver_copy",
			csr_solver_copy,
		)
		_, algorithm_name = deterministic_csr_spmv_algorithm()
		_set_diagnostic(
			diagnostics,
			"spectral_spmv_algorithm",
			algorithm_name,
		)
		_set_diagnostic(
			diagnostics,
			"spectral_spmv_bitwise_deterministic",
			True,
		)
		return eigsh_csr_alg2(
			matrix,
			k,
			which="SA",
			ncv=ncv,
			tol=tol,
			v0=initial_vector,
			maxiter=maxiter,
		)

	if cupyx.scipy.sparse.isspmatrix_csr(matrix):
		csr_preparation = "existing_csr_reused"
		csr_solver_copy = False
	else:
		matrix = cupyx.scipy.sparse.csr_matrix(
			matrix,
			dtype=matrix.dtype,
			copy=False,
		)
		csr_preparation = "converted_to_csr"
		csr_solver_copy = True
	_, algorithm_name = fast_csr_spmv_algorithm()
	_set_diagnostic(
		diagnostics,
		"spectral_csr_preparation",
		csr_preparation,
	)
	_set_diagnostic(
		diagnostics,
		"spectral_csr_solver_copy",
		csr_solver_copy,
	)
	_set_diagnostic(
		diagnostics,
		"spectral_spmv_algorithm",
		algorithm_name,
	)
	_set_diagnostic(
		diagnostics,
		"spectral_spmv_bitwise_deterministic",
		False,
	)
	return eigsh_csr_alg1(
		matrix,
		k,
		which="SA",
		ncv=ncv,
		tol=tol,
		v0=initial_vector,
		maxiter=maxiter,
	)


def _spectral_layout_cupy(
	data,
	graph,
	dim,
	random_state,
	metric="euclidean",
	metric_kwds={},
	init="random",
	method=None,
	tol=0.0,
	maxiter=0,
	ncv=None,
	auto_defaults=False,
	timing=None,
	diagnostics=None,
	deterministic=False,
):
	"""GPU version of umap.spectral._spectral_layout using CuPy
	"""
	t0 = perf_counter()
	n_components, labels = cupyx.scipy.sparse.csgraph.connected_components(graph)
	if deterministic and n_components > 1:
		labels = canonicalize_component_labels_cupy(labels, n_components)
		component_label_order = "minimum_vertex_index"
		component_labels_canonicalized = True
	elif n_components == 1:
		component_label_order = "trivial_single_component"
		component_labels_canonicalized = False
	else:
		component_label_order = "native_connected_components"
		component_labels_canonicalized = False
	_add_timing(timing, "init_connected_components_time", perf_counter() - t0)
	if diagnostics is not None and "spectral_connected_components" not in diagnostics:
		diagnostics["spectral_connected_components"] = int(n_components)
	_set_diagnostic_default(
		diagnostics,
		"spectral_component_label_order",
		component_label_order,
	)
	_set_diagnostic_default(
		diagnostics,
		"spectral_component_labels_canonicalized",
		component_labels_canonicalized,
	)
	feature_data_required = bool(n_components > 2 * dim)
	# Preserve the top-level graph decision while recursively embedding each
	# connected component. Component subgraphs do not need feature data, but
	# that must not overwrite the fact that the top-level component layout did.
	_set_diagnostic_default(
		diagnostics,
		"spectral_feature_data_required",
		feature_data_required,
	)
	if not feature_data_required:
		_set_diagnostic_default(
			diagnostics,
			"spectral_feature_data_materialized",
			False,
		)

	if n_components > 1:
		return multi_component_layout_cupy(
			data,
			graph,
			n_components,
			labels,
			dim,
			random_state,
			metric=metric,
			metric_kwds=metric_kwds,
			init=init,
			method=method,
			tol=tol,
			maxiter=maxiter,
			ncv=ncv,
			auto_defaults=auto_defaults,
			deterministic=deterministic,
			timing=timing,
			diagnostics=diagnostics,
		)

	component_n = int(graph.shape[0])
	k = dim + 1
	gen = random_state
	if component_n <= k + 1:
		_set_diagnostic(diagnostics, "spectral_small_component_fallback", True)
		t0 = perf_counter()
		result = gen.uniform(
			low=-10.0,
			high=10.0,
			size=(component_n, dim),
		).astype(cp.float32)
		_add_timing(timing, "init_component_random_time", perf_counter() - t0)
		return result

	if auto_defaults:
		resolved = resolve_spectral_defaults(
			n_samples=component_n,
			n_components=dim,
			backend="cuda",
		)
		method = resolved.method
		tol = resolved.tol
		maxiter = resolved.maxiter
		ncv = resolved.ncv
		_set_diagnostic(diagnostics, "spectral_auto_defaults_policy", resolved.policy)

	t0 = perf_counter()
	L, sqrt_deg = _build_normalized_laplacian_cupy(
		graph,
		deterministic=bool(deterministic),
		diagnostics=diagnostics,
	)
	_add_timing(timing, "init_laplacian_time", perf_counter() - t0)

	num_lanczos_vectors = resolve_component_ncv(
		component_n=component_n,
		k=k,
		user_ncv=ncv,
	)

	if not method:
		method = "eigsh"
	tol_value = float(tol or 1e-4)
	maxiter_value = int(maxiter or graph.shape[0] * 5)
	_set_diagnostic(diagnostics, "spectral_method", method)
	_set_diagnostic(diagnostics, "spectral_tol", tol_value)
	_set_diagnostic(diagnostics, "spectral_maxiter", maxiter_value)
	_set_diagnostic(diagnostics, "spectral_ncv", int(num_lanczos_vectors))
	_set_diagnostic(
		diagnostics,
		"spectral_deterministic_requested",
		bool(deterministic),
	)
	_set_diagnostic_default(diagnostics, "spectral_fallback", False)
	_set_diagnostic_default(diagnostics, "spectral_exception_type", None)
	_set_diagnostic_default(diagnostics, "spectral_exception_message", None)
	_set_diagnostic_default(diagnostics, "spectral_exceptions", [])

	try:
		t0 = perf_counter()
		if method == "lobpcg" and init == "random":
			X = gen.normal(size=(L.shape[0], k)).astype(cp.float32)
			X[:, 0] = sqrt_deg / cp.linalg.norm(sqrt_deg)
			_set_diagnostic(diagnostics, "spectral_initial_guess", "random")
			_set_diagnostic(
				diagnostics,
				"spectral_initial_guess_generated",
				True,
			)
		elif method == "lobpcg" and init == "tsvd":
			U, S, V = cp.linalg.svd(L.toarray(), full_matrices=False)
			X = U[:, :k] * S[:k]
			X[:, 0] = sqrt_deg / cp.linalg.norm(sqrt_deg)
			_set_diagnostic(diagnostics, "spectral_initial_guess", "tsvd")
			_set_diagnostic(
				diagnostics,
				"spectral_initial_guess_generated",
				True,
			)
		elif method == "lobpcg":
			raise ValueError(
				"The init parameter must be either 'random' or 'tsvd': "
				f"{init} is invalid."
			)
		else:
			X = None
			_set_diagnostic(diagnostics, "spectral_initial_guess", "none")
			_set_diagnostic(
				diagnostics,
				"spectral_initial_guess_generated",
				False,
			)
		_add_timing(timing, "init_solver_setup_time", perf_counter() - t0)

		t0 = perf_counter()
		if method == "eigsh":
			eigenvalues, eigenvectors = _solve_eigsh_cupy(
				L,
				k=k,
				ncv=num_lanczos_vectors,
				tol=tol_value,
				maxiter=maxiter_value,
				deterministic=bool(deterministic),
				matrix_is_canonical=bool(deterministic),
				diagnostics=diagnostics,
			)
		elif method == "lobpcg":
			_set_diagnostic(
				diagnostics,
				"spectral_spmv_algorithm",
				"cupy_lobpcg_default",
			)
			_set_diagnostic(
				diagnostics,
				"spectral_spmv_bitwise_deterministic",
				False,
			)
			with warnings.catch_warnings():
				warnings.filterwarnings(
					category=UserWarning,
					message=r"(?ms).*not reaching the requested tolerance",
					action="error",
				)
				eigenvalues, eigenvectors = cupyx.scipy.sparse.linalg.lobpcg(
					L,
					cp.asarray(X, dtype=cp.float32),
					largest=False,
					tol=tol_value,
					maxiter=maxiter_value,
				)
		else:
			raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")
		_add_timing(timing, "init_eigensolver_time", perf_counter() - t0)

		if not bool(cp.isfinite(eigenvalues).all().item()) or not bool(
			cp.isfinite(eigenvectors).all().item()
		):
			raise FloatingPointError(
				f"spectral eigensolver returned non-finite values for component size {component_n}"
			)

		t0 = perf_counter()
		order = cp.argsort(eigenvalues)[1:k]
		result = eigenvectors[:, order]
		if not bool(cp.isfinite(result).all().item()):
			raise FloatingPointError(
				f"spectral component result is non-finite for component size {component_n}"
			)
		_add_timing(timing, "init_postprocess_time", perf_counter() - t0)
		return result
	except Exception as exc:
		_set_diagnostic(diagnostics, "spectral_fallback", True)
		if diagnostics is not None:
			diagnostics.setdefault("spectral_exceptions", []).append(
				{
					"type": type(exc).__name__,
					"message": str(exc),
					"component_size": component_n,
				}
			)
			if diagnostics.get("spectral_exception_type") is None:
				diagnostics["spectral_exception_type"] = type(exc).__name__
				diagnostics["spectral_exception_message"] = str(exc)
		warnings.warn(
			"Spectral initialisation failed! The eigenvector solver\n"
			"failed. This is likely due to too small an eigengap. Consider\n"
			"adding some noise or jitter to your data.\n\n"
			"Falling back to random initialisation!"
		)
		# return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
		t0 = perf_counter()
		result = gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim)).astype(cp.float32)
		_add_timing(timing, "init_fallback_time", perf_counter() - t0)
		return result
		

def spectral_layout_cupy(
	data,
	graph,
	dim,
	random_state,
	metric="euclidean",
	metric_kwds={},
	method=None,
	tol=0.0,
	maxiter=0,
	ncv=None,
	auto_defaults=False,
	timing=None,
	diagnostics=None,
	deterministic=False,
):
	"""GPU version of umap.spectral.spectral_layout using CuPy
	"""

	# The graph is always needed on the device. Original feature data is moved
	# lazily only when more than 2 * dim connected components require centroid
	# placement in feature space.
	t0 = perf_counter()
	graph = cupyx.scipy.sparse.csr_matrix(graph, dtype=cp.float32)
	if isinstance(random_state, int):
		random_state = cp.random.RandomState(random_state)
	elif isinstance(random_state, np.random.RandomState):
		random_state = cp.random.RandomState(random_state.randint(0, 1000))
	elif random_state is None:
		random_state = cp.random
	_add_timing(timing, "init_transfer_time", perf_counter() - t0)
	_set_diagnostic(diagnostics, "spectral_result_device", "gpu")

	# 调用 _spectral_layout_cupy
	result = _spectral_layout_cupy(
		data,
		graph,
		dim,
		random_state,
		metric=metric,
		metric_kwds=metric_kwds,
		method=method,
		tol=tol,
		maxiter=maxiter,
		ncv=ncv,
		auto_defaults=auto_defaults,
		deterministic=deterministic,
		timing=timing,
		diagnostics=diagnostics,
	)

	return result
