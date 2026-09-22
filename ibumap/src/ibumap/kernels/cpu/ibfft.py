import os
from dataclasses import dataclass

import numpy as np
import numba
import pyfftw
from pyfftw.interfaces.numpy_fft import rfft2
from umap.utils import tau_rand_int
from time import perf_counter, time
from ..p2m import normalize_workspace_policy, resolve_p2m_mode
from ..fft_grid import (
	FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE,
	quantize_fft_box_width,
)
from ..._fft_kernel_cache import (
	DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES,
	FFTKernelLRUCache,
)

try:
	from ._p2m_atomic import p2m_atomic as _native_p2m_atomic
except ImportError:  # Pure-Python installs retain a deterministic fallback.
	_native_p2m_atomic = None
pyfftw.interfaces.cache.enable()
_n_fft_threads = int(os.environ.get("DRFFT_FFT_THREADS", str(min(8, os.cpu_count() or 1))))
pyfftw.config.NUM_THREADS = _n_fft_threads
_FFTW_PLANNER_EFFORT = os.environ.get("DRFFT_FFTW_PLANNER_EFFORT", "FFTW_ESTIMATE")
_FFTW_ALIGNMENT = int(os.environ.get("DRFFT_FFTW_ALIGNMENT", "64"))
_FFTW_PLAN_KEY_PREFIX = "ibfft.fftw.plan."
_CPU_FUSED_BOUNDS_MIN_POINTS = 500_000
_CPU_FUSED_BOUNDS_CHUNK_SIZE = 65_536
_CPU_BOUNDS_WORKSPACE_PREFIX = "ibfft.bounds_finite."

# ---------------------------------------------------------------------------
# Module-level FFT kernel LRU cache. Grid spacing is represented by a stable
# integer quantization level rather than an epoch-specific floating-point span.
# ---------------------------------------------------------------------------
_FFT_KERNEL_CACHE_MAXSIZE = DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES
_fft_kernel_cache = FFTKernelLRUCache(scope="module")
_ALLOWED_N_BOXES_PER_DIM = np.array(
	[
		50, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243,
		256, 288, 324, 384, 432, 576, 648, 768, 864, 972, 1152, 1296,
		1536, 1728, 1944, 2304, 2592, 3072, 3456, 3888, 4608, 5184,
		6144, 6912,
	],
	dtype=np.int32,
)


def _clear_fft_kernel_cache():
	_fft_kernel_cache.clear()


def _get_cached_fft_kernel(key):
	return _fft_kernel_cache.lookup(key).value


def _store_fft_kernel(key, value):
	_fft_kernel_cache.store(key, value)


def _complex_dtype_for(float_dtype):
	float_dtype = np.dtype(float_dtype)
	if float_dtype == np.dtype(np.float32):
		return np.dtype(np.complex64)
	if float_dtype == np.dtype(np.float64):
		return np.dtype(np.complex128)
	raise TypeError(f"unsupported FFT float dtype: {float_dtype}")


def _empty_aligned(shape, dtype):
	return pyfftw.empty_aligned(shape, dtype=np.dtype(dtype), n=_FFTW_ALIGNMENT)


def _is_reusable_array(arr, shape, dtype, *, aligned=False):
	return (
		arr is not None
		and arr.shape == tuple(shape)
		and arr.dtype == np.dtype(dtype)
		and arr.flags.c_contiguous
		and (not aligned or (arr.ctypes.data % _FFTW_ALIGNMENT) == 0)
	)


@numba.njit(parallel=True, cache=True)
def _bounds_finite_partials(
	values,
	chunk_size,
	partial_min,
	partial_max,
	partial_bad,
):
	"""Scan fixed chunks in parallel; the host merges them in index order."""

	flat = values.reshape(values.size)
	for chunk in numba.prange(partial_min.size):
		start = chunk * chunk_size
		stop = min(start + chunk_size, flat.size)
		local_min = np.inf
		local_max = -np.inf
		bad = 0
		for index in range(start, stop):
			value = flat[index]
			if np.isfinite(value):
				if value < local_min:
					local_min = value
				if value > local_max:
					local_max = value
			else:
				bad += 1
		partial_min[chunk] = local_min
		partial_max[chunk] = local_max
		partial_bad[chunk] = bad


def _bounds_scratch(workspace, chunks, dtype):
	workspace = {} if workspace is None else workspace
	shape = (int(chunks),)
	arrays = []
	for suffix, array_dtype in (
		("min", dtype),
		("max", dtype),
		("bad", np.dtype(np.int64)),
	):
		key = f"{_CPU_BOUNDS_WORKSPACE_PREFIX}{suffix}"
		array = workspace.get(key)
		if not _is_reusable_array(array, shape, array_dtype):
			array = np.empty(shape, dtype=array_dtype)
			workspace[key] = array
		arrays.append(array)
	return tuple(arrays)


def _fused_bounds_finite(values, workspace=None):
	"""Return min, max and non-finite count using a deterministic merge."""

	if values.size == 0:
		raise ValueError("embedding must not be empty")
	chunks = max(
		1,
		(int(values.size) + _CPU_FUSED_BOUNDS_CHUNK_SIZE - 1)
		// _CPU_FUSED_BOUNDS_CHUNK_SIZE,
	)
	partial_min, partial_max, partial_bad = _bounds_scratch(
		workspace,
		chunks,
		np.dtype(values.dtype),
	)
	_bounds_finite_partials(
		values,
		_CPU_FUSED_BOUNDS_CHUNK_SIZE,
		partial_min,
		partial_max,
		partial_bad,
	)
	minimum = np.inf
	maximum = -np.inf
	bad = 0
	for chunk in range(chunks):
		bad += int(partial_bad[chunk])
		if partial_min[chunk] < minimum:
			minimum = partial_min[chunk]
		if partial_max[chunk] > maximum:
			maximum = partial_max[chunk]
	return minimum, maximum, bad, chunks


def _fft_plan_workspace_key(shape, float_dtype):
	shape_token = "x".join(str(int(value)) for value in shape)
	return (
		f"{_FFTW_PLAN_KEY_PREFIX}{np.dtype(float_dtype).name}."
		f"{shape_token}.threads{int(_n_fft_threads)}.{_FFTW_PLANNER_EFFORT}"
	)


def _drop_stale_fft_plan_bundles(workspace, keep_key):
	for key in list(workspace.keys()):
		if key.startswith(_FFTW_PLAN_KEY_PREFIX) and key != keep_key:
			del workspace[key]


def _get_or_create_fft_plan_bundle(workspace, real_input, float_dtype):
	"""Return reusable pyFFTW R2C/C2R plans bound to workspace buffers."""

	key = _fft_plan_workspace_key(real_input.shape, float_dtype)
	bundle = workspace.get(key)
	if bundle is not None and bundle.get("input") is real_input:
		return bundle, True, 0.0

	build_start = perf_counter()
	complex_dtype = _complex_dtype_for(float_dtype)
	frequency_shape = (
		real_input.shape[0],
		real_input.shape[1],
		real_input.shape[2] // 2 + 1,
	)
	frequency = _empty_aligned(frequency_shape, complex_dtype)
	real_output = _empty_aligned(real_input.shape, float_dtype)
	forward_plan = pyfftw.FFTW(
		real_input,
		frequency,
		axes=(-2, -1),
		direction="FFTW_FORWARD",
		flags=(_FFTW_PLANNER_EFFORT,),
		threads=_n_fft_threads,
	)
	inverse_plan = pyfftw.FFTW(
		frequency,
		real_output,
		axes=(-2, -1),
		direction="FFTW_BACKWARD",
		flags=(_FFTW_PLANNER_EFFORT,),
		threads=_n_fft_threads,
	)
	bundle = {
		"input": real_input,
		"frequency": frequency,
		"output": real_output,
		"forward": forward_plan,
		"inverse": inverse_plan,
	}
	workspace[key] = bundle
	_drop_stale_fft_plan_bundles(workspace, key)
	return bundle, False, perf_counter() - build_start


def _validate_umap_kernel_subsampling(mode, radius_cells, points):
	if mode not in (None, "near_origin"):
		raise ValueError("ibfft_kernel_subsample_mode must be None or 'near_origin'")
	if (
		isinstance(radius_cells, (bool, np.bool_))
		or not isinstance(radius_cells, (int, np.integer))
		or radius_cells < 0
	):
		raise ValueError("ibfft_kernel_subsample_radius_cells must be a non-negative integer")
	if (
		isinstance(points, (bool, np.bool_))
		or not isinstance(points, (int, np.integer))
		or points < 1
	):
		raise ValueError("ibfft_kernel_subsample_points must be a positive integer")
	return mode, int(radius_cells), int(points)


def _umap_repulsion_kernel_value(dist2, a, b, gamma, epsilon):
	return (2.0 * gamma * b) / (
		(epsilon + dist2) * (a * np.power(dist2, b) + 1)
	)


def _build_umap_kernel_point_sampled(
	dist_matrix,
	umap_a,
	umap_b,
	umap_gamma,
	umap_epsilon,
	ibfft_kernel_clip,
):
	half_kernel = _umap_repulsion_kernel_value(
		dist_matrix,
		umap_a,
		umap_b,
		umap_gamma,
		umap_epsilon,
	)
	return np.clip(
		half_kernel,
		-float(ibfft_kernel_clip),
		float(ibfft_kernel_clip),
	)


def _build_umap_kernel_near_origin_subsampled(
	dist_matrix,
	h,
	umap_a,
	umap_b,
	umap_gamma,
	umap_epsilon,
	ibfft_kernel_clip,
	radius_cells,
	points,
):
	dtype = np.dtype(dist_matrix.dtype)
	half_kernel = _umap_repulsion_kernel_value(
		dist_matrix,
		umap_a,
		umap_b,
		umap_gamma,
		umap_epsilon,
	).copy()
	offsets = np.asarray(((np.arange(points) + 0.5) / points - 0.5) * h, dtype=dtype)
	row_limit = min(radius_cells + 1, half_kernel.shape[0])
	col_limit = min(radius_cells + 1, half_kernel.shape[1])
	for dy_index in range(row_limit):
		y = dy_index * h + offsets
		for dx_index in range(col_limit):
			x = dx_index * h + offsets
			sample_dist2 = y[:, None] * y[:, None] + x[None, :] * x[None, :]
			half_kernel[dy_index, dx_index] = np.mean(
				_umap_repulsion_kernel_value(
					sample_dist2,
					umap_a,
					umap_b,
					umap_gamma,
					umap_epsilon,
				)
			)
	return np.clip(
		half_kernel,
		-float(ibfft_kernel_clip),
		float(ibfft_kernel_clip),
	)


def _build_umap_kernel(
	dist_matrix,
	h,
	umap_a,
	umap_b,
	umap_gamma,
	umap_epsilon,
	ibfft_kernel_clip,
	ibfft_kernel_subsample_mode=None,
	ibfft_kernel_subsample_radius_cells=2,
	ibfft_kernel_subsample_points=4,
):
	mode, radius_cells, points = _validate_umap_kernel_subsampling(
		ibfft_kernel_subsample_mode,
		ibfft_kernel_subsample_radius_cells,
		ibfft_kernel_subsample_points,
	)
	if mode is None:
		return _build_umap_kernel_point_sampled(
			dist_matrix,
			umap_a,
			umap_b,
			umap_gamma,
			umap_epsilon,
			ibfft_kernel_clip,
		)
	return _build_umap_kernel_near_origin_subsampled(
		dist_matrix,
		h,
		umap_a,
		umap_b,
		umap_gamma,
		umap_epsilon,
		ibfft_kernel_clip,
		radius_cells,
		points,
	)


@dataclass(frozen=True)
class GridContext:
	"""Point-to-cell index for the square grid used by the current ibFFT call."""

	point_cell_x: np.ndarray
	point_cell_y: np.ndarray
	flat_cell_id: np.ndarray
	sorted_indices: np.ndarray
	cell_start: np.ndarray
	cell_end: np.ndarray
	density: np.ndarray
	min_xy: np.ndarray
	max_xy: np.ndarray
	cell_size: float
	n_cells_x: int
	n_cells_y: int
	build_time_s: float


def _build_grid_context(Y, box_idx, min_coord, max_coord, box_width, n_boxes_per_dim):
	"""Build cell lookup arrays from ibFFT's unpadded square embedding grid."""
	build_start = perf_counter()
	coord_dtype = np.dtype(Y.dtype)
	n_cells = int(n_boxes_per_dim)
	point_cells = np.asarray(box_idx[:Y.shape[0]], dtype=np.int32)
	flat_cell_id = point_cells[:, 0].astype(np.int64) * n_cells + point_cells[:, 1]
	sorted_indices = np.argsort(flat_cell_id, kind="stable").astype(np.int64, copy=False)
	density = np.bincount(flat_cell_id, minlength=n_cells * n_cells).astype(np.int64, copy=False)
	cell_end = np.cumsum(density, dtype=np.int64)
	cell_start = np.empty_like(cell_end)
	cell_start[0] = 0
	cell_start[1:] = cell_end[:-1]
	min_value = float(min_coord)
	max_value = float(max_coord)
	return GridContext(
		point_cell_x=point_cells[:, 0].copy(),
		point_cell_y=point_cells[:, 1].copy(),
		flat_cell_id=flat_cell_id,
		sorted_indices=sorted_indices,
		cell_start=cell_start,
		cell_end=cell_end,
		density=density,
		min_xy=np.array([min_value, min_value], dtype=coord_dtype),
		max_xy=np.array([max_value, max_value], dtype=coord_dtype),
		cell_size=float(box_width),
		n_cells_x=n_cells,
		n_cells_y=n_cells,
		build_time_s=perf_counter() - build_start,
	)


""" ibFFT utils """

@numba.njit(parallel=True, cache=True)
def Circulant_kernel_tilde(circulant_kernel_tilde, box_width, n_interpolation_points, n_interpolation_points_1d,  n_fft_coeffs, n_boxes_per_dim, N, a, c):
	whsquare = box_width / n_interpolation_points
	whsquare *= whsquare
	for i in numba.prange(n_interpolation_points_1d):
		for j in range(n_interpolation_points_1d):
			tmp = a * np.power(1.0 + (i * i + j * j) * whsquare, -c)
			circulant_kernel_tilde[(
				n_interpolation_points_1d + i)][(n_interpolation_points_1d + j)] = tmp
			circulant_kernel_tilde[(
				n_interpolation_points_1d - i)][(n_interpolation_points_1d + j)] = tmp
			circulant_kernel_tilde[(
				n_interpolation_points_1d + i)][(n_interpolation_points_1d - j)] = tmp
			circulant_kernel_tilde[(
				n_interpolation_points_1d - i)][(n_interpolation_points_1d - j)] = tmp


@numba.njit(parallel=True, cache=True)
def Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N):
	for i in numba.prange(N):
		box_idx[i][0] = max(
			0, min(int((Y[i][0] - min_coord) / (box_width)), n_boxes_per_dim - 1))
		box_idx[i][1] = max(
			0, min(int((Y[i][1] - min_coord) / (box_width)), n_boxes_per_dim - 1))


@numba.njit(parallel=True, cache=True)
def Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N):
	for i in numba.prange(N):
		j = box_idx[i][0]
		k = box_idx[i][1]
		y_in_box[i][0] = (Y[i][0] - box_width * j - min_coord)
		y_in_box[i][1] = (Y[i][1] - box_width * k - min_coord)


@numba.njit(parallel=True, cache=True)
def Interpolate(y_in_box, y_tilde_spacings, denominator, interpolated_values, n_interpolation_points, N):
	for i in numba.prange(N):
		for j in range(n_interpolation_points):
			# interpolated_values[i][j][0] = 1.0
			# interpolated_values[i][j][1] = 1.0
			for k in range(n_interpolation_points):
				if j != k:
					interpolated_values[i][j][0] *= y_in_box[i][0] - \
						y_tilde_spacings[k]
					interpolated_values[i][j][1] *= y_in_box[i][1] - \
						y_tilde_spacings[k]
			interpolated_values[i][j][0] /= denominator[j]
			interpolated_values[i][j][1] /= denominator[j]


@numba.njit(parallel=False, cache=True)
def Compute_w_coeff(w_coefficients, box_idx, chargesQij, interpolated_values, n_interpolation_points, n_boxes_per_dim, n_terms, N):
	# w_coefficients: (插值点总数, 插值点总数, squared_n_terms=3)
	for i in numba.prange(N):
		for j in range(n_interpolation_points):
			for k in range(n_interpolation_points):
				idj = box_idx[i][0] * n_interpolation_points + j	# 插值点在 x 轴上的全局索引
				idk = box_idx[i][1] * n_interpolation_points + k	# 插值点在 y 轴上的全局索引
				prob = interpolated_values[i][j][0] * \
					interpolated_values[i][k][1]
				for term in range(n_terms):
					w_coefficients[idj][idk][term] += prob * \
						chargesQij[i][term]							# chargesQij(N,3): 前两列是坐标，第三列是1


@numba.njit(parallel=False, cache=True)
def Compute_w_coeff_p1(w_coefficients, box_idx, chargesQij, n_terms, N):
	# p=1 is the default ibUMAP path. Its interpolation weights are exactly 1,
	# so the generic p^2 loop reduces to one stable point-order accumulation.
	for i in range(N):
		idj = box_idx[i][0]
		idk = box_idx[i][1]
		for term in range(n_terms):
			w_coefficients[idj][idk][term] += chargesQij[i][term]


@numba.njit(parallel=False, cache=True)
def Compute_mat_w_p1(mat_w, box_idx, Y, N):
	# p=1 can write directly to the padded term-first FFT input. This preserves
	# the old point-order accumulation while skipping ChargesQij, w_coefficients,
	# and the w_coefficients -> mat_w copy.
	one = np.float32(1.0)
	for i in range(N):
		idj = box_idx[i][0]
		idk = box_idx[i][1]
		mat_w[0][idj][idk] += Y[i][0]
		mat_w[1][idj][idk] += Y[i][1]
		mat_w[2][idj][idk] += one


@numba.njit(parallel=True, cache=True)
def Compute_mat_w_p1_segmented(mat_w, ordered_points, segment_cells,
								segment_starts, segment_ends, Y, n_boxes_per_dim):
	"""Reduce one independently owned cell per worker in stable point order."""
	for segment in numba.prange(segment_cells.shape[0]):
		cell = segment_cells[segment]
		idj = cell // n_boxes_per_dim
		idk = cell - idj * n_boxes_per_dim
		x_total = mat_w[0, idj, idk]
		y_total = mat_w[1, idj, idk]
		count = mat_w[2, idj, idk]
		for pos in range(segment_starts[segment], segment_ends[segment]):
			point = ordered_points[pos]
			x_total += Y[point, 0]
			y_total += Y[point, 1]
			count += 1
		mat_w[0, idj, idk] = x_total
		mat_w[1, idj, idk] = y_total
		mat_w[2, idj, idk] = count


def _p2m_segmented_cpu(mat_w, box_idx, Y, n_boxes_per_dim, workspace, policy):
	N = int(Y.shape[0])
	cell_ids = (
		box_idx[:N, 0].astype(np.int64, copy=False) * int(n_boxes_per_dim)
		+ box_idx[:N, 1]
	)
	order = np.argsort(cell_ids, kind='stable').astype(np.int64, copy=False)
	sorted_cells = cell_ids[order]
	if N:
		starts = np.flatnonzero(
			np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
		).astype(np.int64, copy=False)
		ends = np.r_[starts[1:], N].astype(np.int64, copy=False)
		segment_cells = sorted_cells[starts].astype(np.int64, copy=False)
	else:
		starts = ends = segment_cells = np.empty(0, dtype=np.int64)
	Compute_mat_w_p1_segmented(
		mat_w,
		order,
		segment_cells,
		starts,
		ends,
		Y,
		int(n_boxes_per_dim),
	)
	if policy != 'minimal':
		workspace['p2m.cell_ids'] = cell_ids
		workspace['p2m.order'] = order
		workspace['p2m.segment_cells'] = segment_cells
		workspace['p2m.segment_starts'] = starts
		workspace['p2m.segment_ends'] = ends
	return segment_cells, starts, ends


def _p2m_p1_cpu(mat_w, box_idx, Y, n_boxes_per_dim, mode, workspace, policy):
	if mode == 'serial':
		Compute_mat_w_p1(mat_w, box_idx, Y, int(Y.shape[0]))
		return
	if mode == 'segmented':
		_p2m_segmented_cpu(mat_w, box_idx, Y, n_boxes_per_dim, workspace, policy)
		return
	if mode == 'atomic':
		if _native_p2m_atomic is None:
			raise RuntimeError(
				"p2m_mode='atomic' requires the optional CPU atomic extension"
			)
		_native_p2m_atomic(mat_w, box_idx, Y)
		return
	raise ValueError(f"unsupported CPU P2M mode: {mode}")


@numba.njit(parallel=True, cache=True)
def PotentialsQij(potentialsQij, box_idx, interpolated_values, y_tilde_values, n_interpolation_points, n_boxes_per_dim, n_terms, N):
	Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim
	for i in numba.prange(N):
		for j in range(n_interpolation_points):
			for k in range(n_interpolation_points):
				# 插值点的全局索引，加上偏移量 Qij_len_per_dim ，使其能对应 y_tilde_values
				idj = box_idx[i][0] * n_interpolation_points + j + Qij_len_per_dim
				idk = box_idx[i][1] * n_interpolation_points + k + Qij_len_per_dim
				prob = interpolated_values[i][j][0] * \
						interpolated_values[i][k][1]
				for term in range(n_terms):
					potentialsQij[i][term] += prob * \
						y_tilde_values[term][idj][idk]


@numba.njit(parallel=True, cache=True)
def PotentialsQij_p1(potentialsQij, box_idx, y_tilde_values, n_boxes_per_dim, n_terms, N):
	# For p=1, every point reads the single interpolation node in its box.
	Qij_len_per_dim = n_boxes_per_dim
	for i in numba.prange(N):
		idj = box_idx[i][0] + Qij_len_per_dim
		idk = box_idx[i][1] + Qij_len_per_dim
		for term in range(n_terms):
			potentialsQij[i][term] += y_tilde_values[term][idj][idk]


@numba.njit(parallel=True, cache=True)
def NegF_p1(neg_f, box_idx, Y, y_tilde_values, n_boxes_per_dim, N):
	# Fuse p=1 M2P and final force formation:
	# neg_f = potential_common * Y - potential_xy.
	Qij_len_per_dim = n_boxes_per_dim
	for i in numba.prange(N):
		idj = box_idx[i][0] + Qij_len_per_dim
		idk = box_idx[i][1] + Qij_len_per_dim
		common = y_tilde_values[2][idj][idk]
		neg_f[i][0] = common * Y[i][0] - y_tilde_values[0][idj][idk]
		neg_f[i][1] = common * Y[i][1] - y_tilde_values[1][idj][idk]


@numba.njit(inline='always')
def _apply_fused_m2p_update(
	embedding, attr_force, i, raw_x, raw_y, alpha, neg_effect, clip_enabled, clip_norm
):
	# Preserve fallback operation order: raw * alpha, then degree effect, then
	# vector-norm clipping, followed by attr + repl application.
	force_x = raw_x * alpha
	force_y = raw_y * alpha
	force_x *= neg_effect
	force_y *= neg_effect
	if clip_enabled:
		norm = np.sqrt(force_x * force_x + force_y * force_y)
		scale = min(1.0, clip_norm / (norm + 1e-12))
		force_x *= scale
		force_y *= scale
	embedding[i][0] += attr_force[i][0] + force_x
	embedding[i][1] += attr_force[i][1] + force_y


@numba.njit(parallel=True, cache=True)
def NegF_p1_fused_update(
	embedding,
	attr_force,
	box_idx,
	Y,
	y_tilde_values,
	n_boxes_per_dim,
	N,
	alpha,
	neg_effects,
	clip_enabled,
	clip_norm,
):
	Qij_len_per_dim = n_boxes_per_dim
	for i in numba.prange(N):
		idj = box_idx[i][0] + Qij_len_per_dim
		idk = box_idx[i][1] + Qij_len_per_dim
		common = y_tilde_values[2][idj][idk]
		raw_x = common * Y[i][0] - y_tilde_values[0][idj][idk]
		raw_y = common * Y[i][1] - y_tilde_values[1][idj][idk]
		_apply_fused_m2p_update(
			embedding, attr_force, i, raw_x, raw_y, alpha,
			neg_effects[i], clip_enabled, clip_norm,
		)


@numba.njit(parallel=True, cache=True)
def PotentialsQij_fused_update(
	embedding,
	attr_force,
	box_idx,
	Y,
	interpolated_values,
	y_tilde_values,
	n_interpolation_points,
	n_boxes_per_dim,
	N,
	alpha,
	neg_effects,
	clip_enabled,
	clip_norm,
):
	Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim
	for i in numba.prange(N):
		potential_x = 0.0
		potential_y = 0.0
		potential_common = 0.0
		for j in range(n_interpolation_points):
			for k in range(n_interpolation_points):
				idj = box_idx[i][0] * n_interpolation_points + j + Qij_len_per_dim
				idk = box_idx[i][1] * n_interpolation_points + k + Qij_len_per_dim
				prob = (
					interpolated_values[i][j][0]
					* interpolated_values[i][k][1]
				)
				potential_x += prob * y_tilde_values[0][idj][idk]
				potential_y += prob * y_tilde_values[1][idj][idk]
				potential_common += prob * y_tilde_values[2][idj][idk]
		raw_x = potential_common * Y[i][0] - potential_x
		raw_y = potential_common * Y[i][1] - potential_y
		_apply_fused_m2p_update(
			embedding, attr_force, i, raw_x, raw_y, alpha,
			neg_effects[i], clip_enabled, clip_norm,
		)


""" t-FDP utils """

@numba.njit(parallel=True, cache=True)
def computebias(bias, edgesrc, edgetgt, n_vertices):
	for i in numba.prange(n_vertices):
		cntSrc = edgesrc[i + 1] - edgesrc[i]
		for k in range(edgesrc[i], edgesrc[i+1]):
			j = edgetgt[k]
			cntTgt = edgesrc[j + 1] - edgesrc[j]
			bias[k] = cntTgt / (cntSrc + cntTgt)


@numba.njit(parallel=True, cache=True)
def ApplyForce(embedding, dC, n_vertices):
	for i in numba.prange(n_vertices):
		mv0 = dC[i][0]
		mv1 = dC[i][1]
		d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-32
		R = min(d, 1.0)
		embedding[i][0] += mv0 / d * R
		embedding[i][1] += mv1 / d * R


""" t-FDP Attraction and Repulsion with Sampling """

@numba.njit(parallel=True, cache=True)
def AttrAndRelpForce(
	attr_force, dC, embedding, edgesrc, edgetgt, 
	bias, n_vertices, beta, d3alpha, 
	repl_force, paraFactor, gamma,
	epoch_itr, rng_state,
	epochs_per_sample, epoch_of_next_sample,
	epochs_per_negative_sample, epoch_of_next_negative_sample,
):
	for i in numba.prange(n_vertices):
		repl_force[i][0] = 0
		repl_force[i][1] = 0
		tempforce = 0.0
		tempforce1 = 0.0
		mvix = embedding[i][0]
		mviy = embedding[i][1]
		dcix = dC[i][0]
		dciy = dC[i][1]
		constant = np.float32(1.0)
		constant2 = np.float32(0.5)
		for k in range(edgesrc[i], edgesrc[i+1]):
			if epoch_of_next_sample[k] <= epoch_itr:
				epoch_of_next_sample[k] += epochs_per_sample[k]
				j = edgetgt[k]
				mv0 = mvix + dcix
				mv1 = mviy + dciy
				b = bias[k]
				mv0 -= (dC[j][0] + embedding[j][0])
				mv1 -= (dC[j][1] + embedding[j][1])
				R = beta / (constant + mv0 * mv0 + mv1 * mv1) + b
				tempforce -= R*mv0
				tempforce1 -= R*mv1
				# negative sampling for replusive force
				n_neg_samples = int( (epoch_itr - epoch_of_next_negative_sample[k]) / epochs_per_negative_sample[k] )
				epoch_of_next_negative_sample[k] += n_neg_samples * epochs_per_negative_sample[k]
				for neg in range(n_neg_samples):
					neg_j = tau_rand_int(rng_state) % n_vertices
					if neg_j == i: continue
					neg_mv0 = embedding[i][0] - embedding[neg_j][0]
					neg_mv1 = embedding[i][1] - embedding[neg_j][1]
					dsqare = neg_mv0 * neg_mv0 + neg_mv1 * neg_mv1
					R = d3alpha * paraFactor * np.power(1.0 + dsqare, -gamma)
					repl_force[i][0] += R * neg_mv0
					repl_force[i][1] += R * neg_mv1
		mv0 = d3alpha * tempforce
		mv1 = d3alpha * tempforce1
		d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-12
		R = min(d, 2000.0)
		attr_force[i][0] = mv0 / d * R
		attr_force[i][1] = mv1 / d * R


""" t-FDP Attraction """

@numba.njit(parallel=True, cache=True)
def AttrForce(
	attr_force, dC, embedding, edgesrc, edgetgt, 
	bias, n_vertices, beta, d3alpha,
):
	for i in numba.prange(n_vertices):
		tempforce = 0.0
		tempforce1 = 0.0
		mvix = embedding[i][0]
		mviy = embedding[i][1]
		dcix = dC[i][0]
		dciy = dC[i][1]
		constant = np.float32(1.0)
		constant2 = np.float32(0.5)
		for k in range(edgesrc[i], edgesrc[i+1]):
			j = edgetgt[k]
			mv0 = mvix + dcix
			mv1 = mviy + dciy
			b = bias[k]
			mv0 -= (dC[j][0] + embedding[j][0])
			mv1 -= (dC[j][1] + embedding[j][1])
			R = beta / (constant + mv0 * mv0 + mv1 * mv1) + b
			tempforce -= R*mv0
			tempforce1 -= R*mv1
		mv0 = d3alpha * tempforce
		mv1 = d3alpha * tempforce1
		d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-12
		R = min(d, 2000.0)
		attr_force[i][0] = mv0 / d * R
		attr_force[i][1] = mv1 / d * R


@numba.njit(parallel=True, cache=True)
def AttrForce_sampling_effect(
	attr_force, dC, embedding, edgesrc, edgetgt, 
	bias, n_vertices, beta, d3alpha, pos_effects
):
	for i in numba.prange(n_vertices):
		tempforce = 0.0
		tempforce1 = 0.0
		mvix = embedding[i][0]
		mviy = embedding[i][1]
		dcix = dC[i][0]
		dciy = dC[i][1]
		constant = np.float32(1.0)
		constant2 = np.float32(0.5)
		for k in range(edgesrc[i], edgesrc[i+1]):
			j = edgetgt[k]
			mv0 = mvix + dcix
			mv1 = mviy + dciy
			b = bias[k]
			mv0 -= (dC[j][0] + embedding[j][0])
			mv1 -= (dC[j][1] + embedding[j][1])
			R = beta / (constant + mv0 * mv0 + mv1 * mv1) + b
			tempforce -= R*mv0 * pos_effects[k]
			tempforce1 -= R*mv1 * pos_effects[k]
		mv0 = d3alpha * tempforce
		mv1 = d3alpha * tempforce1
		d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-12
		R = min(d, 2000.0)
		attr_force[i][0] = mv0 / d * R
		attr_force[i][1] = mv1 / d * R


@numba.njit(parallel=True, cache=True)
def AttrForce_sampling(
	attr_force, dC, embedding, edgesrc, edgetgt, 
	bias, n_vertices, beta, d3alpha, epoch_itr,
	epochs_per_sample, epoch_of_next_sample,
	whether_known_points,
	known_points_positions,
	known_points_reverse_index,
	soft_constraint=False,
	constraint_weight=0.1,
):
	for i in numba.prange(n_vertices):
		tempforce0 = 0.0
		tempforce1 = 0.0
		mvix = embedding[i][0]
		mviy = embedding[i][1]
		dcix = dC[i][0]
		dciy = dC[i][1]
		constant = np.float32(1.0)
		constant2 = np.float32(0.5)
		i_known = whether_known_points[i]
		if soft_constraint and i_known:
			pidx = known_points_reverse_index[i]
			pull_0 = constraint_weight * (known_points_positions[pidx][0] - embedding[i][0])
			pull_1 = constraint_weight * (known_points_positions[pidx][1] - embedding[i][1])
			tempforce0 += pull_0
			tempforce1 += pull_1
		for k in range(edgesrc[i], edgesrc[i+1]):
			if epoch_of_next_sample[k] <= epoch_itr:
				epoch_of_next_sample[k] += epochs_per_sample[k]
				j = edgetgt[k]
				mv0 = mvix + dcix
				mv1 = mviy + dciy
				b = bias[k]
				mv0 -= (dC[j][0] + embedding[j][0])
				mv1 -= (dC[j][1] + embedding[j][1])
				R = beta / (constant + mv0 * mv0 + mv1 * mv1) + b
				tempforce0 -= R*mv0
				tempforce1 -= R*mv1
		mv0 = d3alpha * tempforce0
		mv1 = d3alpha * tempforce1
		d = np.sqrt(mv0 * mv0 + mv1 * mv1) + 1e-12
		R = min(d, 2000.0)
		attr_force[i][0] = mv0 / d * R
		attr_force[i][1] = mv1 / d * R


""" t-FDP Repulsion """

@numba.njit(parallel=True, cache=True)
def ReplForce(
	repl_force, embedding, n_vertices, paraFactor, gamma, d3alpha
):
	for i in numba.prange(n_vertices):
		# repl_force[i][0] = 0
		# repl_force[i][1] = 0
		for j in range(n_vertices):
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dsqare = mv0 * mv0 + mv1 * mv1
			R = d3alpha * paraFactor * np.power(1.0 + dsqare, -gamma)
			repl_force[i][0] += R * mv0
			repl_force[i][1] += R * mv1


@numba.njit(parallel=True, cache=True)
def ReplForce_sampling(
	repl_force, embedding, edgesrc, n_vertices, 
	paraFactor, gamma, d3alpha, epoch_itr, rng_state, 
	epochs_per_sample, epoch_of_next_sample,
	epochs_per_negative_sample, epoch_of_next_negative_sample,
):
	for i in numba.prange(n_vertices):
		# repl_force[i][0] = 0
		# repl_force[i][1] = 0
		for k in range(edgesrc[i], edgesrc[i+1]):
			if epoch_of_next_sample[k] <= epoch_itr:
				epoch_of_next_sample[k] += epochs_per_sample[k]
				# negative sampling for replusive force
				n_neg_samples = int( (epoch_itr - epoch_of_next_negative_sample[k]) / epochs_per_negative_sample[k] )
				epoch_of_next_negative_sample[k] += n_neg_samples * epochs_per_negative_sample[k]
				for neg in range(n_neg_samples):
					neg_j = tau_rand_int(rng_state) % n_vertices
					if neg_j == i: continue
					neg_mv0 = embedding[i][0] - embedding[neg_j][0]
					neg_mv1 = embedding[i][1] - embedding[neg_j][1]
					dsqare = neg_mv0 * neg_mv0 + neg_mv1 * neg_mv1
					R = d3alpha * paraFactor * np.power(1.0 + dsqare, -gamma)
					repl_force[i][0] += R * neg_mv0
					repl_force[i][1] += R * neg_mv1


def ibFFT_repulsive_sampling(
	Y,
	n_interpolation_points,
	intervals_per_integer,
	min_num_intervals,
	gamma,
	paraFactor, 			# for tFDP kernel
	probabilities = None, 	# Sampling probabilities
	n_boxes_per_dim = 1.0,	# 为 int 时指定，为 float 时，根据 N 和 Y 计算
	kernel_method = 'tFDP',	# t-FDP, UMAP_kernel, UMAP_gauss
	umap_a = 1.0,
	umap_b = 1.0,
	umap_gamma = 1.0,
	umap_epsilon = 0.001,
	gauss_sigma = 1.0,
	ibfft_kernel_clip = 1.0,
	# H-2 FIX: optional mutable dict for workspace buffer reuse.
	# The function will read existing arrays from the dict (reusing them when
	# shapes match) and write back any newly allocated buffers so subsequent
	# calls can reuse them without re-allocating.
	_workspace = None,
	return_grid_context = False,
	ibfft_kernel_subsample_mode = None,
	ibfft_kernel_subsample_radius_cells = 2,
	ibfft_kernel_subsample_points = 4,
	random_state = None,
	deterministic = False,
	p2m_mode = 'auto',
	workspace_policy = 'auto',
	workspace_limit_bytes = None,
	fft_kernel_cache = None,
	fft_kernel_cache_policy = 'auto',
	fft_kernel_cache_limit_bytes = None,
	fft_kernel_cache_max_entries = None,
	p2m_diagnostics = None,
	timing_diagnostics = None,
	fused_update = None,
):
	ibfft_total_start = perf_counter()
	timing_values = {}

	def _timed(name, start):
		timing_values[name] = timing_values.get(name, 0.0) + (perf_counter() - start)

	def _finish_timing():
		if timing_diagnostics is None:
			return
		total = perf_counter() - ibfft_total_start
		exclusive = sum(timing_values.values())
		timing_values['ibfft_other_time_s'] = max(0.0, total - exclusive)
		timing_values['ibfft_total_time_s'] = total
		timing_diagnostics.update(
			{key: float(value) for key, value in timing_values.items()}
		)

	setup_start = perf_counter()
	N = Y.shape[0]
	float_dtype = np.dtype(Y.dtype)
	if float_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
		float_dtype = np.dtype(np.float32)
		Y = np.asarray(Y, dtype=float_dtype)
	_ws = _workspace if _workspace is not None else {}
	workspace_policy = normalize_workspace_policy(workspace_policy)
	use_fused_bounds = (
		N >= _CPU_FUSED_BOUNDS_MIN_POINTS
		and Y.flags.c_contiguous
		and Y.size > 0
	)
	if use_fused_bounds:
		bounds_workspace = None if workspace_policy == 'minimal' else _ws
		full_min_coord, full_max_coord, nonfinite_count, bounds_chunks = (
			_fused_bounds_finite(Y, bounds_workspace)
		)
	else:
		full_min_coord = None
		full_max_coord = None
		bounds_chunks = 0
		nonfinite_count = int(np.count_nonzero(~np.isfinite(Y)))
	if p2m_diagnostics is not None:
		p2m_diagnostics.update({
			'bounds_finite_fused_reduction': bool(use_fused_bounds),
			'bounds_finite_backend': (
				'fixed_chunk_numba' if use_fused_bounds else 'numpy'
			),
			'bounds_finite_chunks': int(bounds_chunks),
			'bounds_finite_min_points': int(_CPU_FUSED_BOUNDS_MIN_POINTS),
		})
	if nonfinite_count:
		raise FloatingPointError(
			"ibFFT input embedding contains "
			f"{nonfinite_count} non-finite value(s)"
		)
	N_original = N   # always equals len(Y) before any subsetting
	Y_original = Y
	fused_update_enabled = fused_update is not None
	if fused_update_enabled:
		if probabilities is not None or return_grid_context:
			raise ValueError(
				"fused ibFFT update requires exact repulsion without grid context"
			)
		fused_embedding = fused_update['embedding']
		fused_attr_force = fused_update['attr_force']
		fused_alpha = float(fused_update['alpha'])
		fused_neg_effects = fused_update['neg_effects']
		fused_clip_norm = fused_update.get('clip_norm')
		fused_clip_enabled = fused_clip_norm is not None
		fused_clip_norm = 0.0 if fused_clip_norm is None else float(fused_clip_norm)
		if fused_embedding is not Y_original:
			raise ValueError("fused ibFFT update must target the input embedding")
	if kernel_method == 'UMAP_kernel':
		(
			ibfft_kernel_subsample_mode,
			ibfft_kernel_subsample_radius_cells,
			ibfft_kernel_subsample_points,
		) = _validate_umap_kernel_subsampling(
			ibfft_kernel_subsample_mode,
			ibfft_kernel_subsample_radius_cells,
			ibfft_kernel_subsample_points,
		)

	# 0. Sampling: 以 degrees 为概率，随机决定样本是否参与计算：
	if probabilities is not None:
		if random_state is None:
			if_apply_point = np.random.rand(N) <= probabilities
		else:
			if_apply_point = random_state.random_sample(N) <= probabilities
		active_points_idx = np.where(if_apply_point)[0]
		# Y is rebound to a smaller view; N_original keeps the full count.
		Y = Y[active_points_idx]
		N = Y.shape[0]

	# 1.初始化变量和参数：
	# 计算最大和最小坐标以及其他一些初始参数，例如：
	# n_boxes_per_dim: 每个维度的盒子数量
	if probabilities is None and use_fused_bounds:
		max_coord = full_max_coord
		min_coord = full_min_coord
	else:
		max_coord = Y.max()
		min_coord = Y.min()

	n_boxes_scale = float(n_boxes_per_dim)
	boxes_1 = np.sqrt(16 * N)
	boxes_2 = np.sqrt(4 * N / np.log(N))
	boxes_3 = (max_coord.item() - min_coord.item()) / intervals_per_integer
	n_temp = int(np.minimum(boxes_1,
							np.maximum(boxes_2,
										np.maximum(min_num_intervals,
													boxes_3)
							)
		))
	n_temp = int(n_boxes_scale * n_temp)
	# Select the smallest supported FFT grid strictly above the estimate,
	# capping at the largest supported size.
	if n_temp >= _ALLOWED_N_BOXES_PER_DIM[-1]:
		n_boxes_per_dim = int(_ALLOWED_N_BOXES_PER_DIM[-1])
	else:
		n_boxes_per_dim = int(
			_ALLOWED_N_BOXES_PER_DIM[_ALLOWED_N_BOXES_PER_DIM > n_temp][0]
		)

	####
	squared_n_terms = 3
	n_terms = squared_n_terms

	# 计算箱子的宽度、FFT系数的数量，以及一些与插值点和核相关的参数
	grid_span = max_coord.item() - min_coord.item()
	raw_box_width, quantized_box_width, grid_quantization_level = (
		quantize_fft_box_width(grid_span, n_boxes_per_dim, float_dtype)
	)
	box_width = float_dtype.type(quantized_box_width)						# 量化后的盒子宽度
	n_boxes = n_boxes_per_dim * n_boxes_per_dim								# 盒子总数
	n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim	# 插值点总数
	n_fft_coeffs = 2 * n_interpolation_points_1d							# FFT 系数总数

	whsquare = box_width / n_interpolation_points							# 插值点间隔
	whsquare *= whsquare													# 插值点间隔的平方
	h = 1.0 / n_interpolation_points * box_width							# 插值点间隔
	y_tilde_spacings = np.array(
		np.arange(n_interpolation_points) * h + h/2, dtype=float_dtype)		# 每个盒子内的插值点坐标（一维）


	# -----------------------------------------------------------------------
	# Cache fft_kernel_tilde across epochs when the quantized grid and every
	# kernel parameter are equal.
	# -----------------------------------------------------------------------
	_cache_key = (
		float_dtype.name,
		int(grid_quantization_level),
		FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE,
		int(n_boxes_per_dim),
		int(n_interpolation_points),
		kernel_method,
		float(umap_a), float(umap_b), float(umap_gamma), float(umap_epsilon),
		float(ibfft_kernel_clip),
		ibfft_kernel_subsample_mode,
		ibfft_kernel_subsample_radius_cells if ibfft_kernel_subsample_mode is not None else None,
		ibfft_kernel_subsample_points if ibfft_kernel_subsample_mode is not None else None,
		float(gamma), float(paraFactor), float(gauss_sigma),
	)
	resolved_fft_kernel_cache = (
		fft_kernel_cache if fft_kernel_cache is not None else _fft_kernel_cache
	)
	resolved_cache_policy = resolved_fft_kernel_cache.configure(
		policy=fft_kernel_cache_policy,
		limit_bytes=fft_kernel_cache_limit_bytes,
		max_entries=fft_kernel_cache_max_entries,
	)
	cache_lookup = resolved_fft_kernel_cache.lookup(_cache_key)
	fft_kernel_tilde = cache_lookup.value
	kernel_cache_hit = cache_lookup.hit
	if p2m_diagnostics is not None:
		cache_snapshot = resolved_fft_kernel_cache.snapshot()
		p2m_diagnostics.update({
			'grid_raw_box_width': float(raw_box_width),
			'grid_box_width': float(box_width),
			'grid_quantization_level': int(grid_quantization_level),
			'grid_quantization_steps_per_octave': (
				FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE
			),
			'kernel_cache_hit': bool(kernel_cache_hit),
			'kernel_cache_exact_hit': bool(cache_lookup.hit),
			'kernel_cache_miss_type': cache_lookup.miss_type,
			'kernel_cache_key_token': cache_lookup.key_token,
			'kernel_cache_entry_bytes': int(cache_lookup.entry_bytes),
			'kernel_cache_policy': str(resolved_cache_policy),
			'kernel_cache_scope': str(cache_snapshot['scope']),
			'kernel_cache_entries': int(cache_snapshot['entries']),
			'kernel_cache_bytes': int(cache_snapshot['bytes']),
			'kernel_cache_peak_entries': int(cache_snapshot['peak_entries']),
			'kernel_cache_peak_bytes': int(cache_snapshot['peak_bytes']),
			'kernel_cache_limit_bytes': cache_snapshot['limit_bytes'],
			'kernel_cache_max_entries': cache_snapshot['max_entries'],
			'kernel_cache_seen_exact_keys': int(cache_snapshot['seen_exact_keys']),
			'kernel_cache_evictions': 0,
			'kernel_cache_evicted_bytes': 0,
			'kernel_cache_store_oversized': False,
		})
	_timed('setup_time_s', setup_start)

	if fft_kernel_tilde is None:
		kernel_build_start = perf_counter()
		# 2.计算矩阵核 half_kernel, shape: (n_interpolation_points_1d, n_interpolation_points_1d)
		matrix = np.arange(n_interpolation_points_1d)**2 + \
				 (np.arange(n_interpolation_points_1d)**2).reshape(-1, 1)
		dist_matrix = (matrix * whsquare).astype(float_dtype, copy=False)

		if kernel_method == 'tFDP':
			if int(gamma) == gamma:
				half_kernel = paraFactor / ((1.0 + dist_matrix) ** int(gamma))
			else:
				half_kernel = paraFactor * np.power(1.0 + dist_matrix, -gamma)

		# 套用 UMAP 使用的斥力公式
		elif kernel_method == 'UMAP_kernel':
			half_kernel = _build_umap_kernel(
				dist_matrix,
				h,
				umap_a,
				umap_b,
				umap_gamma,
				umap_epsilon,
				ibfft_kernel_clip,
				ibfft_kernel_subsample_mode,
				ibfft_kernel_subsample_radius_cells,
				ibfft_kernel_subsample_points,
			)

		elif kernel_method == 'UMAP_gauss':
			epsilon = 0.001
			w_ij = np.exp( - dist_matrix / (2 * gauss_sigma ** 2) )
			half_kernel = (w_ij / np.clip(1.0 - w_ij, epsilon, np.inf)) / (gauss_sigma ** 2)

		# 3.计算 Circulant Kernel
		circulant_kernel_tilde = np.zeros(
			(n_fft_coeffs, n_fft_coeffs), dtype=float_dtype)
		circulant_kernel_tilde[n_interpolation_points_1d:,
							   n_interpolation_points_1d:] = half_kernel
		circulant_kernel_tilde[1:n_interpolation_points_1d+1,
							   n_interpolation_points_1d:] = np.flipud(half_kernel)
		circulant_kernel_tilde[n_interpolation_points_1d:,
							   1:n_interpolation_points_1d+1] = np.fliplr(half_kernel)
		circulant_kernel_tilde[1:n_interpolation_points_1d+1,
								   1:n_interpolation_points_1d+1] = np.fliplr(np.flipud(half_kernel))
		_timed('kernel_build_time_s', kernel_build_start)
		kernel_fft_start = perf_counter()
		fft_kernel_tilde = rfft2(circulant_kernel_tilde)
		_timed('kernel_fft_time_s', kernel_fft_start)
		cache_store = resolved_fft_kernel_cache.store(_cache_key, fft_kernel_tilde)
		if p2m_diagnostics is not None:
			cache_snapshot = resolved_fft_kernel_cache.snapshot()
			p2m_diagnostics.update({
				'kernel_cache_entry_bytes': int(cache_store.entry_bytes),
				'kernel_cache_entries': int(cache_snapshot['entries']),
				'kernel_cache_bytes': int(cache_snapshot['bytes']),
				'kernel_cache_peak_entries': int(cache_snapshot['peak_entries']),
				'kernel_cache_peak_bytes': int(cache_snapshot['peak_bytes']),
				'kernel_cache_seen_exact_keys': int(cache_snapshot['seen_exact_keys']),
				'kernel_cache_evictions': int(cache_store.evictions),
				'kernel_cache_evicted_bytes': int(cache_store.evicted_bytes),
				'kernel_cache_store_oversized': bool(cache_store.oversized),
			})
	else:
		timing_values['kernel_build_time_s'] = 0.0
		timing_values['kernel_fft_time_s'] = 0.0


	# -----------------------------------------------------------------------
	# H-2 FIX: Use workspace dict for buffer reuse across epochs.
	# _workspace is a mutable dict shared by the caller. On first call (or when
	# shapes change) we allocate fresh arrays and write them back to the dict so
	# subsequent calls can reuse them without heap allocation.
	#
	# A-FIX (Round-4): Workspace stability for sampling mode.
	# When probabilities is not None, N (active count) varies each epoch
	# (e.g. 1688–1854 for MNIST 5k), causing per-epoch re-allocation of all
	# point-indexed buffers (ChargesQij, box_idx, y_in_box, interpolate_values,
	# potentialsQij, neg_f — ~116KB each epoch).
	# Fix: use N_original (= len(Y) before sampling) as the workspace size key
	# for point-indexed arrays and pass N (active count) as the computation
	# size to the numba kernels.  Buffers are allocated at N_original size once,
	# then only the first N rows are touched per epoch.  The numba kernels
	# already accept N as an explicit argument, so passing a smaller N to a
	# larger buffer is safe (extra rows are never read or written).
	# -----------------------------------------------------------------------
	# performance reserves full-N point buffers. auto retains a high-water
	# active capacity; minimal uses exactly the current active capacity.
	N_ws = N_original if workspace_policy == 'performance' else N
	use_p1_fast_path = int(n_interpolation_points) == 1

	def _get_or_alloc(key, shape, dtype=None, active_capacity=False, aligned=False):
		arr = _ws.get(key)
		dtype = float_dtype if dtype is None else np.dtype(dtype)
		if active_capacity and workspace_policy == 'auto':
			compatible = (
				arr is not None
				and arr.dtype == dtype
				and arr.shape[1:] == shape[1:]
				and arr.flags.c_contiguous
				and (not aligned or (arr.ctypes.data % _FFTW_ALIGNMENT) == 0)
			)
			if compatible and arr.shape[0] >= shape[0]:
				return arr
			capacity = shape[0]
			if compatible:
				capacity = max(capacity, arr.shape[0])
			alloc_shape = (capacity, *shape[1:])
			arr = (
				_empty_aligned(alloc_shape, dtype)
				if aligned
				else np.empty(alloc_shape, dtype=dtype)
			)
			_ws[key] = arr
		elif not _is_reusable_array(arr, shape, dtype, aligned=aligned):
			arr = (
				_empty_aligned(shape, dtype)
				if aligned
				else np.empty(shape, dtype=dtype)
			)
			_ws[key] = arr
		return arr

	point_setup_start = perf_counter()
	if not use_p1_fast_path:
		# ChargesQij: N_ws*3 buffer, only first N rows used per epoch.
		# Initialise to 1.0 (weight column), then overwrite [:N, :2] with Y.
		ChargesQij = _get_or_alloc(
			'ChargesQij', (N_ws, squared_n_terms), active_capacity=True
		)
		N_ws = ChargesQij.shape[0]
		ChargesQij[:N].fill(1.0)
		ChargesQij[:N, :2] = Y

	# 4.匹配点和盒子：box_idx 和 y_in_box
	box_idx = _get_or_alloc(
		'box_idx', (N_ws, 2), dtype=np.int32, active_capacity=True
	)
	Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N)

	if not use_p1_fast_path:
		y_in_box = _get_or_alloc('y_in_box', (N_ws, 2), active_capacity=True)
		y_in_box[:N].fill(0.0)
		Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N)

		# 5.计算插值权重的分母 denominator
		denominator_sub = (y_tilde_spacings.reshape(-1, 1) - y_tilde_spacings)
		np.fill_diagonal(denominator_sub, 1)
		denominator = denominator_sub.prod(axis=0)

		# 6.为每个点 Y[i] 计算其插值权重 interpolate_values
		# NOTE: interpolate_values must be initialised to 1.0, not 0.0,
		# because Interpolate() multiplies (not adds) into it.
		# Only the first N rows are used; initialise just those rows for safety.
		interpolate_values = _get_or_alloc(
			'interpolate_values',
			(N_ws, n_interpolation_points, 2),
			active_capacity=True,
		)
		interpolate_values[:N].fill(1.0)  # critical: must be 1.0, not 0.0
		Interpolate(y_in_box, y_tilde_spacings, denominator,
					interpolate_values, n_interpolation_points, N)
	_timed('point_setup_time_s', point_setup_start)

	# 7.计算权重系数 w_coefficients
	_wc_dim = n_boxes_per_dim * n_interpolation_points
	p2m_elapsed = 0.0
	if not use_p1_fast_path:
		p2m_prepare_start = perf_counter()
		w_coefficients = _get_or_alloc('w_coefficients', (_wc_dim, _wc_dim, squared_n_terms))
		w_coefficients.fill(0.0)
		Compute_w_coeff(w_coefficients, box_idx, ChargesQij, interpolate_values,
						n_interpolation_points, n_boxes_per_dim, n_terms, N)
		p2m_elapsed += perf_counter() - p2m_prepare_start

	# 8.使用 FFT 计算卷积
	_mw_dim = 2 * n_boxes_per_dim * n_interpolation_points
	mat_w = _get_or_alloc('mat_w', (n_terms, _mw_dim, _mw_dim), aligned=True)
	fft_plan_bundle, fft_plan_cache_hit, fft_plan_build_time = (
		_get_or_create_fft_plan_bundle(_ws, mat_w, float_dtype)
	)
	timing_values['fft_plan_build_time_s'] = (
		timing_values.get('fft_plan_build_time_s', 0.0)
		+ float(fft_plan_build_time)
	)
	if p2m_diagnostics is not None:
		p2m_diagnostics['fft_plan_cache_hit'] = bool(fft_plan_cache_hit)
		p2m_diagnostics['fftw_planner_effort'] = str(_FFTW_PLANNER_EFFORT)
	mesh_clear_start = perf_counter()
	mat_w.fill(0.0)
	_timed('mesh_clear_time_s', mesh_clear_start)
	# fill upper-left quadrant of each term-slice
	p2m_start = perf_counter()
	if use_p1_fast_path:
		resolution = resolve_p2m_mode(
			p2m_mode,
			deterministic=bool(deterministic),
			device='cpu',
			n_points=N,
			workspace_limit_bytes=workspace_limit_bytes,
			atomic_available=_native_p2m_atomic is not None,
		)
		_p2m_p1_cpu(
			mat_w,
			box_idx,
			Y,
			n_boxes_per_dim,
			resolution.resolved,
			_ws,
			workspace_policy,
		)
	else:
		if p2m_mode not in ('auto', 'serial'):
			raise NotImplementedError(
				"non-serial CPU P2M modes currently require "
				"n_interpolation_points=1"
			)
		resolution = None
		for _t in range(n_terms):
			mat_w[_t, :_wc_dim, :_wc_dim] = w_coefficients[:, :, _t]
	p2m_elapsed += perf_counter() - p2m_start
	timing_values['p2m_time_s'] = float(p2m_elapsed)
	if p2m_diagnostics is not None:
		p2m_diagnostics['time_s'] = float(p2m_elapsed)
		p2m_diagnostics['requested_mode'] = str(p2m_mode)
		p2m_diagnostics['resolved_mode'] = (
			'serial' if resolution is None else resolution.resolved
		)
		p2m_diagnostics['reason'] = (
			'generic_interpolation' if resolution is None else resolution.reason
		)
		p2m_diagnostics['estimated_workspace_bytes'] = (
			0 if resolution is None else resolution.estimated_workspace_bytes
		)
	fft_forward_start = perf_counter()
	fft_w = fft_plan_bundle['forward']()
	_timed('fft_forward_time_s', fft_forward_start)
	fft_multiply_start = perf_counter()
	fft_w *= fft_kernel_tilde
	_timed('fft_multiply_time_s', fft_multiply_start)
	fft_inverse_start = perf_counter()
	output = fft_plan_bundle['inverse']()
	_timed('fft_inverse_time_s', fft_inverse_start)

	# 9.计算最终每个点在 x 和 y 方向上的斥力
	m2p_start = perf_counter()
	if use_p1_fast_path:
		if fused_update_enabled:
			NegF_p1_fused_update(
				fused_embedding,
				fused_attr_force,
				box_idx,
				Y,
				output,
				n_boxes_per_dim,
				N,
				fused_alpha,
				fused_neg_effects,
				fused_clip_enabled,
				fused_clip_norm,
			)
		else:
			neg_f_buf = _get_or_alloc('neg_f', (N_ws, 2), active_capacity=True)
			NegF_p1(neg_f_buf, box_idx, Y, output, n_boxes_per_dim, N)
	else:
		if fused_update_enabled:
			PotentialsQij_fused_update(
				fused_embedding,
				fused_attr_force,
				box_idx,
				Y,
				interpolate_values,
				output,
				n_interpolation_points,
				n_boxes_per_dim,
				N,
				fused_alpha,
				fused_neg_effects,
				fused_clip_enabled,
				fused_clip_norm,
			)
		else:
			neg_f_buf = _get_or_alloc('neg_f', (N_ws, 2), active_capacity=True)
			potentialsQij = _get_or_alloc(
				'potentialsQij', (N_ws, n_terms), active_capacity=True
			)
			potentialsQij[:N].fill(0.0)
			PotentialsQij(potentialsQij, box_idx, interpolate_values,
						output, n_interpolation_points, n_boxes_per_dim, n_terms, N)
			# Operate on views of the first N rows to avoid creating temporaries.
			PotentialsCom = potentialsQij[:N, 2].reshape((-1, 1))
			PotentialsXY  = potentialsQij[:N, :2]
			np.multiply(PotentialsCom, Y, out=neg_f_buf[:N])
			neg_f_buf[:N] -= PotentialsXY
	_timed('m2p_time_s', m2p_start)

	# 11. Sampling: 恢复 neg_f 的形状
	sampling_restore_start = perf_counter()
	grid_context = None
	if fused_update_enabled:
		_timed('sampling_restore_time_s', sampling_restore_start)
		if p2m_diagnostics is not None:
			p2m_diagnostics['fused_m2p_update'] = True
		_finish_timing()
		return None
	if return_grid_context:
		if probabilities is None:
			context_box_idx = box_idx
		else:
			# Sampling changes the active FFT points, but local mechanisms need a
			# query index for every embedding point. Assign all points to the same
			# unpadded grid bounds used by this ibFFT call.
			context_box_idx = np.empty((N_original, 2), dtype=np.int32)
			Box_idx(
				context_box_idx,
				Y_original,
				box_width,
				min_coord,
				n_boxes_per_dim,
				N_original,
			)
		grid_context = _build_grid_context(
			Y_original,
			context_box_idx,
			min_coord,
			max_coord,
			box_width,
			n_boxes_per_dim,
		)

	if probabilities is not None:
		# Scale active-point forces by N_original / active_N
		scale = float(N_original) / float(N)
		# Allocate output restore buffer in workspace (stable size N_original)
		neg_f_restore = _get_or_alloc('neg_f_restore', (N_original, 2))
		neg_f_restore.fill(0.0)
		neg_f_restore[active_points_idx] = neg_f_buf[:N] * scale
		_timed('sampling_restore_time_s', sampling_restore_start)
		_finish_timing()
		if return_grid_context:
			return neg_f_restore, grid_context
		return neg_f_restore

	# Exact mode: return a view of the first N rows (no copy needed)
	_timed('sampling_restore_time_s', sampling_restore_start)
	_finish_timing()
	if return_grid_context:
		return neg_f_buf[:N_ws], grid_context
	return neg_f_buf[:N_ws]


# ---------------------------------------------------------------------------
# Change A (Round-5): AOT Warmup — eliminate cold-start JIT penalty
#
# Background: all @numba.njit functions above carry cache=True, which saves
# compiled native code to __pycache__/*.nbc files. However, Numba still
# needs to (a) deserialise the cached bytecode, (b) specialise to the exact
# argument type signatures used at runtime, and (c) link the parallel
# thread-pool for prange kernels.  For the sampling code path (which uses
# UMAP_AttrForce_sampling + all ibFFT numba kernels) this deserialization
# and specialization accounts for ~10 s on a cold process.
#
# Fix: call every JIT kernel once with tiny dummy arrays during module
# import.  The first call triggers Numba to compile (or load from cache and
# specialize) for float32/int32 inputs — the only types used at runtime.
# Subsequent calls in the same process (and any process after a warm cache)
# pay only the fast lookup cost.
#
# Implementation note: we use the minimum viable data (N=4, nip=1) so the
# warmup itself completes in < 0.1 s even on a cold cache.  The prange
# kernels still spin up all threads once here, which is the main source of
# the sampling-mode cold-start penalty.
# ---------------------------------------------------------------------------

def _warmup_jit_kernels() -> None:
	"""Pre-trigger JIT compilation for all ibFFT numba kernels.

	Called once at module import so that the first user call to
	ibFFT_repulsive_sampling pays only the fast cache-lookup cost instead
	of the ~10 s JIT compile + thread-pool initialization penalty.
	"""
	import warnings

	N = 4
	nip = 1
	n_bpd = 50
	n_ipl = nip * n_bpd  # interpolation points per dimension

	# Dummy embedding: 4 points in 2D, well-separated
	Y_dummy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
						dtype=np.float32)

	min_c = np.float32(Y_dummy.min())
	max_c = np.float32(Y_dummy.max())
	box_w = np.float32((max_c - min_c) / n_bpd)
	h = np.float32(box_w / nip)

	# Dummy arrays sized for the tiny N=4 case
	circulant = np.zeros((2 * n_ipl, 2 * n_ipl), dtype=np.float32)
	box_idx_d = np.zeros((N, 2), dtype=np.int32)
	y_in_box_d = np.zeros((N, 2), dtype=np.float32)
	interp_d = np.ones((N, nip, 2), dtype=np.float32)
	w_coeff_d = np.zeros((n_ipl, n_ipl, 3), dtype=np.float32)
	potq_d = np.zeros((N, 3), dtype=np.float32)
	chargesq_d = np.ones((N, 3), dtype=np.float32)
	chargesq_d[:, :2] = Y_dummy
	y_tilde = np.array([h / 2], dtype=np.float32)
	denom = np.ones(nip, dtype=np.float32)

	output_d = np.zeros((3, 2 * n_ipl, 2 * n_ipl), dtype=np.float32)
	neg_f_d = np.zeros((N, 2), dtype=np.float32)
	attr_f_d = np.zeros((N, 2), dtype=np.float32)
	neg_effects_d = np.ones(N, dtype=np.float32)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		try:
			# Trigger JIT compilation for all ibFFT numba kernels
			bounds_min = np.empty(1, dtype=np.float32)
			bounds_max = np.empty(1, dtype=np.float32)
			bounds_bad = np.empty(1, dtype=np.int64)
			_bounds_finite_partials(
				Y_dummy, Y_dummy.size, bounds_min, bounds_max, bounds_bad
			)
			Circulant_kernel_tilde(circulant, box_w, nip, n_ipl, 2*n_ipl, n_bpd, N, 1.0, 1.0)
			Box_idx(box_idx_d, Y_dummy, box_w, min_c, n_bpd, N)
			Y_in_box(y_in_box_d, Y_dummy, box_idx_d, box_w, min_c, n_bpd, N)
			Interpolate(y_in_box_d, y_tilde, denom, interp_d, nip, N)
			Compute_w_coeff(w_coeff_d, box_idx_d, chargesq_d, interp_d, nip, n_bpd, 3, N)
			Compute_mat_w_p1(output_d, box_idx_d, Y_dummy, N)
			PotentialsQij(potq_d, box_idx_d, interp_d, output_d, nip, n_bpd, 3, N)
			NegF_p1(neg_f_d, box_idx_d, Y_dummy, output_d, n_bpd, N)
			NegF_p1_fused_update(
				Y_dummy.copy(), attr_f_d, box_idx_d, Y_dummy, output_d,
				n_bpd, N, np.float32(1.0), neg_effects_d, True, np.float32(1.0),
			)
			PotentialsQij_fused_update(
				Y_dummy.copy(), attr_f_d, box_idx_d, Y_dummy, interp_d, output_d,
				nip, n_bpd, N, np.float32(1.0), neg_effects_d, True, np.float32(1.0),
			)
		except Exception:
			# Warmup failures are non-fatal; the real call will handle errors.
			pass


# Run warmup at module import time.  Wrapped in a try/except so that import
# errors (e.g. missing pyfftw in a test environment) don't prevent importing
# the module.
try:
	_warmup_jit_kernels()
except Exception:
	pass
