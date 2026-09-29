# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, the ibUMAP authors.
#
# Parts of this file are adapted from t-FDP (https://github.com/Ideas-Laboratory/t-fdp)
# with the permission of its author, and follow FIt-SNE's nbodyfft (MIT).
# See THIRD_PARTY_NOTICES.md.

import os
from time import time

import numpy as np
import cupy

from ..p2m import normalize_workspace_policy, resolve_p2m_mode
from ..fft_grid import (
	FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE,
	quantize_fft_box_width,
)
from ..._fft_kernel_cache import (
	DEFAULT_CUDA_FFT_KERNEL_CACHE_MAX_ENTRIES,
	DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES,
	FFTKernelLRUCache,
)

_DEFAULT_THREADS_PER_BLOCK = 256
_MAX_GRID_BLOCKS = 65535
grid_dim = (256,)
block_dim = (_DEFAULT_THREADS_PER_BLOCK,)
_FFT_KERNEL_CACHE_MAXSIZE = DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES
_fft_kernel_cache = FFTKernelLRUCache(scope="module")


def _clear_fft_kernel_cache():
	_fft_kernel_cache.clear()


def _get_cached_fft_kernel(cache, key):
	return cache.lookup(key).value


def _store_fft_kernel(cache, key, value):
	cache.store(key, value)


def point_launch(n_items, *, threads_per_block=_DEFAULT_THREADS_PER_BLOCK):
	n_items = int(n_items)
	threads_per_block = int(threads_per_block)
	blocks = max(1, min(_MAX_GRID_BLOCKS, (n_items + threads_per_block - 1) // threads_per_block))
	return (blocks,), (threads_per_block,)


def stable_segment_layout(cell_ids):
	"""Return stable point-order segments for one integer key per point."""
	n_points = int(cell_ids.shape[0])
	if n_points == 0:
		empty = cupy.empty(0, dtype=cupy.int64)
		return empty, empty.copy(), empty.copy(), empty.copy()

	# Stable single-key sorting preserves the original point order within each
	# cell, exactly matching lexsort((point_ids, cell_ids)) without materializing
	# point_ids or a two-row key array.
	order = cupy.argsort(cell_ids, kind='stable').astype(cupy.int64, copy=False)
	sorted_cells = cell_ids[order]
	boundary = cupy.empty(n_points, dtype=cupy.bool_)
	boundary[0] = True
	cupy.not_equal(sorted_cells[1:], sorted_cells[:-1], out=boundary[1:])
	starts = cupy.flatnonzero(boundary).astype(cupy.int64, copy=False)
	ends = cupy.empty_like(starts)
	ends[:-1] = starts[1:]
	ends[-1] = n_points
	segment_cells = sorted_cells[starts].astype(cupy.int64, copy=False)
	return order, segment_cells, starts, ends


_ALLOWED_N_BOXES_PER_DIM = np.array(
	[
		50, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243,
		256, 288, 324, 384, 432, 576, 648, 768, 864, 972, 1152, 1296,
		1536, 1728, 1944, 2304, 2592, 3072, 3456, 3888, 4608, 5184,
		6144, 6912,
	],
	dtype=np.int32,
)

cuda_raw_text = open(os.path.join(os.path.dirname(__file__), 'RawCudafloat.cu')).read()

# ibFFT utils
BoundsFinitePartials_cu			= cupy.RawKernel(cuda_raw_text, 'BoundsFinitePartials_cu')
BoundsFiniteFinalize_cu			= cupy.RawKernel(cuda_raw_text, 'BoundsFiniteFinalize_cu')
Circulant_kernel_tilde_cu		= cupy.RawKernel(cuda_raw_text, 'Circulant_kernel_tilde_cu')
Circulant_kernel_tilde_UMAP_cu	= cupy.RawKernel(cuda_raw_text, 'Circulant_kernel_tilde_UMAP_cu')
Box_idx_cu						= cupy.RawKernel(cuda_raw_text, 'Box_idx_cu')
Y_in_box_cu						= cupy.RawKernel(cuda_raw_text, 'Y_in_box_cu')
Denominator_cu					= cupy.RawKernel(cuda_raw_text, 'Denominator_cu')
Interpolate_cu					= cupy.RawKernel(cuda_raw_text, 'Interpolate_cu')
Compute_w_coeff_cu				= cupy.RawKernel(cuda_raw_text, 'Compute_w_coeff_cu')
Compute_w_coeff_p1_cu			= cupy.RawKernel(cuda_raw_text, 'Compute_w_coeff_p1_cu')
Compute_mat_w_p1_cu				= cupy.RawKernel(cuda_raw_text, 'Compute_mat_w_p1_cu')
Compute_mat_w_p1_block_atomic_cu = cupy.RawKernel(cuda_raw_text, 'Compute_mat_w_p1_block_atomic_cu')
Compute_mat_w_p1_segmented_cu = cupy.RawKernel(cuda_raw_text, 'Compute_mat_w_p1_segmented_cu')
Compute_mat_w_segmented_cu = cupy.RawKernel(cuda_raw_text, 'Compute_mat_w_segmented_cu')
PotentialsQij_cu				= cupy.RawKernel(cuda_raw_text, 'PotentialsQij_cu')
PotentialsQij_p1_cu				= cupy.RawKernel(cuda_raw_text, 'PotentialsQij_p1_cu')
NegF_p1_cu						= cupy.RawKernel(cuda_raw_text, 'NegF_p1_cu')
NegF_p1_fused_update_cu = cupy.RawKernel(cuda_raw_text, 'NegF_p1_fused_update_cu')
PotentialsQij_fused_update_cu = cupy.RawKernel(
	cuda_raw_text, 'PotentialsQij_fused_update_cu'
)


""" ibFFT_repulsive_sampling_GPU """

def ibFFT_repulsive_sampling_GPU(
	Y, 
	n_interpolation_points, 
	intervals_per_integer, 
	min_num_intervals, 
	gamma, 
	paraFactor, 
	probabilities = None, 	# Sampling probabilities
	n_boxes_per_dim = 1.0,	# 为 int 时指定，为 float 时，根据 N 和 Y 计算
	kernel_method = 'tFDP',	# t-FDP, UMAP_kernel, UMAP_gauss
	umap_a = 1.0,
	umap_b = 1.0,
	umap_gamma = 1.0,
	umap_epsilon = 0.001,
	gauss_sigma = 1.0,
	ibfft_kernel_clip = 1.0,
	random_state = None,
	deterministic = False,
	p2m_mode = 'auto',
	workspace = None,
	workspace_policy = 'auto',
	workspace_limit_bytes = None,
	fft_kernel_cache_policy = 'auto',
	fft_kernel_cache_limit_bytes = None,
	fft_kernel_cache_max_entries = None,
	p2m_diagnostics = None,
	p2m_event_pairs = None,
	timing_event_pairs = None,
	fused_update = None,
):
	timing_cursor = None
	if timing_event_pairs is not None:
		timing_cursor = cupy.cuda.Event()
		timing_cursor.record()

	def _timing_mark(name):
		nonlocal timing_cursor
		if timing_event_pairs is None:
			return None
		end_event = cupy.cuda.Event()
		end_event.record()
		pair = (timing_cursor, end_event)
		timing_event_pairs.setdefault(name, []).append(pair)
		timing_cursor = end_event
		return pair

	N_original = Y.shape[0]
	Y_original = Y
	N = N_original
	fused_update_enabled = fused_update is not None
	if fused_update_enabled:
		if probabilities is not None:
			raise ValueError("fused CUDA ibFFT update requires exact repulsion")
		fused_embedding = fused_update['embedding']
		fused_attr_force = fused_update['attr_force']
		fused_alpha = cupy.float32(fused_update['alpha'])
		fused_neg_effects = fused_update['neg_effects']
		fused_clip_norm = fused_update.get('clip_norm')
		fused_clip_enabled = fused_clip_norm is not None
		fused_clip_norm = cupy.float32(
			0.0 if fused_clip_norm is None else fused_clip_norm
		)
		if fused_embedding is not Y_original:
			raise ValueError("fused CUDA ibFFT update must target the input embedding")
	workspace_policy = normalize_workspace_policy(workspace_policy)
	if int(n_interpolation_points) != 1 and p2m_mode == 'block_atomic':
		raise NotImplementedError(
			"block_atomic CUDA P2M currently requires "
			"n_interpolation_points=1"
		)
	_ws = getattr(workspace, "buffers", workspace)
	if _ws is None:
		_ws = {}
	fft_kernel_cache = getattr(workspace, "fft_kernel_cache", _fft_kernel_cache)
	resolved_fft_kernel_cache_max_entries = fft_kernel_cache_max_entries
	if (
		fft_kernel_cache_policy == 'auto'
		and resolved_fft_kernel_cache_max_entries is None
		and getattr(fft_kernel_cache, "scope", None) == "workspace"
	):
		resolved_fft_kernel_cache_max_entries = (
			DEFAULT_CUDA_FFT_KERNEL_CACHE_MAX_ENTRIES
		)
	resolved_cache_policy = fft_kernel_cache.configure(
		policy=fft_kernel_cache_policy,
		limit_bytes=fft_kernel_cache_limit_bytes,
		max_entries=resolved_fft_kernel_cache_max_entries,
	)

	def _get_or_alloc(key, shape, dtype=cupy.float32, active_capacity=False):
		dtype = cupy.dtype(dtype)
		shape = tuple(int(dim) for dim in shape)
		arr = _ws.get(key)
		if active_capacity and workspace_policy == 'auto':
			compatible = (
				arr is not None
				and arr.dtype == dtype
				and tuple(arr.shape[1:]) == shape[1:]
			)
			if compatible and arr.shape[0] >= shape[0]:
				return arr
			capacity = shape[0]
			if compatible:
				capacity = max(capacity, int(arr.shape[0]))
			arr = cupy.empty((capacity, *shape[1:]), dtype=dtype)
			_ws[key] = arr
		elif arr is None or tuple(arr.shape) != shape or arr.dtype != dtype:
			arr = cupy.empty(shape, dtype=dtype)
			_ws[key] = arr
		return arr

	def _bounds_finite_reduction(values, active_mask=None):
		reduction_grid_dim, reduction_block_dim = point_launch(values.shape[0])
		n_partials = int(reduction_grid_dim[0])
		partials = _get_or_alloc(
			'ibfft.bounds_finite_partials',
			(n_partials, 3),
			dtype=cupy.float32,
		)
		result = _get_or_alloc(
			'ibfft.bounds_finite_result',
			(3,),
			dtype=cupy.float32,
		)
		shared_mem = int(reduction_block_dim[0]) * 3 * cupy.dtype(cupy.float32).itemsize
		mask_arg = active_mask if active_mask is not None else result
		BoundsFinitePartials_cu(
			reduction_grid_dim,
			reduction_block_dim,
			(
				values,
				mask_arg,
				partials,
				int(values.shape[0]),
				int(active_mask is not None),
			),
			shared_mem=shared_mem,
		)
		BoundsFiniteFinalize_cu(
			(1,),
			reduction_block_dim,
			(partials, result, n_partials),
			shared_mem=shared_mem,
		)
		# Single host transfer for min, max, and non-finite count.
		min_value, max_value, nonfinite_value = cupy.asnumpy(result)
		if p2m_diagnostics is not None:
			p2m_diagnostics['bounds_finite_fused_reduction'] = True
		return float(min_value), float(max_value), int(round(float(nonfinite_value)))
	
	# 0. Sampling: 以 degrees 为概率，随机决定样本是否参与计算：
	if probabilities is not None:
		if random_state is None:
			if_apply_point = cupy.random.rand(N) <= probabilities
		else:
			if_apply_point = random_state.random_sample(N) <= probabilities
		min_coord_value, max_coord_value, nonfinite_count = (
			_bounds_finite_reduction(Y_original, if_apply_point)
		)
		if nonfinite_count:
			raise FloatingPointError(
				"ibFFT input embedding contains "
				f"{nonfinite_count} non-finite value(s)"
			)
		active_points_idx = cupy.where(if_apply_point)[0]
		# replace Y
		original_Y = Y
		Y = Y[active_points_idx]
		N = Y.shape[0]
	else:
		min_coord_value, max_coord_value, nonfinite_count = (
			_bounds_finite_reduction(Y_original)
		)
		if nonfinite_count:
			raise FloatingPointError(
				"ibFFT input embedding contains "
				f"{nonfinite_count} non-finite value(s)"
			)
		original_Y = Y_original
	if not np.isfinite(min_coord_value) or not np.isfinite(max_coord_value):
		raise ValueError("ibFFT active embedding subset is empty")

	# 1.初始化变量和参数：
	# 计算最大和最小坐标以及其他一些初始参数，例如：
	# n_boxes_per_dim: 每个维度的盒子数量
	# Raw CUDA kernels below take min_coord as a float* device scalar.
	# Keep the Python host values for grid sizing, but pass a CuPy 0-d array to
	# Box_idx_cu/Y_in_box_cu to match the existing kernel ABI.
	min_coord = cupy.asarray(min_coord_value, dtype=cupy.float32)
	
	n_boxes_scale = float(n_boxes_per_dim)
	boxes_1 = np.sqrt(16 * N)
	boxes_2 = np.sqrt(4 * N / np.log(N))
	grid_span = max_coord_value - min_coord_value
	boxes_3 = grid_span / intervals_per_integer
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
	N_ws = N_original if workspace_policy == 'performance' else N
	use_p1_fast_path = int(n_interpolation_points) == 1

	def _retain_p2m_vector(key, values):
		"""Copy an active segmented-P2M vector into reusable workspace storage."""
		if workspace_policy == 'minimal':
			return values
		active_size = int(values.shape[0])
		# Every segmented layout vector contains at most one entry per active
		# point.  Allocate all retained vectors at the point-buffer capacity so
		# changes in the number of occupied cells do not replace their workspace
		# objects between optimizer/update calls.
		retained = _get_or_alloc(
			key,
			(N_ws,),
			dtype=values.dtype,
			active_capacity=True,
		)
		retained[:active_size] = values
		return retained[:active_size]

	# 计算箱子的宽度、FFT系数的数量，以及一些与插值点和核相关的参数
	raw_box_width, quantized_box_width, grid_quantization_level = (
		quantize_fft_box_width(grid_span, n_boxes_per_dim, np.float32)
	)
	# Raw CUDA kernels take box_width by pointer, so retain a device scalar.
	box_width = cupy.asarray(quantized_box_width, dtype=cupy.float32)		# 量化后的盒子宽度
	n_boxes = n_boxes_per_dim * n_boxes_per_dim								# 盒子总数
	n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim	# 插值点总数
	n_fft_coeffs = 2 * n_interpolation_points_1d							# FFT 系数总数

	# whsquare = box_width / n_interpolation_points							# 插值点间隔
	# whsquare *= whsquare													# 插值点间隔的平方
	# y_in_box uses physical embedding coordinates, so interpolation nodes must
	# use the same scale. Omitting box_width is harmless only when p == 1.
	h = box_width / n_interpolation_points									# 插值点间隔
	if not use_p1_fast_path:
		y_tilde_spacings = cupy.array(
			cupy.arange(n_interpolation_points) * h + h/2, dtype=cupy.float32)	# 每个盒子内的插值点坐标（一维）
	_cache_key = (
		int(cupy.cuda.runtime.getDevice()),
		int(grid_quantization_level),
		FFT_GRID_QUANTIZATION_STEPS_PER_OCTAVE,
		int(n_boxes_per_dim),
		int(n_interpolation_points),
		str(kernel_method),
		float(umap_a), float(umap_b), float(umap_gamma), float(umap_epsilon),
		float(ibfft_kernel_clip),
		float(gamma), float(paraFactor), float(gauss_sigma),
	)
	cache_lookup = fft_kernel_cache.lookup(_cache_key)
	fft_kernel_tilde = cache_lookup.value
	kernel_cache_hit = cache_lookup.hit
	if p2m_diagnostics is not None:
		cache_snapshot = fft_kernel_cache.snapshot()
		p2m_diagnostics.update({
			'grid_raw_box_width': float(raw_box_width),
			'grid_box_width': float(quantized_box_width),
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
	_timing_mark('setup_time_s')


	# 2.计算矩阵核 half_kernel, shape: (n_interpolation_points_1d, n_interpolation_points_1d)
	# 3.计算 Circulant Kernel：
	# 使用 half_kernel 计算 circulant_kernel_tilde ，即循环卷积核。
	# circulant_kernel_tilde 长宽都是 half_kernel 的两倍，其右下角是 half_kernel，其余部分是 half_kernel 的镜像。
	# circulant_kernel_tilde 是一个循环矩阵，其中half_kernel被复制和翻转以填充整个矩阵。
	# 这样做的目的是为了确保卷积的循环性质，这是利用 FFT 进行卷积计算的关键。
	if fft_kernel_tilde is None:
		circulant_kernel_tilde = cupy.zeros(
			(n_fft_coeffs, n_fft_coeffs), dtype=cupy.float32)
		if kernel_method == 'tFDP':
			Circulant_kernel_tilde_cu(*point_launch(n_interpolation_points_1d), (
				circulant_kernel_tilde, box_width, n_interpolation_points, n_interpolation_points_1d,
				n_fft_coeffs, n_boxes_per_dim, N, cupy.float32(paraFactor), cupy.float32(gamma)
			))
		elif kernel_method == 'UMAP_kernel':
			# RawKernel does not coerce Python floats to the kernel's ``float`` ABI.
			# Normalize every scalar explicitly so direct callers and the optimizer
			# produce the same kernel arguments.
			Circulant_kernel_tilde_UMAP_cu(*point_launch(n_interpolation_points_1d), (
				circulant_kernel_tilde, box_width, n_interpolation_points, n_interpolation_points_1d,
				n_fft_coeffs, cupy.float32(umap_a), cupy.float32(umap_b),
				cupy.float32(umap_gamma), cupy.float32(umap_epsilon),
				cupy.float32(ibfft_kernel_clip),
			))
	_timing_mark('kernel_build_time_s')

	# rfft2: 二维实数快速傅里叶变换
	# shape: (n_fft_coeffs, n_fft_coeffs//2 +1)
	# 对于垂直方向（第一个维度），频率分量的数量保持不变
	# 对于水平方向（第二个维度），由于我们只计算了正频率部分的结果，所以维度大小被减半并加1
	# rfft2 是为实数输入优化的，而对于实数信号，其傅里叶变换是对称的，
	# 因此我们只需要存储一半的结果（正频率部分）。这使得存储和计算更加高效。
	# fft_kernel_tilde = rfft2(circulant_kernel_tilde)
	# fft_kernel_tilde = rfft2(circulant_kernel_tilde, threads=4)
	if fft_kernel_tilde is None:
		fft_kernel_tilde = cupy.fft.rfft2(circulant_kernel_tilde)
		cache_store = fft_kernel_cache.store(_cache_key, fft_kernel_tilde)
		if p2m_diagnostics is not None:
			cache_snapshot = fft_kernel_cache.snapshot()
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
	_timing_mark('kernel_fft_time_s')


	# 4.匹配点和盒子：box_idx 和 y_in_box
	# 为每个点 Y[i] 在二维空间中分配一个盒子索引
	# box_idx[i][0] 和 box_idx[i][1] 分别存储点 Y[i] 在x轴和y轴上的盒子索引。
	box_idx = _get_or_alloc(
		'ibfft.box_idx', (N_ws, 2), dtype=cupy.int32, active_capacity=True
	)
	point_grid_dim, point_block_dim = point_launch(N)
	Box_idx_cu(point_grid_dim, point_block_dim, (box_idx, Y, box_width, min_coord, n_boxes_per_dim, N))

	if not use_p1_fast_path:
		# 计算每个点 Y[i] 相对于其所在盒子左下角的坐标
		# y_in_box[i][0] 和 y_in_box[i][1] 分别存储了点 Y[i] 相对于其所在盒子左下角的x和y坐标。
		y_in_box = _get_or_alloc(
			'ibfft.y_in_box', (N_ws, 2), active_capacity=True
		)
		Y_in_box_cu(point_grid_dim, point_block_dim, (y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N))


		# 5.计算插值权重的分母 denominator ，用于计算插值值时的正规化
		# denominator_sub = (y_tilde_spacings.reshape(-1, 1) - y_tilde_spacings)
		# np.fill_diagonal(denominator_sub, 1)
		# denominator = denominator_sub.prod(axis=0)	# 每一列的乘积，shape: (n_interpolation_points,)
		denominator = _get_or_alloc(
			f'ibfft.denominator.p{int(n_interpolation_points)}',
			(n_interpolation_points,),
		)
		Denominator_cu((1,), (n_interpolation_points,),
						(denominator, y_tilde_spacings, n_interpolation_points))

		# 6.为每个点 Y[i] 计算其插值权重 interpolate_values
		# 每个点 Y[i] 对应 n_interpolation_points 个插值值，每个插值值都是一个二维向量。
		# 数组 interpolate_values 包含了基于拉格朗日插值计算的插值权重。
		# 每个点 Y[i] 都与 y_tilde_spacings 中的每个插值点相关联的插值权重。
		# 	interpolate_values[i][j][0] 表示点 Y[i] 的 x 坐标相对于第 j 个插值点的插值权重。
		# 	interpolate_values[i][j][1] 表示点 Y[i] 的 y 坐标相对于第 j 个插值点的插值权重。
		interpolate_values = _get_or_alloc(				# CPU 算法此处原本是 np.ndarray，后改为 np.ones
			f'ibfft.interpolate_values.p{int(n_interpolation_points)}',
			(N_ws, n_interpolation_points, 2),
			active_capacity=True,
		)
		Interpolate_cu(point_grid_dim, point_block_dim, (y_in_box, y_tilde_spacings,
						denominator, interpolate_values, n_interpolation_points, N))
	_timing_mark('point_setup_time_s')


	# 7.计算权重系数 w_coefficients
	# w_coefficients 存储了基于每个点的位置和其对应的电荷或权重（由 chargesQij 给出）的权重系数。
	# 简而言之，计算了每个盒子内的每个插值点与所有其他插值点之间的相互作用
	w_coeff_len_per_dim = int(n_boxes_per_dim * n_interpolation_points)
	grid_key = f'p{int(n_interpolation_points)}.b{int(n_boxes_per_dim)}'

	# 8.使用 FFT 计算卷积
	# 将 w_coefficients 转置存储到 mat_w 矩阵中
	# 使用 rfft2() 计算 mat_w 的二维实数快速傅里叶变换，将其转换为频率域表示
	# 在频率域中，将 mat_w 与 fft_kernel_tilde 相乘（卷积运算），得到频率域中的结果
	# 逆FFT，得到 mat_w 和 fft_kernel_tilde 卷积的结果
	n_fft_coeffs_int = int(n_fft_coeffs)
	mat_w_key = f'ibfft.mat_w.{grid_key}'
	if workspace_policy != 'performance':
		for stale_key in tuple(_ws):
			if stale_key.startswith('ibfft.mat_w.') and stale_key != mat_w_key:
				del _ws[stale_key]
	mat_w = _get_or_alloc(
		mat_w_key,
		(n_terms, n_fft_coeffs_int, n_fft_coeffs_int),
	)
	mat_w.fill(0.0)
	_timing_mark('mesh_clear_time_s')
	p2m_start_event = None
	if timing_event_pairs is None and p2m_event_pairs is not None:
		p2m_start_event = cupy.cuda.Event()
		p2m_start_event.record()
	if use_p1_fast_path:
		resolution = resolve_p2m_mode(
			p2m_mode,
			deterministic=bool(deterministic),
			device='cuda',
			n_points=N,
			workspace_limit_bytes=workspace_limit_bytes,
		)
	else:
		resolution = resolve_p2m_mode(
			p2m_mode,
			deterministic=bool(deterministic),
			device='cuda',
			n_points=N,
			workspace_limit_bytes=workspace_limit_bytes,
		)

	if use_p1_fast_path:
		if resolution.resolved == 'atomic':
			Compute_mat_w_p1_cu(point_grid_dim, point_block_dim, (
				mat_w, box_idx, Y, n_fft_coeffs_int, n_terms, N
			))
		elif resolution.resolved == 'block_atomic':
			shared_mem = int(point_block_dim[0]) * (4 + 4 + 4)
			Compute_mat_w_p1_block_atomic_cu(
				point_grid_dim,
				point_block_dim,
				(mat_w, box_idx, Y, n_fft_coeffs_int, n_terms, N),
				shared_mem=shared_mem,
			)
		elif resolution.resolved == 'serial':
			Compute_mat_w_p1_cu((1,), (1,), (
				mat_w, box_idx, Y, n_fft_coeffs_int, n_terms, N
			))
		elif resolution.resolved == 'segmented':
			cell_ids = (
				box_idx[:N, 0].astype(cupy.int64, copy=False) * n_fft_coeffs_int
				+ box_idx[:N, 1]
			)
			cell_ids = _retain_p2m_vector('ibfft.p2m.cell_ids', cell_ids)
			order, segment_cells, starts, ends = stable_segment_layout(cell_ids)
			order = _retain_p2m_vector('ibfft.p2m.order', order)
			segment_cells = _retain_p2m_vector(
				'ibfft.p2m.segment_cells', segment_cells
			)
			starts = _retain_p2m_vector('ibfft.p2m.segment_starts', starts)
			ends = _retain_p2m_vector('ibfft.p2m.segment_ends', ends)
			segment_grid, segment_block = point_launch(int(segment_cells.shape[0]))
			Compute_mat_w_p1_segmented_cu(segment_grid, segment_block, (
				mat_w,
				Y,
				order,
				segment_cells,
				starts,
				ends,
				int(segment_cells.shape[0]),
				n_fft_coeffs_int,
				n_terms,
			))
		else:
			raise RuntimeError(f"unresolved CUDA P2M mode: {resolution.resolved}")
	else:
		if resolution.resolved == 'segmented':
			# Interpolation nodes belonging to different boxes never overlap. Sort
			# points once by box, then assign every (box, j, k) node to exactly one
			# CUDA thread. Temporary storage remains O(N), independent of p.
			box_ids = (
				box_idx[:N, 0].astype(cupy.int64, copy=False) * int(n_boxes_per_dim)
				+ box_idx[:N, 1]
			)
			box_ids = _retain_p2m_vector('ibfft.p2m.box_ids', box_ids)
			order, segment_boxes, starts, ends = stable_segment_layout(box_ids)
			order = _retain_p2m_vector('ibfft.p2m.order', order)
			segment_boxes = _retain_p2m_vector(
				'ibfft.p2m.segment_boxes', segment_boxes
			)
			starts = _retain_p2m_vector('ibfft.p2m.segment_starts', starts)
			ends = _retain_p2m_vector('ibfft.p2m.segment_ends', ends)
			n_segment_outputs = (
				int(segment_boxes.shape[0])
				* int(n_interpolation_points)
				* int(n_interpolation_points)
			)
			segment_grid, segment_block = point_launch(n_segment_outputs)
			Compute_mat_w_segmented_cu(segment_grid, segment_block, (
				mat_w,
				Y,
				interpolate_values,
				order,
				segment_boxes,
				starts,
				ends,
				int(segment_boxes.shape[0]),
				int(n_interpolation_points),
				int(n_boxes_per_dim),
				n_fft_coeffs_int,
				n_terms,
			))
		elif resolution.resolved in ('atomic', 'serial'):
			# ChargesQij: N*3 数组，包含点坐标和常量权重 1。
			ChargesQij = _get_or_alloc(
				'ibfft.ChargesQij', (N_ws, squared_n_terms), active_capacity=True
			)
			ChargesQij[:N].fill(1.0)
			ChargesQij[:N, :2] = Y
			w_coefficients = _get_or_alloc(
				f'ibfft.w_coefficients.{grid_key}',
				(w_coeff_len_per_dim, w_coeff_len_per_dim, squared_n_terms),
			)
			w_coefficients.fill(0.0)
			p2m_grid_dim = (1,) if resolution.resolved == 'serial' else point_grid_dim
			p2m_block_dim = (1,) if resolution.resolved == 'serial' else point_block_dim
			Compute_w_coeff_cu(p2m_grid_dim, p2m_block_dim, (
				w_coefficients,
				box_idx,
				ChargesQij,
				interpolate_values,
				n_interpolation_points,
				n_boxes_per_dim,
				n_terms,
				N,
			))
			for term in range(n_terms):
				mat_w[term, :w_coeff_len_per_dim, :w_coeff_len_per_dim] = (
					w_coefficients[:, :, term]
				)
		else:
			raise RuntimeError(f"unresolved CUDA P2M mode: {resolution.resolved}")
	p2m_pair = _timing_mark('p2m_time_s')
	if p2m_event_pairs is not None:
		if p2m_pair is not None:
			p2m_event_pairs.append(p2m_pair)
		else:
			p2m_end_event = cupy.cuda.Event()
			p2m_end_event.record()
			p2m_event_pairs.append((p2m_start_event, p2m_end_event))
	if p2m_diagnostics is not None:
		p2m_diagnostics['requested_mode'] = str(p2m_mode)
		p2m_diagnostics['resolved_mode'] = resolution.resolved
		p2m_diagnostics['reason'] = resolution.reason
		p2m_diagnostics['estimated_workspace_bytes'] = (
			resolution.estimated_workspace_bytes
		)
	if not mat_w.flags.c_contiguous:
		raise RuntimeError("CUDA ibFFT mat_w must be C-contiguous")
	use_workspace_fft = bool(
		deterministic
		and workspace_policy != 'minimal'
		and hasattr(workspace, 'get_fft_plan')
	)
	if use_workspace_fft:
		from cupyx.scipy.fft import get_fft_plan

		fft_shape = (n_fft_coeffs_int, n_fft_coeffs_int)
		fft_axes = (-2, -1)
		device_id = int(cupy.cuda.runtime.getDevice())
		fft_w_shape = (
			n_terms,
			n_fft_coeffs_int,
			n_fft_coeffs_int // 2 + 1,
		)
		fft_w = _get_or_alloc(
			'ibfft.fft.frequency', fft_w_shape, dtype=cupy.complex64
		)
		output = _get_or_alloc(
			'ibfft.fft.output',
			(n_terms, n_fft_coeffs_int, n_fft_coeffs_int),
			dtype=cupy.float32,
		)
		forward_key = (
			device_id,
			tuple(mat_w.shape),
			str(mat_w.dtype),
			fft_shape,
			fft_axes,
		)
		inverse_key = (
			device_id,
			tuple(fft_w.shape),
			str(fft_w.dtype),
			fft_shape,
			fft_axes,
		)
		forward_plan, forward_hit = workspace.get_fft_plan(
			'ibfft.fft.r2c',
			forward_key,
			lambda: get_fft_plan(
				mat_w,
				shape=fft_shape,
				axes=fft_axes,
				value_type='R2C',
			),
		)
		inverse_plan, inverse_hit = workspace.get_fft_plan(
			'ibfft.fft.c2r',
			inverse_key,
			lambda: get_fft_plan(
				fft_w,
				shape=fft_shape,
				axes=fft_axes,
				value_type='C2R',
			),
		)
		_timing_mark('fft_plan_build_time_s')
		forward_plan.fft(mat_w, fft_w, cupy.cuda.cufft.CUFFT_FORWARD)
		if p2m_diagnostics is not None:
			fft_snapshot = workspace.fft_snapshot()
			p2m_diagnostics.update({
				'fft_backend': 'workspace_planned_buffers',
				'fft_plan_cache_hit': bool(forward_hit and inverse_hit),
				'fft_forward_plan_hit': bool(forward_hit),
				'fft_inverse_plan_hit': bool(inverse_hit),
				'fft_plan_entries': int(fft_snapshot['plan_entries']),
				'fft_plan_hits': int(fft_snapshot['plan_hits']),
				'fft_plan_misses': int(fft_snapshot['plan_misses']),
				'fft_frequency_buffer_bytes': int(fft_w.nbytes),
				'fft_output_buffer_bytes': int(output.nbytes),
			})
	else:
		fft_w = cupy.fft.rfft2(mat_w)
		output = None
		if p2m_diagnostics is not None:
			p2m_diagnostics['fft_backend'] = 'cupy_wrapper'
	_timing_mark('fft_forward_time_s')
	fft_w *= fft_kernel_tilde
	_timing_mark('fft_multiply_time_s')
	if use_workspace_fft:
		inverse_plan.fft(fft_w, output, cupy.cuda.cufft.CUFFT_INVERSE)
		output *= cupy.float32(1.0 / (n_fft_coeffs_int * n_fft_coeffs_int))
	else:
		output = cupy.fft.irfft2(fft_w)
	_timing_mark('fft_inverse_time_s')


	# 9.计算最终每个点在 x 和 y 方向上的斥力。
	# p=1 直接从 FFT 输出 gather potential_common/potential_x/potential_y，
	# 跳过 potentialsQij 中间 buffer；p>1 保留通用插值 M2P 路径。
	if use_p1_fast_path:
		if fused_update_enabled:
			NegF_p1_fused_update_cu(point_grid_dim, point_block_dim, (
				fused_embedding,
				fused_attr_force,
				box_idx,
				Y,
				output,
				n_boxes_per_dim,
				N,
				fused_alpha,
				fused_neg_effects,
				int(fused_clip_enabled),
				fused_clip_norm,
			))
		else:
			neg_f = _get_or_alloc('ibfft.neg_f', (N_ws, 2), active_capacity=True)
			NegF_p1_cu(point_grid_dim, point_block_dim, (
				neg_f, box_idx, Y, output, n_boxes_per_dim, N
			))
	else:
		if fused_update_enabled:
			PotentialsQij_fused_update_cu(point_grid_dim, point_block_dim, (
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
				int(fused_clip_enabled),
				fused_clip_norm,
			))
		else:
			neg_f = _get_or_alloc('ibfft.neg_f', (N_ws, 2), active_capacity=True)
			potentialsQij = _get_or_alloc(
				'ibfft.potentialsQij', (N_ws, n_terms), active_capacity=True
			)
			potentialsQij[:N].fill(0.0)
			PotentialsQij_cu(point_grid_dim, point_block_dim, (potentialsQij, box_idx, interpolate_values,
						output, n_interpolation_points, n_boxes_per_dim, n_terms, N))
			PotentialsCom = potentialsQij[:, 2].reshape((-1, 1))	# 第三列，表示每个点的共同斥力潜势。
			PotentialsXY = potentialsQij[:, :2]						# 前两列，表示每个点的 x 和 y 斥力潜势。
			neg_f[:N] = PotentialsCom[:N] * Y - PotentialsXY[:N]
	_timing_mark('m2p_time_s')
	
	# 10. Sampling: 恢复 neg_f 的形状
	if fused_update_enabled:
		if p2m_diagnostics is not None:
			p2m_diagnostics['fused_m2p_update'] = True
		_timing_mark('sampling_restore_time_s')
		return None
	if probabilities is not None:
		# 放大斥力，排除采样影响
		# neg_f[:, 0] /= probabilities[active_points_idx]
		# neg_f[:, 1] /= probabilities[active_points_idx]
		active_N = int(len(active_points_idx))
		neg_f_active = neg_f[:N] * (len(original_Y) / active_N)

		# restore neg_f to original shape
		# neg_f_restore = np.zeros_like(original_Y)
		# neg_f_restore[active_points_idx] = neg_f
		neg_f_restore = _get_or_alloc(
			'ibfft.neg_f_restore', (N_original, 2)
		)
		neg_f_restore.fill(0.0)
		neg_f_restore[active_points_idx] = neg_f_active
		neg_f = neg_f_restore
	else:
		neg_f = neg_f[:N]
	_timing_mark('sampling_restore_time_s')
	
	return neg_f
