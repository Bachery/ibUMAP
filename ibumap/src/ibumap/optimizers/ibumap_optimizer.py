import csv
import ctypes
import numba
import platform
import numpy as np
from pathlib import Path
from time import perf_counter, time
from umap.utils import tau_rand_int
from umap.layouts import clip, rdist
from ..kernels.cpu.ibfft import ibFFT_repulsive_sampling
from ..fft_schedule import (
	fft_stage_index_for_epoch,
	format_resolved_fft_schedule,
	resolve_fft_schedule,
	validate_fft_schedule_execution,
)
cupy = None
grid_dim = block_dim = point_launch = None
if 'Linux' in platform.platform():
	try:
		import cupy
		from ..kernels.gpu.cupy_ibfft import grid_dim, block_dim, point_launch, cuda_raw_text
		from ..kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
		UMAP_AttrForce_cu			= cupy.RawKernel(cuda_raw_text, 'UMAP_AttrForce_cu')
		UMAP_AttrForce_sampling_cu	= cupy.RawKernel(cuda_raw_text, 'UMAP_AttrForce_sampling_cu')
		UMAP_AttrForce_warp_per_row_cu = cupy.RawKernel(cuda_raw_text, 'UMAP_AttrForce_warp_per_row_cu')
		UMAP_AttrForce_sampling_warp_per_row_cu = cupy.RawKernel(cuda_raw_text, 'UMAP_AttrForce_sampling_warp_per_row_cu')
		UMAP_ApplyForce_cu			= cupy.RawKernel(cuda_raw_text, 'UMAP_ApplyForce_cu')
		UMAP_ReplForcePostprocess_cu	= cupy.RawKernel(cuda_raw_text, 'UMAP_ReplForcePostprocess_cu')
	except Exception:
		cupy = None


_PROFILE_SECONDS_PER_TICK = 0.0
if platform.system() == "Darwin":
	_libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
	_mach_absolute_time = _libsystem.mach_absolute_time
	_mach_absolute_time.argtypes = []
	_mach_absolute_time.restype = ctypes.c_uint64

	class _MachTimebaseInfo(ctypes.Structure):
		_fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

	_timebase_info = _MachTimebaseInfo()
	_libsystem.mach_timebase_info(ctypes.byref(_timebase_info))
	_PROFILE_SECONDS_PER_TICK = (
		float(_timebase_info.numer) / float(_timebase_info.denom) * 1e-9
	)

	@numba.njit(inline="always")
	def _profile_ticks():
		return _mach_absolute_time()
else:
	@numba.njit(inline="always")
	def _profile_ticks():
		return np.uint64(0)


""" UMAP_original_vertex """

@numba.njit(parallel=True, cache=True)
def umap_original_optimization_single_epoch_vertex(
	embedding,
	edgesrc,
	edgetgt,
	n_vertices,
	n_components,
	rng_state,
	a,
	b,
	gamma,
	alpha,
	epoch_itr,
	epochs_per_sample,
	epoch_of_next_sample,
	epochs_per_negative_sample,
	epoch_of_next_negative_sample,
	epsilon=0.001,
	attr_gauss=False,
	repl_gauss=False,
	gauss_sigma=1.0,
):
	for i in numba.prange(n_vertices):
		for k in range(edgesrc[i], edgesrc[i+1]):
			if epoch_of_next_sample[k] <= epoch_itr:
				j = edgetgt[k]
				current = embedding[i]
				other = embedding[j]
				dist_squared = rdist(current, other)
				# 计算梯度系数
				if dist_squared > 0.0:
					if attr_gauss:
						w_ij = np.exp( - dist_squared / (2 * gauss_sigma ** 2) )
						grad_coeff = ( - 1.0 / (1.0 - w_ij) ) / (gauss_sigma ** 2)
					else:
						grad_coeff = - 2.0 * a * b * pow(dist_squared, b - 1.0)
						grad_coeff /= a * pow(dist_squared, b) + 1.0
				else:
					grad_coeff = 0.0
				# 计算梯度，更新 embedding
				for d in range(n_components):
					grad_d = clip(grad_coeff * (current[d] - other[d]))
					current[d] += grad_d * alpha
					other[d] -= grad_d * alpha
				# 更新采样次数
				epoch_of_next_sample[k] += epochs_per_sample[k]
				# 更新负采样次数
				n_neg_samples = int( (epoch_itr - epoch_of_next_negative_sample[k]) / epochs_per_negative_sample[k] )
				epoch_of_next_negative_sample[k] += n_neg_samples * epochs_per_negative_sample[k]
				# 负采样
				for neg in range(n_neg_samples):
					neg_j = tau_rand_int(rng_state) % n_vertices
					if neg_j == j: continue
					other = embedding[neg_j]
					dist_squared = rdist(current, other)
					# 计算梯度系数
					if dist_squared > 0.0:
						if repl_gauss:
							w_ij = np.exp( - dist_squared / (2 * gauss_sigma ** 2) )
							grad_coeff = (w_ij / (1.0 - w_ij + epsilon) ) / (gauss_sigma ** 2)
						else:
							grad_coeff = 2.0 * gamma * b
							grad_coeff /= (epsilon + dist_squared) * ( a * pow(dist_squared, b) + 1 )
					else:
						grad_coeff = 0.0
					# 计算梯度
					for d in range(n_components):
						if grad_coeff > 0.0:	grad_d = clip(grad_coeff * (current[d] - other[d]))
						else:					grad_d = 4.0
						current[d] += grad_d * alpha


def umap_original_optimization_vertex(
	embedding,
	edgesrc,
	edgetgt,
	n_epochs,
	n_vertices,
	n_components,
	rng_state,
	epochs_per_sample,
	epochs_per_negative_sample,
	drfft_params,
):
	t1 = time()
	initial_alpha = drfft_params['umap_initial_alpha']
	alpha = initial_alpha
	epoch_of_next_sample = epochs_per_sample.copy()
	epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
	time_costs = {'opt_prep_time': time() - t1}

	# Embedding optimization
	t1 = time()
	for epoch_itr in range(n_epochs):
		umap_original_optimization_single_epoch_vertex(
			embedding,
			edgesrc,
			edgetgt,
			n_vertices,
			n_components,
			rng_state,
			drfft_params['umap_a'],
			drfft_params['umap_b'],
			drfft_params['umap_gamma'],
			alpha,
			epoch_itr,
			epochs_per_sample,
			epoch_of_next_sample,
			epochs_per_negative_sample,
			epoch_of_next_negative_sample,
			drfft_params['umap_epsilon'],
			drfft_params['attr_gauss'],
			drfft_params['repl_gauss'],
			drfft_params['gauss_sigma'],
		)
		alpha = initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))
	time_costs['optimization_time'] = time() - t1

	time_costs['appl_time'] = 0.0
	return embedding, time_costs



""" True Loss """

""" Using Attraction and Repulsion Force """

@numba.njit(parallel=True, cache=True)
def UMAP_AttrForce(
	attr_force,
	embedding,
	edgesrc,
	edgetgt,
	pos_effects,
	n_vertices,
	n_components,
	a,
	b,
	alpha,
	whether_known_points,
	known_points_positions,
	known_points_reverse_index,
	soft_constraint=False,
	constraint_weight=0.1,
):
	for i in numba.prange(n_vertices):
		# cntSrc = edgesrc[i + 1] - edgesrc[i]
		i_known = whether_known_points[i]
		# pulls_i = np.zeros(n_components, dtype=np.float32)
		temp0 = 0.0
		temp1 = 0.0
		if soft_constraint and i_known:
			pidx = known_points_reverse_index[i]
			known_pos = known_points_positions[pidx]
			pull_0 = constraint_weight * (known_pos[0] - embedding[i][0])
			pull_1 = constraint_weight * (known_pos[1] - embedding[i][1])
			temp0 += pull_0
			temp1 += pull_1
		for k in range(edgesrc[i], edgesrc[i+1]):
			j = edgetgt[k]
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			# 计算梯度系数
			if dist_squared > 0.0:
				distance_power_b = pow(dist_squared, b)
				grad_coeff = -2.0 * a * b * (distance_power_b / dist_squared)
				grad_coeff /= a * distance_power_b + 1.0
				grad_coeff *= pos_effects[k]	### 边的权重，起到sampling的作用
			else:
				grad_coeff = 0.0
			# 计算梯度
			temp0 += alpha * clip(grad_coeff * mv0)
			temp1 += alpha * clip(grad_coeff * mv1)
		attr_force[i][0] = temp0
		attr_force[i][1] = temp1
			# attr_force[i][0] += alpha * clip(grad_coeff * mv0) + pull_0
			# attr_force[i][1] += alpha * clip(grad_coeff * mv1) + pull_1
			# CSR格式将无向图的边变成两条有向边进行存储，因此无需在此更新另一端的梯度
			# attr_force[j][0] -= alpha * clip(grad_coeff * mv0)
			# attr_force[j][1] -= alpha * clip(grad_coeff * mv1)


@numba.njit(parallel=True, cache=True)
def UMAP_AttrForce_sampling(
	attr_force,
	embedding,
	edgesrc,
	edgetgt,
	n_vertices,
	a,
	b,
	alpha,
	epoch_itr,
	epochs_per_sample, 
	epoch_of_next_sample, 
	whether_known_points,
	known_points_positions,
	known_points_reverse_index,
	soft_constraint=False,
	constraint_weight=0.1,
):
	for i in numba.prange(n_vertices):
		# cntSrc = edgesrc[i + 1] - edgesrc[i]
		i_known = whether_known_points[i]
		temp0 = 0.0
		temp1 = 0.0
		if soft_constraint and i_known:
			pidx = known_points_reverse_index[i]
			known_pos = known_points_positions[pidx]
			# Apply the soft pull once per point and epoch, independently of how
			# many positive edges happen to be sampled in this epoch.
			temp0 += constraint_weight * (known_pos[0] - embedding[i][0])
			temp1 += constraint_weight * (known_pos[1] - embedding[i][1])
		for k in range(edgesrc[i], edgesrc[i+1]):
			if epoch_of_next_sample[k] <= epoch_itr:
				epoch_of_next_sample[k] += epochs_per_sample[k]
				j = edgetgt[k]
				mv0 = embedding[i][0] - embedding[j][0]
				mv1 = embedding[i][1] - embedding[j][1]
				dist_squared = mv0 * mv0 + mv1 * mv1
				# 计算梯度系数
				if dist_squared > 0.0:
					distance_power_b = pow(dist_squared, b)
					grad_coeff = -2.0 * a * b * (distance_power_b / dist_squared)
					grad_coeff /= a * distance_power_b + 1.0
				else:
					grad_coeff = 0.0
				# 计算梯度
				temp0 += alpha * clip(grad_coeff * mv0)
				temp1 += alpha * clip(grad_coeff * mv1)
				# attr_force[j][0] -= alpha * clip(grad_coeff * mv0)
				# attr_force[j][1] -= alpha * clip(grad_coeff * mv1)
		attr_force[i][0] = temp0
		attr_force[i][1] = temp1


@numba.njit(cache=True)
def _count_sampling_calendar_events(edgesrc, epochs_per_sample, n_epochs):
	n_vertices = edgesrc.shape[0] - 1
	total_events = 0
	for i in range(n_vertices):
		for k in range(edgesrc[i], edgesrc[i + 1]):
			sample_interval = epochs_per_sample[k]
			if sample_interval <= 0.0:
				continue
			next_sample = sample_interval
			next_epoch = int(np.ceil(next_sample))
			while next_epoch < n_epochs:
				total_events += 1
				next_sample += sample_interval
				ceil_epoch = int(np.ceil(next_sample))
				if ceil_epoch <= next_epoch:
					next_epoch += 1
				else:
					next_epoch = ceil_epoch
	return total_events


@numba.njit(cache=True)
def _fill_sampling_calendar_events(
	edgesrc,
	epochs_per_sample,
	n_epochs,
	event_epochs,
	event_sources,
	event_edges,
):
	n_vertices = edgesrc.shape[0] - 1
	write = 0
	for i in range(n_vertices):
		for k in range(edgesrc[i], edgesrc[i + 1]):
			sample_interval = epochs_per_sample[k]
			if sample_interval <= 0.0:
				continue
			next_sample = sample_interval
			next_epoch = int(np.ceil(next_sample))
			while next_epoch < n_epochs:
				event_epochs[write] = next_epoch
				event_sources[write] = i
				event_edges[write] = k
				write += 1
				next_sample += sample_interval
				ceil_epoch = int(np.ceil(next_sample))
				if ceil_epoch <= next_epoch:
					next_epoch += 1
				else:
					next_epoch = ceil_epoch


@numba.njit(cache=True)
def _edge_sources_from_csr(edgesrc, edge_count):
	sources = np.empty(edge_count, dtype=np.int32)
	n_vertices = edgesrc.shape[0] - 1
	for i in range(n_vertices):
		for k in range(edgesrc[i], edgesrc[i + 1]):
			sources[k] = i
	return sources


@numba.njit(cache=True)
def _fill_periodic_bucket_edges(active_edges, edge_sources, bucket_ids, bucket_offsets):
	bucket_edges = np.empty(active_edges.shape[0], dtype=np.int32)
	bucket_edge_sources = np.empty(active_edges.shape[0], dtype=np.int32)
	write_offsets = bucket_offsets[:-1].copy()
	for pos in range(active_edges.shape[0]):
		edge = active_edges[pos]
		bucket = bucket_ids[pos]
		write = write_offsets[bucket]
		bucket_edges[write] = edge
		bucket_edge_sources[write] = edge_sources[edge]
		write_offsets[bucket] = write + 1
	return bucket_edges, bucket_edge_sources


@numba.njit(cache=True)
def _fill_periodic_bucket_due_epochs(bucket_intervals, n_epochs, bucket_due_epochs):
	total_events = 0
	for bucket in range(bucket_intervals.shape[0]):
		sample_interval = bucket_intervals[bucket]
		if sample_interval <= 0.0:
			continue
		next_sample = sample_interval
		next_epoch = int(np.ceil(next_sample))
		while next_epoch < n_epochs:
			bucket_due_epochs[bucket, next_epoch] = 1
			total_events += 1
			next_sample += sample_interval
			ceil_epoch = int(np.ceil(next_sample))
			if ceil_epoch <= next_epoch:
				next_epoch += 1
			else:
				next_epoch = ceil_epoch
	return total_events


@numba.njit(cache=True)
def _calendar_row_boundaries(event_epochs, event_sources):
	n_events = event_epochs.shape[0]
	if n_events == 0:
		return np.empty(0, dtype=np.int64)
	count = 1
	for idx in range(1, n_events):
		if (
			event_epochs[idx] != event_epochs[idx - 1]
			or event_sources[idx] != event_sources[idx - 1]
		):
			count += 1
	boundaries = np.empty(count, dtype=np.int64)
	boundaries[0] = 0
	write = 1
	for idx in range(1, n_events):
		if (
			event_epochs[idx] != event_epochs[idx - 1]
			or event_sources[idx] != event_sources[idx - 1]
		):
			boundaries[write] = idx
			write += 1
	return boundaries


def _build_sampling_attraction_calendar(
	edgesrc,
	epochs_per_sample,
	n_epochs,
	memory_limit_bytes=None,
):
	t_count = perf_counter()
	total_events = int(
		_count_sampling_calendar_events(edgesrc, epochs_per_sample, int(n_epochs))
	)
	count_time = perf_counter() - t_count
	# Three int32 event arrays, one int64 sort order, and reordered int32 arrays.
	# The estimate is intentionally conservative enough to catch accidental
	# full-scale pre-expansion before allocating.
	estimated_bytes = int(total_events * (3 * 4 + 8 + 3 * 4))
	if memory_limit_bytes is None:
		memory_limit_bytes = 512 * 1024 * 1024
	if memory_limit_bytes > 0 and estimated_bytes > int(memory_limit_bytes):
		raise MemoryError(
			"active-edge attraction calendar would allocate approximately "
			f"{estimated_bytes:,} bytes for {total_events:,} events; increase "
			"attraction_calendar_memory_limit_bytes or use row_scan"
		)
	t_fill = perf_counter()
	event_epochs = np.empty(total_events, dtype=np.int32)
	event_sources = np.empty(total_events, dtype=np.int32)
	event_edges = np.empty(total_events, dtype=np.int32)
	_fill_sampling_calendar_events(
		edgesrc,
		epochs_per_sample,
		int(n_epochs),
		event_epochs,
		event_sources,
		event_edges,
	)
	fill_time = perf_counter() - t_fill
	if total_events == 0:
		return (
			np.zeros(int(n_epochs) + 1, dtype=np.int64),
			np.empty(0, dtype=np.int32),
			np.zeros(1, dtype=np.int64),
			np.empty(0, dtype=np.int32),
			{
				"events": 0,
				"active_rows": 0,
				"estimated_bytes": estimated_bytes,
				"build_count_time": count_time,
				"build_fill_time": fill_time,
				"build_sort_time": 0.0,
				"build_group_time": 0.0,
				"build_schedule_time": 0.0,
				"build_bucket_count": 0,
			},
		)
	t_sort = perf_counter()
	order = np.lexsort((event_edges, event_sources, event_epochs))
	event_epochs = np.ascontiguousarray(event_epochs[order])
	event_sources = np.ascontiguousarray(event_sources[order])
	event_edges = np.ascontiguousarray(event_edges[order])
	sort_time = perf_counter() - t_sort
	t_group = perf_counter()
	boundaries = _calendar_row_boundaries(event_epochs, event_sources)
	calendar_rows = np.ascontiguousarray(event_sources[boundaries])
	calendar_edge_offsets = np.empty(boundaries.shape[0] + 1, dtype=np.int64)
	calendar_edge_offsets[:-1] = boundaries
	calendar_edge_offsets[-1] = total_events
	rows_per_epoch = np.bincount(
		event_epochs[boundaries], minlength=int(n_epochs)
	).astype(np.int64, copy=False)
	calendar_epoch_row_offsets = np.empty(int(n_epochs) + 1, dtype=np.int64)
	calendar_epoch_row_offsets[0] = 0
	np.cumsum(rows_per_epoch, out=calendar_epoch_row_offsets[1:])
	group_time = perf_counter() - t_group
	return (
		calendar_epoch_row_offsets,
		calendar_rows,
		calendar_edge_offsets,
		event_edges,
		{
			"events": total_events,
			"active_rows": int(calendar_rows.shape[0]),
			"estimated_bytes": estimated_bytes,
			"build_count_time": count_time,
			"build_fill_time": fill_time,
			"build_sort_time": sort_time,
			"build_group_time": group_time,
			"build_schedule_time": 0.0,
			"build_bucket_count": 0,
		},
	)


def _build_sampling_attraction_periodic(
	edgesrc,
	epochs_per_sample,
	n_epochs,
	memory_limit_bytes=None,
):
	t_bucket = perf_counter()
	edge_count = int(epochs_per_sample.shape[0])
	active_edges = np.flatnonzero(np.asarray(epochs_per_sample) > 0.0).astype(
		np.int32,
		copy=False,
	)
	if active_edges.size:
		bucket_intervals, bucket_ids = np.unique(
			np.asarray(epochs_per_sample)[active_edges],
			return_inverse=True,
		)
		bucket_intervals = np.ascontiguousarray(bucket_intervals)
		bucket_ids = np.asarray(bucket_ids, dtype=np.int32)
		bucket_counts = np.bincount(
			bucket_ids, minlength=bucket_intervals.shape[0]
		).astype(np.int64, copy=False)
	else:
		bucket_intervals = np.empty(0, dtype=np.asarray(epochs_per_sample).dtype)
		bucket_ids = np.empty(0, dtype=np.int32)
		bucket_counts = np.empty(0, dtype=np.int64)
	bucket_offsets = np.empty(bucket_counts.shape[0] + 1, dtype=np.int64)
	bucket_offsets[0] = 0
	np.cumsum(bucket_counts, out=bucket_offsets[1:])
	bucket_time = perf_counter() - t_bucket

	t_fill = perf_counter()
	edge_sources = _edge_sources_from_csr(edgesrc, edge_count)
	bucket_edges, bucket_edge_sources = _fill_periodic_bucket_edges(
		active_edges,
		edge_sources,
		bucket_ids,
		bucket_offsets,
	)
	fill_time = perf_counter() - t_fill

	t_schedule = perf_counter()
	estimated_bytes = int(
		bucket_offsets.nbytes
		+ bucket_edges.nbytes
		+ bucket_edge_sources.nbytes
		+ (bucket_intervals.shape[0] * int(n_epochs))
		+ bucket_intervals.nbytes
		+ (bucket_intervals.shape[0] * np.dtype(np.int64).itemsize)
	)
	if memory_limit_bytes is None:
		memory_limit_bytes = 512 * 1024 * 1024
	if memory_limit_bytes > 0 and estimated_bytes > int(memory_limit_bytes):
		raise MemoryError(
			"periodic active-edge attraction schedule would allocate approximately "
			f"{estimated_bytes:,} bytes for {bucket_intervals.shape[0]:,} buckets; "
			"increase attraction_calendar_memory_limit_bytes or use row_scan"
		)
	bucket_due_epochs = np.zeros(
		(bucket_intervals.shape[0], int(n_epochs)),
		dtype=np.uint8,
	)
	bucket_due_counts = np.zeros(bucket_intervals.shape[0], dtype=np.int64)
	due_pattern_count = int(
		_fill_periodic_bucket_due_epochs(
			bucket_intervals,
			int(n_epochs),
			bucket_due_epochs,
		)
	)
	if bucket_intervals.shape[0]:
		bucket_due_counts = np.sum(bucket_due_epochs, axis=1, dtype=np.int64)
	total_events = int(np.sum(bucket_due_counts * bucket_counts))
	schedule_time = perf_counter() - t_schedule

	return (
		bucket_offsets,
		bucket_edges,
		bucket_edge_sources,
		bucket_due_epochs,
		{
			"events": total_events,
			"active_rows": 0,
			"estimated_bytes": estimated_bytes,
			"build_count_time": bucket_time,
			"build_fill_time": fill_time,
			"build_sort_time": 0.0,
			"build_group_time": 0.0,
			"build_schedule_time": schedule_time,
			"build_bucket_count": int(bucket_intervals.shape[0]),
			"build_due_pattern_count": due_pattern_count,
		},
	)


@numba.njit(parallel=True, cache=True)
def _zero_calendar_attr_rows(attr_force, calendar_rows, row_start, row_end):
	for row_pos in numba.prange(row_start, row_end):
		i = calendar_rows[row_pos]
		attr_force[i][0] = 0.0
		attr_force[i][1] = 0.0


@numba.njit(parallel=True, cache=True)
def UMAP_AttrForce_sampling_calendar(
	attr_force,
	embedding,
	calendar_rows,
	calendar_edge_offsets,
	calendar_edges,
	edgetgt,
	row_start,
	row_end,
	a,
	b,
	alpha,
):
	for row_pos in numba.prange(row_start, row_end):
		i = calendar_rows[row_pos]
		temp0 = 0.0
		temp1 = 0.0
		for pos in range(calendar_edge_offsets[row_pos], calendar_edge_offsets[row_pos + 1]):
			k = calendar_edges[pos]
			j = edgetgt[k]
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			if dist_squared > 0.0:
				distance_power_b = pow(dist_squared, b)
				grad_coeff = -2.0 * a * b * (distance_power_b / dist_squared)
				grad_coeff /= a * distance_power_b + 1.0
			else:
				grad_coeff = 0.0
			temp0 += alpha * clip(grad_coeff * mv0)
			temp1 += alpha * clip(grad_coeff * mv1)
		attr_force[i][0] = temp0
		attr_force[i][1] = temp1


@numba.njit(cache=True)
def UMAP_AttrForce_sampling_periodic(
	attr_force,
	embedding,
	bucket_edge_offsets,
	bucket_edges,
	bucket_edge_sources,
	bucket_due_epochs,
	row_epoch_marks,
	edgetgt,
	epoch_itr,
	a,
	b,
	alpha,
):
	active_events = 0
	active_rows = 0
	row_stamp = epoch_itr + 1
	for bucket in range(bucket_due_epochs.shape[0]):
		if bucket_due_epochs[bucket, epoch_itr] == 0:
			continue
		for pos in range(bucket_edge_offsets[bucket], bucket_edge_offsets[bucket + 1]):
			k = bucket_edges[pos]
			i = bucket_edge_sources[pos]
			j = edgetgt[k]
			if row_epoch_marks[i] != row_stamp:
				row_epoch_marks[i] = row_stamp
				active_rows += 1
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			if dist_squared > 0.0:
				distance_power_b = pow(dist_squared, b)
				grad_coeff = -2.0 * a * b * (distance_power_b / dist_squared)
				grad_coeff /= a * distance_power_b + 1.0
			else:
				grad_coeff = 0.0
			attr_force[i][0] += alpha * clip(grad_coeff * mv0)
			attr_force[i][1] += alpha * clip(grad_coeff * mv1)
			active_events += 1
	return active_events, active_rows


@numba.njit(parallel=True, cache=True)
def UMAP_ReplForce(
	repl_force,
	embedding,
	n_vertices,
	n_components,
	a,
	b,
	alpha,
	gamma,
	epsilon,
	neg_effects,
):
	for i in numba.prange(n_vertices):
		repl_force[i][0] = 0.0
		repl_force[i][1] = 0.0
		for j in range(n_vertices):
			if i == j: continue
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			# 计算梯度系数
			grad_coeff = 2.0 * gamma * b
			grad_coeff /= (epsilon + dist_squared) * ( a * pow(dist_squared, b) + 1 )
			grad_coeff *= neg_effects[i]	### 负采样效果
			# 计算梯度
			repl_force[i][0] += alpha * clip(grad_coeff * mv0)
			repl_force[i][1] += alpha * clip(grad_coeff * mv1)
			# for d in range(n_components):
			# 	if grad_coeff > 0.0:	grad_d = clip(grad_coeff * (current[d] - other[d]))
			# 	else:					grad_d = 4.0
			# 	repl_force[i][d] += grad_d * alpha


@numba.njit(parallel=True, cache=True)
def UMAP_ReplForce_full(
	repl_force,
	embedding,
	edgesrc,
	edgetgt,
	pos_effects,	# weights, `neg_effect = 1 - weight` for full repl
	n_vertices,
	n_components,
	a,
	b,
	alpha,
	gamma,
	epsilon,
):
	for i in numba.prange(n_vertices):
		repl_force[i][0] = 0.0
		repl_force[i][1] = 0.0
		connected_vertices = []
		connected_edges_idx = []
		for k in range(edgesrc[i], edgesrc[i+1]):
			connected_vertices.append(edgetgt[k])
			connected_edges_idx.append(k)
		for j in range(n_vertices):
			if i == j: continue
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			# 计算梯度系数
			grad_coeff = 2.0 * gamma * b
			grad_coeff /= (epsilon + dist_squared) * ( a * pow(dist_squared, b) + 1 )
			if j in connected_vertices:
				idx = connected_vertices.index(j)
				grad_coeff *= (1.0 - pos_effects[connected_edges_idx[idx]])
			# 计算梯度
			repl_force[i][0] += alpha * clip(grad_coeff * mv0)
			repl_force[i][1] += alpha * clip(grad_coeff * mv1)


@numba.njit(parallel=True, cache=True)
def UMAP_ApplyForce(
	embedding,
	attr_force,
	repl_force,
	n_vertices,
	n_components,
):
	# H-4 FIX (Option A): Apply attr_force + repl_force directly.
	# dC momentum accumulator removed — it was built up each epoch but never
	# read for the embedding update (the update lines were commented out in the
	# original code).  Removing dC eliminates ~50k float32 ops per epoch
	# (2×10k additions + 10k multiplies + 10k RNG samples) with zero change
	# to numerical results, since the old code also did not use dC.
	for i in numba.prange(n_vertices):
		embedding[i][0] += attr_force[i][0] + repl_force[i][0]
		embedding[i][1] += attr_force[i][1] + repl_force[i][1]


""" t-FDP Repulsive Force """

@numba.njit(parallel=True, cache=True)
def tFDP_ReplForce_Exact(
	repl_force,
	embedding,
	n_vertices,
	alpha,		# d3alpha
	gamma,		# 2.0
	paraFactor,	# 2.0
	neg_effects,
):
	for i in numba.prange(n_vertices):
		repl_force[i][0] = 0.0
		repl_force[i][1] = 0.0
		for j in range(n_vertices):
			if i == j: continue
			mv0 = embedding[i][0] - embedding[j][0]
			mv1 = embedding[i][1] - embedding[j][1]
			dist_squared = mv0 * mv0 + mv1 * mv1
			grad_coeff = alpha * paraFactor * pow(1.0 + dist_squared, -gamma) * neg_effects[i]
			repl_force[i][0] += grad_coeff * mv0
			repl_force[i][1] += grad_coeff * mv1


@numba.njit(parallel=True, cache=True)
def tFDP_ReplForce_ibFFT_neg_effect(
	repl_force,
	alpha,
	neg_effects,
):
	repl_force *= alpha
	repl_force[:, 0] = repl_force[:,0] * neg_effects
	repl_force[:, 1] = repl_force[:,1] * neg_effects
	return repl_force


@numba.njit(cache=True)
def _local_exact_repulsion_force(
	embedding,
	point_cell_x,
	point_cell_y,
	sorted_indices,
	cell_start,
	cell_end,
	n_cells_x,
	n_cells_y,
	k,
	radius,
	a,
	b,
	gamma,
	epsilon,
	alpha,
	component_clip,
	symmetric,
	density,
	flat_cell_id,
	density_filter,
	density_threshold,
	density_include_neighbor_cells,
):
	"""Compute an additive near-field correction from adjacent grid cells."""
	n_vertices = embedding.shape[0]
	force = np.zeros((n_vertices, 2), dtype=embedding.dtype)
	if k <= 0 or radius <= 0.0:
		return force
	radius_squared = radius * radius
	for i in range(n_vertices):
		if density_filter:
			if density_include_neighbor_cells:
				occupancy = 0
				cell_x = point_cell_x[i]
				cell_y = point_cell_y[i]
				for offset_x in range(-1, 2):
					neighbor_x = cell_x + offset_x
					if neighbor_x < 0 or neighbor_x >= n_cells_x:
						continue
					for offset_y in range(-1, 2):
						neighbor_y = cell_y + offset_y
						if neighbor_y < 0 or neighbor_y >= n_cells_y:
							continue
						occupancy += density[neighbor_x * n_cells_y + neighbor_y]
			else:
				occupancy = density[flat_cell_id[i]]
			if occupancy < density_threshold:
				continue
		best_indices = np.empty(k, dtype=np.int64)
		best_distances = np.empty(k, dtype=embedding.dtype)
		best_count = 0
		cell_x = point_cell_x[i]
		cell_y = point_cell_y[i]
		for offset_x in range(-1, 2):
			neighbor_x = cell_x + offset_x
			if neighbor_x < 0 or neighbor_x >= n_cells_x:
				continue
			for offset_y in range(-1, 2):
				neighbor_y = cell_y + offset_y
				if neighbor_y < 0 or neighbor_y >= n_cells_y:
					continue
				flat_cell = neighbor_x * n_cells_y + neighbor_y
				for position in range(cell_start[flat_cell], cell_end[flat_cell]):
					j = sorted_indices[position]
					if j == i or (symmetric and j < i):
						continue
					dx = embedding[i, 0] - embedding[j, 0]
					dy = embedding[i, 1] - embedding[j, 1]
					distance_squared = dx * dx + dy * dy
					if distance_squared <= 0.0 or distance_squared > radius_squared:
						continue
					if best_count < k:
						best_indices[best_count] = j
						best_distances[best_count] = distance_squared
						best_count += 1
					else:
						worst_slot = 0
						worst_distance = best_distances[0]
						for slot in range(1, k):
							if best_distances[slot] > worst_distance:
								worst_slot = slot
								worst_distance = best_distances[slot]
						if distance_squared < worst_distance:
							best_indices[worst_slot] = j
							best_distances[worst_slot] = distance_squared

		for slot in range(best_count):
			j = best_indices[slot]
			distance_squared = best_distances[slot]
			dx = embedding[i, 0] - embedding[j, 0]
			dy = embedding[i, 1] - embedding[j, 1]
			grad_coeff = 2.0 * gamma * b
			grad_coeff /= (epsilon + distance_squared) * (
				a * pow(distance_squared, b) + 1.0
			)
			grad_x = grad_coeff * dx
			grad_y = grad_coeff * dy
			grad_x = min(component_clip, max(-component_clip, grad_x)) * alpha
			grad_y = min(component_clip, max(-component_clip, grad_y)) * alpha
			force[i, 0] += grad_x
			force[i, 1] += grad_y
			if symmetric:
				force[j, 0] -= grad_x
				force[j, 1] -= grad_y
	return force


@numba.njit(cache=False)
def _local_exact_repulsion_force_profiled(
	embedding,
	point_cell_x,
	point_cell_y,
	sorted_indices,
	cell_start,
	cell_end,
	n_cells_x,
	n_cells_y,
	k,
	radius,
	a,
	b,
	gamma,
	epsilon,
	alpha,
	component_clip,
	symmetric,
	density,
	flat_cell_id,
	density_filter,
	density_threshold,
	density_include_neighbor_cells,
	timing_sample_size,
):
	"""Profile the fused local-exact path without changing its pair ordering."""
	n_vertices = embedding.shape[0]
	force = np.zeros((n_vertices, 2), dtype=embedding.dtype)
	candidate_counts = np.zeros(n_vertices, dtype=np.int64)
	distance_counts = np.zeros(n_vertices, dtype=np.int64)
	selected_counts = np.zeros(n_vertices, dtype=np.int64)
	eligible = np.zeros(n_vertices, dtype=np.uint8)
	stage_ticks = np.zeros(5, dtype=np.uint64)
	sampled_points = 0
	if k <= 0 or radius <= 0.0:
		return (
			force,
			candidate_counts,
			distance_counts,
			selected_counts,
			eligible,
			stage_ticks,
			sampled_points,
		)
	radius_squared = radius * radius
	timing_stride = max(n_vertices // max(int(timing_sample_size), 1), 1)
	for i in range(n_vertices):
		if density_filter:
			if density_include_neighbor_cells:
				occupancy = 0
				cell_x = point_cell_x[i]
				cell_y = point_cell_y[i]
				for offset_x in range(-1, 2):
					neighbor_x = cell_x + offset_x
					if neighbor_x < 0 or neighbor_x >= n_cells_x:
						continue
					for offset_y in range(-1, 2):
						neighbor_y = cell_y + offset_y
						if neighbor_y < 0 or neighbor_y >= n_cells_y:
							continue
						occupancy += density[neighbor_x * n_cells_y + neighbor_y]
			else:
				occupancy = density[flat_cell_id[i]]
			if occupancy < density_threshold:
				continue
		eligible[i] = 1
		profile_point = i % timing_stride == 0
		if profile_point:
			sampled_points += 1
		best_indices = np.empty(k, dtype=np.int64)
		best_distances = np.empty(k, dtype=embedding.dtype)
		best_count = 0
		cell_x = point_cell_x[i]
		cell_y = point_cell_y[i]
		for offset_x in range(-1, 2):
			neighbor_x = cell_x + offset_x
			if neighbor_x < 0 or neighbor_x >= n_cells_x:
				continue
			for offset_y in range(-1, 2):
				neighbor_y = cell_y + offset_y
				if neighbor_y < 0 or neighbor_y >= n_cells_y:
					continue
				flat_cell = neighbor_x * n_cells_y + neighbor_y
				for position in range(cell_start[flat_cell], cell_end[flat_cell]):
					if profile_point:
						stage_start = _profile_ticks()
					j = sorted_indices[position]
					pair_allowed = j != i and not (symmetric and j < i)
					if profile_point:
						stage_ticks[0] += _profile_ticks() - stage_start
					if not pair_allowed:
						continue
					candidate_counts[i] += 1
					if profile_point:
						stage_start = _profile_ticks()
					dx = embedding[i, 0] - embedding[j, 0]
					dy = embedding[i, 1] - embedding[j, 1]
					distance_squared = dx * dx + dy * dy
					in_radius = distance_squared > 0.0 and distance_squared <= radius_squared
					if profile_point:
						stage_ticks[1] += _profile_ticks() - stage_start
					if not in_radius:
						continue
					distance_counts[i] += 1
					if profile_point:
						stage_start = _profile_ticks()
					if best_count < k:
						best_indices[best_count] = j
						best_distances[best_count] = distance_squared
						best_count += 1
					else:
						worst_slot = 0
						worst_distance = best_distances[0]
						for slot in range(1, k):
							if best_distances[slot] > worst_distance:
								worst_slot = slot
								worst_distance = best_distances[slot]
						if distance_squared < worst_distance:
							best_indices[worst_slot] = j
							best_distances[worst_slot] = distance_squared
					if profile_point:
						stage_ticks[2] += _profile_ticks() - stage_start

		selected_counts[i] = best_count
		for slot in range(best_count):
			if profile_point:
				stage_start = _profile_ticks()
			j = best_indices[slot]
			distance_squared = best_distances[slot]
			dx = embedding[i, 0] - embedding[j, 0]
			dy = embedding[i, 1] - embedding[j, 1]
			grad_coeff = 2.0 * gamma * b
			grad_coeff /= (epsilon + distance_squared) * (
				a * pow(distance_squared, b) + 1.0
			)
			grad_x = grad_coeff * dx
			grad_y = grad_coeff * dy
			grad_x = min(component_clip, max(-component_clip, grad_x)) * alpha
			grad_y = min(component_clip, max(-component_clip, grad_y)) * alpha
			if profile_point:
				stage_ticks[3] += _profile_ticks() - stage_start
				stage_start = _profile_ticks()
			force[i, 0] += grad_x
			force[i, 1] += grad_y
			if symmetric:
				force[j, 0] -= grad_x
				force[j, 1] -= grad_y
			if profile_point:
				stage_ticks[4] += _profile_ticks() - stage_start
	return (
		force,
		candidate_counts,
		distance_counts,
		selected_counts,
		eligible,
		stage_ticks,
		sampled_points,
	)


def _local_density_pressure_force(
	embedding,
	grid_context,
	min_count,
	component_clip,
	power,
):
	"""Cheap grid-density heuristic for diagnosing local embedding collapse."""
	dtype = np.dtype(embedding.dtype)
	force = np.zeros_like(embedding, dtype=dtype)
	occupancy = grid_context.density[grid_context.flat_cell_id]
	active = occupancy > int(min_count)
	if not np.any(active):
		return force
	centers_x = grid_context.min_xy[0] + (
		grid_context.point_cell_x[active].astype(dtype) + dtype.type(0.5)
	) * grid_context.cell_size
	centers_y = grid_context.min_xy[1] + (
		grid_context.point_cell_y[active].astype(dtype) + dtype.type(0.5)
	) * grid_context.cell_size
	direction = embedding[active].astype(dtype, copy=False).copy()
	direction[:, 0] -= centers_x
	direction[:, 1] -= centers_y
	norm = np.sqrt(np.sum(direction * direction, axis=1)) + dtype.type(1e-12)
	pressure = np.power(
		(occupancy[active].astype(dtype) - dtype.type(min_count)) / dtype.type(min_count),
		float(power),
	)
	active_force = pressure.reshape((-1, 1)) * direction / norm.reshape((-1, 1))
	force[active] = np.clip(active_force, -float(component_clip), float(component_clip))
	return force


def _resolve_local_exact_epoch_bounds(drfft_params, n_epochs):
	start_epoch = drfft_params.get("local_exact_start_epoch")
	end_epoch = drfft_params.get("local_exact_end_epoch")
	start_frac = drfft_params.get("local_exact_start_frac")
	end_frac = drfft_params.get("local_exact_end_frac")
	if start_frac is not None:
		start_epoch = int(np.floor(float(start_frac) * int(n_epochs)))
	if end_frac is not None:
		end_epoch = int(np.ceil(float(end_frac) * int(n_epochs)))
	start_epoch = 0 if start_epoch is None else int(start_epoch)
	end_epoch = int(n_epochs) if end_epoch is None else int(end_epoch)
	return max(0, start_epoch), min(int(n_epochs), end_epoch)


def _local_enabled_for_epoch(enabled, every, epoch_itr, start_epoch=0, end_epoch=None):
	if end_epoch is None:
		end_epoch = np.iinfo(np.int64).max
	return (
		bool(enabled)
		and int(start_epoch) <= int(epoch_itr) < int(end_epoch)
		and int(epoch_itr) % int(every) == 0
	)


def _local_exact_density_threshold(drfft_params, grid_context):
	if not bool(drfft_params.get("local_exact_density_filter", False)):
		return 0.0
	min_count = drfft_params.get("local_exact_min_cell_count")
	if min_count is not None:
		return float(min_count)
	quantile = drfft_params.get("local_exact_min_cell_count_quantile")
	if quantile is None:
		return 1.0
	occupied = grid_context.density[grid_context.density > 0]
	if occupied.size == 0:
		return np.inf
	return float(np.quantile(occupied.astype(np.float64), float(quantile)))


def _local_edge_sample(edgesrc, edgetgt, max_edges=100000):
	"""Choose a deterministic positive-edge sample for adaptive radius estimates."""
	edge_count = int(len(edgetgt))
	if edge_count == 0:
		return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
	if edge_count <= int(max_edges):
		edge_positions = np.arange(edge_count, dtype=np.int64)
	else:
		edge_positions = np.linspace(0, edge_count - 1, int(max_edges), dtype=np.int64)
	sources = np.searchsorted(np.asarray(edgesrc[1:]), edge_positions, side="right")
	targets = np.asarray(edgetgt)[edge_positions]
	return sources.astype(np.int64, copy=False), targets.astype(np.int64, copy=False)


def _local_exact_radius(embedding, edge_sources, edge_targets, radius_factor, cell_size):
	if edge_sources.size:
		difference = embedding[edge_sources] - embedding[edge_targets]
		edge_lengths = np.sqrt(np.sum(difference * difference, axis=1))
		positive_lengths = edge_lengths[edge_lengths > 0.0]
		if positive_lengths.size:
			return float(radius_factor) * float(np.median(positive_lengths))
	return float(radius_factor) * float(cell_size)


def _local_force_diagnostics(prefix, enabled, weight, norms, n_vertices, extra=None):
	row = {
		f"{prefix}_enabled": int(bool(enabled)),
		f"{prefix}_weight": float(weight),
		f"{prefix}_active_count": 0,
		f"{prefix}_active_pct": 0.0,
	}
	if extra:
		row.update(extra)
	if norms is None:
		row.update({
			f"{prefix}_force_max": 0.0,
			f"{prefix}_force_mean": 0.0,
			f"{prefix}_force_p95": 0.0,
			f"{prefix}_force_p99": 0.0,
			f"{prefix}_force_p999": 0.0,
		})
		return row
	active = int(np.sum(norms > 0.0))
	row[f"{prefix}_active_count"] = active
	row[f"{prefix}_active_pct"] = 100.0 * float(active) / float(max(int(n_vertices), 1))
	row.update(_norm_stats(f"{prefix}_force", norms))
	return row


def _noise_scale_for_epoch(drfft_params, epoch_itr, n_epochs):
	noise_mode = drfft_params.get('noise_mode', 'none')
	base_scale = float(drfft_params.get('noise_scale', 0.0) or 0.0)
	if noise_mode == 'none' or base_scale <= 0.0:
		return 0.0

	active_until = drfft_params.get('noise_until_epoch')
	active_until = n_epochs if active_until is None else int(active_until)

	if drfft_params.get('hybrid_mode', 'none') == 'early_noisy_late_deterministic':
		switch_epoch = drfft_params.get('hybrid_switch_epoch')
		switch_epoch = n_epochs // 2 if switch_epoch is None else int(switch_epoch)
		active_until = min(active_until, switch_epoch)

	if active_until <= 0 or epoch_itr >= active_until:
		return 0.0

	decay = drfft_params.get('noise_decay', 'linear')
	if decay == 'constant':
		return base_scale
	progress = float(epoch_itr) / float(max(active_until, 1))
	if decay == 'linear':
		return base_scale * max(0.0, 1.0 - progress)
	if decay == 'exponential':
		return base_scale * float(np.exp(-5.0 * progress))
	raise ValueError(f"Unsupported noise_decay: {decay}")


def _make_noise_rng(is_gpu, seed):
	if is_gpu:
		if seed is None:
			return cupy.random.RandomState()
		return cupy.random.RandomState(int(seed))
	return np.random.default_rng(seed)


def _make_sampling_rng(is_gpu, seed):
	if seed is None:
		return None
	if is_gpu:
		return cupy.random.RandomState(int(seed))
	return np.random.RandomState(int(seed))


def _resolve_noise_seed(drfft_params):
	explicit_seed = drfft_params.get('noise_seed')
	if explicit_seed is not None:
		return explicit_seed
	return drfft_params.get('noise_random_state')


def _add_gaussian_noise_inplace(target, rng, scale, is_gpu, hard_constraint, known_mask):
	if scale <= 0.0:
		return
	if is_gpu:
		noise = rng.normal(0.0, scale, size=target.shape).astype(target.dtype, copy=False)
	else:
		noise = rng.standard_normal(target.shape).astype(target.dtype, copy=False)
		noise *= scale
	if hard_constraint:
		noise[known_mask, :] = 0.0
	target += noise


def _scalar_float(value):
	if hasattr(value, "item"):
		value = value.item()
	return float(value)


def _validate_clip_epoch_range_for_epochs(epoch_range, n_epochs, name):
	if epoch_range is None:
		return None
	start, end = epoch_range
	if start < 0 or start >= end or end > int(n_epochs):
		raise ValueError(
			f"{name} must satisfy 0 <= start < end <= resolved n_epochs ({int(n_epochs)})"
		)
	return int(start), int(end)


def _clip_enabled_for_epoch(clip_norm, epoch_range, epoch_itr):
	if clip_norm is None:
		return False
	if epoch_range is None:
		return True
	start, end = epoch_range
	return int(start) <= int(epoch_itr) < int(end)


def _resolve_repulsion_clip_norm(drfft_params, alpha, epoch_itr=0, n_epochs=None):
	clip_norm = drfft_params.get("repulsion_clip_norm")
	epoch_range = drfft_params.get("repulsion_clip_epoch_range")
	if n_epochs is not None:
		epoch_range = _validate_clip_epoch_range_for_epochs(
			epoch_range, n_epochs, "repulsion_clip_epoch_range"
		)
	if not _clip_enabled_for_epoch(clip_norm, epoch_range, epoch_itr):
		return None
	clip_norm = float(clip_norm)
	if clip_norm <= 0.0:
		return None
	if drfft_params.get("repulsion_clip_with_alpha", False):
		clip_norm *= _scalar_float(alpha)
	return clip_norm


def _resolve_total_update_clip_norm(drfft_params, alpha, epoch_itr=0, n_epochs=None):
	clip_norm = drfft_params.get("total_update_clip_norm")
	epoch_range = drfft_params.get("total_update_clip_epoch_range")
	if n_epochs is not None:
		epoch_range = _validate_clip_epoch_range_for_epochs(
			epoch_range, n_epochs, "total_update_clip_epoch_range"
		)
	if not _clip_enabled_for_epoch(clip_norm, epoch_range, epoch_itr):
		return None
	clip_norm = float(clip_norm)
	if clip_norm <= 0.0:
		return None
	if drfft_params.get("total_update_clip_with_alpha", False):
		clip_norm *= _scalar_float(alpha)
	return clip_norm


def _force_norms_numpy(force, is_gpu):
	if cupy is not None and isinstance(force, cupy.ndarray):
		norms = cupy.sqrt(cupy.sum(force * force, axis=1))
		return cupy.asnumpy(norms)
	return np.sqrt(np.sum(force * force, axis=1))


def _clip_force_norm_inplace(force, clip_norm, is_gpu, return_norms=True):
	# Diagnostic option: clip final per-point ibFFT repulsion updates by vector norm.
	if is_gpu:
		norms = cupy.sqrt(cupy.sum(force * force, axis=1))
		scale = cupy.minimum(1.0, cupy.float32(clip_norm) / (norms + cupy.float32(1e-12)))
		force *= scale.reshape((-1, 1))
		# Host copies synchronize the GPU and are needed only for diagnostics.
		return cupy.asnumpy(norms) if return_norms else None
	norms = np.sqrt(np.sum(force * force, axis=1))
	scale = np.minimum(1.0, float(clip_norm) / (norms + 1e-12))
	force *= scale.reshape((-1, 1))
	return norms


def _clip_total_update_norm_inplace(
	attr_force, repl_force, clip_norm, is_gpu, return_norms=True
):
	if is_gpu:
		total = attr_force + repl_force
		norms = cupy.sqrt(cupy.sum(total * total, axis=1))
		scale = cupy.minimum(1.0, cupy.float32(clip_norm) / (norms + cupy.float32(1e-12)))
		scale = scale.reshape((-1, 1))
		attr_force *= scale
		repl_force *= scale
		return cupy.asnumpy(norms) if return_norms else None
	total = attr_force + repl_force
	norms = np.sqrt(np.sum(total * total, axis=1))
	scale = np.minimum(1.0, float(clip_norm) / (norms + 1e-12)).reshape((-1, 1))
	attr_force *= scale
	repl_force *= scale
	return norms


def _norm_stats(prefix, norms):
	if norms.size == 0:
		values = [np.nan, np.nan, np.nan, np.nan, np.nan]
	else:
		values = [
			np.max(norms),
			np.mean(norms),
			*np.percentile(norms, [95.0, 99.0, 99.9]),
		]
	return {
		f"{prefix}_max": float(values[0]),
		f"{prefix}_mean": float(values[1]),
		f"{prefix}_p95": float(values[2]),
		f"{prefix}_p99": float(values[3]),
		f"{prefix}_p999": float(values[4]),
	}


def _threshold_label(value):
	return ("%g" % float(value)).replace("-", "m").replace(".", "p")


def _add_threshold_counts(row, prefix, norms, thresholds):
	n = max(int(norms.size), 1)
	for threshold in thresholds:
		label = _threshold_label(threshold)
		count = int(np.sum(norms > float(threshold)))
		row[f"{prefix}_gt_{label}_count"] = count
		row[f"{prefix}_gt_{label}_pct"] = 100.0 * float(count) / float(n)


def _embedding_bbox(embedding, is_gpu):
	values = _to_numpy_array(embedding).astype(np.float64, copy=False)
	x = values[:, 0]
	y = values[:, 1]
	x_min, x_max = float(np.min(x)), float(np.max(x))
	y_min, y_max = float(np.min(y)), float(np.max(y))
	x_p01, x_p99 = np.percentile(x, [1.0, 99.0])
	y_p01, y_p99 = np.percentile(y, [1.0, 99.0])
	bbox_width = x_max - x_min
	bbox_height = y_max - y_min
	robust_width = float(x_p99 - x_p01)
	robust_height = float(y_p99 - y_p01)
	bbox_area = float(bbox_width * bbox_height)
	robust_area = float(robust_width * robust_height)
	center = np.median(values[:, :2], axis=0)
	radii = np.linalg.norm(values[:, :2] - center, axis=1)
	expansion = bbox_area / max(robust_area, np.finfo(np.float64).tiny)
	return {
		"x_min": x_min,
		"x_max": x_max,
		"y_min": y_min,
		"y_max": y_max,
		"bbox_width": bbox_width,
		"bbox_height": bbox_height,
		"bbox_area": bbox_area,
		"robust_x_min": float(x_p01),
		"robust_x_max": float(x_p99),
		"robust_y_min": float(y_p01),
		"robust_y_max": float(y_p99),
		"robust_bbox_width": robust_width,
		"robust_bbox_height": robust_height,
		"robust_bbox_area": robust_area,
		"bbox_area_expansion_ratio": expansion,
		"full_to_robust_bbox_ratio": expansion,
		"embedding_center_x": float(center[0]),
		"embedding_center_y": float(center[1]),
		"radial_max": float(np.max(radii)),
		"radial_p95": float(np.percentile(radii, 95.0)),
		"radial_p99": float(np.percentile(radii, 99.0)),
		"radial_p999": float(np.percentile(radii, 99.9)),
	}


def _diagnostic_fieldnames(thresholds):
	fields = [
		"epoch",
		"alpha",
		"attraction_degree_damping_enabled",
		"attraction_degree_damping_mode",
		"attraction_degree_damping_ref",
		"attraction_degree_damping_degree_ref",
		"attraction_degree_damping_power",
		"attraction_degree_damping_min_scale",
		"attraction_degree_damping_degree_mean",
		"attraction_degree_damping_degree_median",
		"attraction_degree_damping_degree_p90",
		"attraction_degree_damping_degree_p95",
		"attraction_degree_damping_degree_p99",
		"attraction_degree_damping_degree_max",
		"attraction_degree_damping_scale_mean",
		"attraction_degree_damping_scale_median",
		"attraction_degree_damping_scale_p90",
		"attraction_degree_damping_scale_p95",
		"attraction_degree_damping_scale_p99",
		"attraction_degree_damping_scale_min",
		"attraction_degree_damping_active_count",
		"attraction_degree_damping_active_pct",
		"ibfft_kernel_clip",
		"umap_epsilon",
		"ibfft_kernel_subsample_mode",
		"ibfft_kernel_subsample_radius_cells",
		"ibfft_kernel_subsample_points",
		"repulsion_clip_enabled_this_epoch",
		"repulsion_clip_effective_norm",
		"repulsion_clip_norm",
		"repulsion_clip_active_count",
		"repulsion_clip_active_pct",
		"total_update_clip_enabled_this_epoch",
		"total_update_clip_effective_norm",
		"total_update_clip_norm",
		"total_update_clip_active_count",
		"total_update_clip_active_pct",
		"local_exact_enabled",
		"local_exact_weight",
		"local_exact_k",
		"local_exact_radius",
		"local_exact_active_count",
		"local_exact_active_pct",
		"local_exact_force_max",
		"local_exact_force_mean",
		"local_exact_force_p95",
		"local_exact_force_p99",
		"local_exact_force_p999",
		"local_density_pressure_enabled",
		"local_density_pressure_weight",
		"local_density_pressure_active_count",
		"local_density_pressure_active_pct",
		"local_density_pressure_force_max",
		"local_density_pressure_force_mean",
		"local_density_pressure_force_p95",
		"local_density_pressure_force_p99",
		"local_density_pressure_force_p999",
	]
	for prefix in (
		"attr_norm_pre_degree_damping",
		"attr_norm_post_degree_damping",
		"attr_force",
		"repl_force_pre_clip",
		"repl_force",
		"total_update_pre_clip",
		"total_update",
	):
		fields.extend([
			f"{prefix}_max",
			f"{prefix}_mean",
			f"{prefix}_p95",
			f"{prefix}_p99",
			f"{prefix}_p999",
		])
	fields.extend([
		"x_min", "x_max", "y_min", "y_max", "bbox_width", "bbox_height", "bbox_area",
		"robust_x_min", "robust_x_max", "robust_y_min", "robust_y_max",
		"robust_bbox_width", "robust_bbox_height", "robust_bbox_area",
		"bbox_area_expansion_ratio", "full_to_robust_bbox_ratio",
		"embedding_center_x", "embedding_center_y",
		"radial_max", "radial_p95", "radial_p99", "radial_p999",
		"grid_x_min", "grid_x_max", "grid_y_min", "grid_y_max",
		"grid_cell_size", "grid_h", "n_boxes_per_dim",
	])
	for prefix in ("repl_pre_clip", "repl", "total_update_pre_clip", "total_update"):
		for threshold in thresholds:
			label = _threshold_label(threshold)
			fields.extend([f"{prefix}_gt_{label}_count", f"{prefix}_gt_{label}_pct"])
	return fields


def _open_diagnostic_writer(path, thresholds):
	if not path:
		return None, None
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	handle = output_path.open("w", newline="")
	writer = csv.DictWriter(handle, fieldnames=_diagnostic_fieldnames(thresholds))
	writer.writeheader()
	return handle, writer


def _runtime_diagnostic_fieldnames():
	return [
		"epoch",
		"epoch_time_s",
		"attraction_time_s",
		"ibfft_time_s",
		"local_exact_total_time_s",
		"local_grid_time_s",
		"local_prep_time_s",
		"local_candidate_time_s",
		"local_distance_time_s",
		"local_topk_time_s",
		"local_force_time_s",
		"local_accum_time_s",
		"local_array_copy_time_s",
		"update_time_s",
		"diagnostics_time_s",
		"local_exact_ran",
		"local_exact_density_threshold",
		"local_exact_eligible_count",
		"local_exact_eligible_pct",
		"local_exact_active_count",
		"local_exact_active_pct",
		"local_exact_pair_count",
		"local_exact_candidate_count_mean",
		"local_exact_candidate_count_p95",
		"local_exact_candidate_count_p99",
		"local_exact_distance_count_mean",
		"local_exact_distance_count_p95",
		"local_exact_distance_count_p99",
		"local_exact_selected_count_mean",
		"local_exact_selected_count_p95",
		"local_exact_selected_count_p99",
		"cell_occupancy_mean",
		"cell_occupancy_p95",
		"cell_occupancy_p99",
		"cell_occupancy_max",
		"local_timing_sample_count",
	]


def _open_runtime_diagnostic_writer(path):
	if not path:
		return None, None
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	handle = output_path.open("w", newline="")
	writer = csv.DictWriter(handle, fieldnames=_runtime_diagnostic_fieldnames())
	writer.writeheader()
	return handle, writer


def _count_stats(prefix, values):
	if values.size == 0:
		return {
			f"{prefix}_mean": 0.0,
			f"{prefix}_p95": 0.0,
			f"{prefix}_p99": 0.0,
		}
	return {
		f"{prefix}_mean": float(np.mean(values)),
		f"{prefix}_p95": float(np.percentile(values, 95.0)),
		f"{prefix}_p99": float(np.percentile(values, 99.0)),
	}


def _to_numpy_array(values):
	if cupy is not None and isinstance(values, cupy.ndarray):
		return cupy.asnumpy(values)
	return np.asarray(values)


def _compute_attraction_degree_damping_scale(
	edgesrc,
	weights,
	n_vertices,
	mode="weighted_degree",
	ref="p99",
	ref_value=None,
	power=0.5,
	min_scale=None,
	enabled=True,
	is_gpu=False,
	weighted_degrees=None,
):
	"""Compute source-row graph degrees and the optional attraction scale.

	ibUMAP accumulates attraction into the source point of each CSR row. The
	fuzzy graph is normally symmetric, so a row contains each undirected edge
	incident to that point once. Matching row semantics also keeps caller-supplied
	asymmetric graphs consistent with the force actually accumulated.
	"""
	if mode not in ("weighted_degree", "degree"):
		raise ValueError("mode must be 'weighted_degree' or 'degree'")
	if ref not in ("median", "mean", "p90", "p95", "p99", "manual"):
		raise ValueError("unsupported attraction degree damping reference")
	if float(power) <= 0.0:
		raise ValueError("power must be positive")
	if min_scale is not None and not 0.0 < float(min_scale) <= 1.0:
		raise ValueError("min_scale must be in (0, 1]")

	xp = cupy if is_gpu else np
	value_dtype = cupy.float32 if is_gpu else np.dtype(np.asarray(weights).dtype)
	if not is_gpu and value_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
		value_dtype = np.dtype(np.float32)
	scalar = cupy.float32 if is_gpu else value_dtype.type
	n_vertices = int(n_vertices)
	if mode == "weighted_degree":
		if weighted_degrees is not None:
			degree_values = xp.asarray(weighted_degrees, dtype=value_dtype).reshape(-1)
		else:
			offsets = xp.asarray(edgesrc)
			weight_values = xp.asarray(weights, dtype=value_dtype)
			prefix = xp.concatenate(
				(xp.zeros(1, dtype=value_dtype), xp.cumsum(weight_values, dtype=value_dtype))
			)
			degree_values = prefix[offsets[1:]] - prefix[offsets[:-1]]
	else:
		offsets = xp.asarray(edgesrc)
		degree_values = (offsets[1:] - offsets[:-1]).astype(value_dtype, copy=False)
	if int(degree_values.size) != n_vertices:
		raise ValueError("degree values must contain one value per vertex")

	if ref == "manual":
		if ref_value is None or float(ref_value) <= 0.0:
			raise ValueError("manual degree reference requires a positive ref_value")
		degree_ref = float(ref_value)
	elif int(degree_values.size) == 0:
		degree_ref = 0.0
	elif ref == "mean":
		degree_ref = _scalar_float(xp.mean(degree_values))
	elif ref == "median":
		degree_ref = _scalar_float(xp.median(degree_values))
	else:
		percentile = {"p90": 90.0, "p95": 95.0, "p99": 99.0}[ref]
		degree_ref = _scalar_float(xp.percentile(degree_values, percentile))

	if enabled and (not np.isfinite(degree_ref) or degree_ref <= 0.0):
		raise ValueError(
			"attraction degree damping requires a positive resolved degree_ref"
		)
	scale = xp.ones(n_vertices, dtype=value_dtype)
	if enabled:
		positive = degree_values > scalar(0.0)
		ratio = xp.where(
			positive,
			scalar(degree_ref) / xp.maximum(degree_values, scalar(1e-12)),
			scalar(1.0),
		)
		scale = xp.minimum(
			scalar(1.0), ratio ** scalar(power)
		).astype(value_dtype, copy=False)
		# Isolated/non-positive-degree points must not be damped.
		scale = xp.where(positive, scale, scalar(1.0))
		if min_scale is not None:
			scale = xp.maximum(scale, scalar(min_scale))
	return degree_values.astype(value_dtype, copy=False), degree_ref, scale


def _attraction_degree_damping_metadata(
	enabled,
	mode,
	ref,
	degree_ref,
	power,
	min_scale,
	degree_values,
	damping_scale,
):
	degree_values = _to_numpy_array(degree_values).astype(np.float64, copy=False)
	damping_scale = _to_numpy_array(damping_scale).astype(np.float64, copy=False)
	if degree_values.size:
		degree_stats = {
			"mean": np.mean(degree_values),
			"median": np.median(degree_values),
			"p90": np.percentile(degree_values, 90.0),
			"p95": np.percentile(degree_values, 95.0),
			"p99": np.percentile(degree_values, 99.0),
			"max": np.max(degree_values),
		}
		scale_stats = {
			"mean": np.mean(damping_scale),
			"median": np.median(damping_scale),
			"p90": np.percentile(damping_scale, 90.0),
			"p95": np.percentile(damping_scale, 95.0),
			"p99": np.percentile(damping_scale, 99.0),
			"min": np.min(damping_scale),
		}
	else:
		degree_stats = {
			name: np.nan for name in ("mean", "median", "p90", "p95", "p99", "max")
		}
		scale_stats = {
			name: np.nan for name in ("mean", "median", "p90", "p95", "p99", "min")
		}
	active_count = int(np.sum(damping_scale < 1.0))
	metadata = {
		"attraction_degree_damping_enabled": int(bool(enabled)),
		"attraction_degree_damping_mode": str(mode),
		"attraction_degree_damping_ref": str(ref),
		"attraction_degree_damping_degree_ref": float(degree_ref),
		"attraction_degree_damping_power": float(power),
		"attraction_degree_damping_min_scale": (
			np.nan if min_scale is None else float(min_scale)
		),
		"attraction_degree_damping_active_count": active_count,
		"attraction_degree_damping_active_pct": (
			100.0 * active_count / float(max(damping_scale.size, 1))
		),
	}
	metadata.update({
		f"attraction_degree_damping_degree_{name}": float(value)
		for name, value in degree_stats.items()
	})
	metadata.update({
		f"attraction_degree_damping_scale_{name}": float(value)
		for name, value in scale_stats.items()
	})
	return metadata


def _degree_percentiles(degrees):
	degree_values = _to_numpy_array(degrees).astype(np.float64, copy=False)
	if degree_values.size == 0:
		return degree_values
	order = np.argsort(degree_values, kind="mergesort")
	percentiles = np.empty(degree_values.shape[0], dtype=np.float64)
	if degree_values.shape[0] == 1:
		percentiles[order] = 100.0
	else:
		percentiles[order] = 100.0 * np.arange(degree_values.shape[0], dtype=np.float64) / float(degree_values.shape[0] - 1)
	return percentiles


def _topk_fieldnames():
	return [
		"epoch",
		"rank",
		"criterion",
		"alpha",
		"repulsion_clip_norm",
		"point_id",
		"attr_norm",
		"repl_pre_clip_norm",
		"repl_post_clip_norm",
		"total_update_norm",
		"total_pre_clip_norm",
		"total_post_clip_norm",
		"attr_x",
		"attr_y",
		"repl_x",
		"repl_y",
		"total_x",
		"total_y",
		"x",
		"y",
		"radius",
		"previous_radius",
		"delta_radius",
		"cos_attr_repl",
		"cos_total_radial_direction",
		"degree",
		"degree_percentile",
		"graph_degree",
		"graph_degree_percentile",
		"attraction_degree_damping_scale",
		"label",
	]


def _open_topk_writer(path, topk):
	if not path or int(topk) <= 0:
		return None, None
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	handle = output_path.open("w", newline="")
	writer = csv.DictWriter(handle, fieldnames=_topk_fieldnames())
	writer.writeheader()
	return handle, writer


def _take_rows_numpy(array, indices):
	if cupy is not None and isinstance(array, cupy.ndarray):
		return cupy.asnumpy(array[indices])
	return np.asarray(array)[indices]


def _write_topk_rows(
	writer,
	epoch_itr,
	alpha_value,
	attr_force,
	repl_force,
	repl_force_pre_clip_norms,
	total_force_pre_clip,
	total_force_post_clip,
	embedding,
	previous_radii,
	degrees,
	degree_percentiles,
	labels,
	point_ids,
	topk,
	clip_norm,
	is_gpu,
	attraction_degree_damping_scale=None,
):
	attr_norms = _force_norms_numpy(attr_force, is_gpu)
	repl_norms = _force_norms_numpy(repl_force, is_gpu)
	total_force = total_force_pre_clip if total_force_pre_clip is not None else attr_force + repl_force
	total_norms = _force_norms_numpy(total_force, is_gpu)
	total_pre_rows_all = _to_numpy_array(total_force)
	total_pre_norms = np.linalg.norm(total_pre_rows_all, axis=1)
	total_post_rows_all = _to_numpy_array(total_force_post_clip) if total_force_post_clip is not None else total_pre_rows_all
	total_post_norms = np.linalg.norm(total_post_rows_all, axis=1)
	if repl_force_pre_clip_norms is None:
		repl_force_pre_clip_norms = repl_norms

	n = int(total_norms.size)
	if n == 0:
		return
	k = min(max(int(topk), 0), n)
	if k == 0:
		return
	embedding_values = _to_numpy_array(embedding).astype(np.float64, copy=False)
	center = np.median(embedding_values[:, :2], axis=0)
	radial_vectors = embedding_values[:, :2] - center
	radii = np.linalg.norm(radial_vectors, axis=1)
	if previous_radii is None or len(previous_radii) != n:
		previous_radii = np.full(n, np.nan, dtype=np.float64)
	delta_radii = radii - previous_radii
	degree_values = _to_numpy_array(degrees)
	if attraction_degree_damping_scale is None:
		damping_scale_values = np.ones(n, dtype=np.float32)
	else:
		damping_scale_values = _to_numpy_array(attraction_degree_damping_scale)
	criteria = (
		("total_update_norm", total_norms),
		("attr_norm", attr_norms),
		("repl_post_clip_norm", repl_norms),
		("radius", radii),
		("delta_radius", np.nan_to_num(delta_radii, nan=-np.inf)),
	)
	attr_values = _to_numpy_array(attr_force)
	repl_values = _to_numpy_array(repl_force)
	total_values = total_pre_rows_all

	for criterion, scores in criteria:
		if k == n:
			top_indices = np.arange(n)
		else:
			top_indices = np.argpartition(-scores, k - 1)[:k]
		top_indices = top_indices[np.argsort(-scores[top_indices])]
		for rank, local_idx in enumerate(top_indices, start=1):
			local_idx = int(local_idx)
			point_id = point_ids[local_idx] if point_ids is not None and local_idx < len(point_ids) else local_idx
			label = labels[local_idx] if labels is not None and local_idx < len(labels) else ""
			attr_row = attr_values[local_idx]
			repl_row = repl_values[local_idx]
			total_row = total_values[local_idx]
			radial_row = radial_vectors[local_idx]
			cos_attr_repl = float(np.dot(attr_row, repl_row) / max(attr_norms[local_idx] * repl_norms[local_idx], 1e-12))
			cos_total_radial = float(np.dot(total_row, radial_row) / max(total_norms[local_idx] * radii[local_idx], 1e-12))
			writer.writerow({
			"epoch": int(epoch_itr),
			"rank": int(rank),
			"criterion": criterion,
			"alpha": float(alpha_value),
			"repulsion_clip_norm": np.nan if clip_norm is None else float(clip_norm),
			"point_id": point_id,
			"attr_norm": float(attr_norms[local_idx]),
			"repl_pre_clip_norm": float(repl_force_pre_clip_norms[local_idx]),
			"repl_post_clip_norm": float(repl_norms[local_idx]),
			"total_update_norm": float(total_norms[local_idx]),
			"total_pre_clip_norm": float(total_pre_norms[local_idx]),
			"total_post_clip_norm": float(total_post_norms[local_idx]),
			"attr_x": float(attr_row[0]),
			"attr_y": float(attr_row[1]),
			"repl_x": float(repl_row[0]),
			"repl_y": float(repl_row[1]),
			"total_x": float(total_row[0]),
			"total_y": float(total_row[1]),
			"x": float(embedding_values[local_idx, 0]),
			"y": float(embedding_values[local_idx, 1]),
			"radius": float(radii[local_idx]),
			"previous_radius": float(previous_radii[local_idx]),
			"delta_radius": float(delta_radii[local_idx]),
			"cos_attr_repl": cos_attr_repl,
			"cos_total_radial_direction": cos_total_radial,
			"degree": float(degree_values[local_idx]),
			"degree_percentile": float(degree_percentiles[local_idx]),
			"graph_degree": float(degree_values[local_idx]),
			"graph_degree_percentile": float(degree_percentiles[local_idx]),
			"attraction_degree_damping_scale": float(damping_scale_values[local_idx]),
			"label": label,
			})
	return radii


def _write_diagnostic_row(
	writer,
	epoch_itr,
	alpha_value,
	attr_force,
	repl_force,
	repl_force_pre_clip_norms,
	total_update_pre_clip_norms,
	total_force_post_clip,
	embedding,
	thresholds,
	repulsion_clip_norm,
	total_update_clip_norm,
	local_exact_diagnostics,
	local_density_diagnostics,
	ibfft_kernel_clip,
	umap_epsilon,
	ibfft_kernel_subsample_mode,
	ibfft_kernel_subsample_radius_cells,
	ibfft_kernel_subsample_points,
	grid_context,
	n_interpolation_points,
	attr_norm_pre_degree_damping,
	attr_norm_post_degree_damping,
	attraction_degree_damping_metadata,
	is_gpu,
):
	attr_norms = _force_norms_numpy(attr_force, is_gpu)
	repl_norms = _force_norms_numpy(repl_force, is_gpu)
	total_norms = _force_norms_numpy(total_force_post_clip, is_gpu)
	if repl_force_pre_clip_norms is None:
		repl_force_pre_clip_norms = repl_norms
	if total_update_pre_clip_norms is None:
		total_update_pre_clip_norms = total_norms
	row = {
		"epoch": int(epoch_itr),
		"alpha": float(alpha_value),
		"ibfft_kernel_clip": float(ibfft_kernel_clip),
		"umap_epsilon": float(umap_epsilon),
		"ibfft_kernel_subsample_mode": (
			"" if ibfft_kernel_subsample_mode is None else str(ibfft_kernel_subsample_mode)
		),
		"ibfft_kernel_subsample_radius_cells": int(ibfft_kernel_subsample_radius_cells),
		"ibfft_kernel_subsample_points": int(ibfft_kernel_subsample_points),
		"repulsion_clip_enabled_this_epoch": int(repulsion_clip_norm is not None),
		"repulsion_clip_effective_norm": np.nan if repulsion_clip_norm is None else float(repulsion_clip_norm),
		"repulsion_clip_norm": np.nan if repulsion_clip_norm is None else float(repulsion_clip_norm),
		"repulsion_clip_active_count": 0,
		"repulsion_clip_active_pct": 0.0,
		"total_update_clip_enabled_this_epoch": int(total_update_clip_norm is not None),
		"total_update_clip_effective_norm": np.nan if total_update_clip_norm is None else float(total_update_clip_norm),
		"total_update_clip_norm": np.nan if total_update_clip_norm is None else float(total_update_clip_norm),
		"total_update_clip_active_count": 0,
		"total_update_clip_active_pct": 0.0,
	}
	row.update(attraction_degree_damping_metadata)
	row.update(local_exact_diagnostics)
	row.update(local_density_diagnostics)
	if repulsion_clip_norm is not None:
		active = int(np.sum(repl_force_pre_clip_norms > float(repulsion_clip_norm)))
		row["repulsion_clip_active_count"] = active
		row["repulsion_clip_active_pct"] = 100.0 * float(active) / float(max(repl_force_pre_clip_norms.size, 1))
	if total_update_clip_norm is not None:
		active = int(np.sum(total_update_pre_clip_norms > float(total_update_clip_norm)))
		row["total_update_clip_active_count"] = active
		row["total_update_clip_active_pct"] = 100.0 * float(active) / float(max(total_update_pre_clip_norms.size, 1))
	row.update(_norm_stats("attr_force", attr_norms))
	row.update(_norm_stats("attr_norm_pre_degree_damping", attr_norm_pre_degree_damping))
	row.update(_norm_stats("attr_norm_post_degree_damping", attr_norm_post_degree_damping))
	row.update(_norm_stats("repl_force_pre_clip", repl_force_pre_clip_norms))
	row.update(_norm_stats("repl_force", repl_norms))
	row.update(_norm_stats("total_update_pre_clip", total_update_pre_clip_norms))
	row.update(_norm_stats("total_update", total_norms))
	row.update(_embedding_bbox(embedding, is_gpu))
	if grid_context is None:
		row.update({
			"grid_x_min": np.nan, "grid_x_max": np.nan,
			"grid_y_min": np.nan, "grid_y_max": np.nan,
			"grid_cell_size": np.nan, "grid_h": np.nan,
			"n_boxes_per_dim": np.nan,
		})
	else:
		row.update({
			"grid_x_min": float(grid_context.min_xy[0]),
			"grid_x_max": float(grid_context.max_xy[0]),
			"grid_y_min": float(grid_context.min_xy[1]),
			"grid_y_max": float(grid_context.max_xy[1]),
			"grid_cell_size": float(grid_context.cell_size),
			"grid_h": float(grid_context.cell_size) / float(n_interpolation_points),
			"n_boxes_per_dim": int(grid_context.n_cells_x),
		})
	_add_threshold_counts(row, "repl_pre_clip", repl_force_pre_clip_norms, thresholds)
	_add_threshold_counts(row, "repl", repl_norms, thresholds)
	_add_threshold_counts(row, "total_update_pre_clip", total_update_pre_clip_norms, thresholds)
	_add_threshold_counts(row, "total_update", total_norms, thresholds)
	writer.writerow(row)


""" Controller """

def _resolve_attraction_schedule_mode(mode):
	if mode not in ("row_scan", "active_edge_calendar", "active_edge_periodic", "auto"):
		raise ValueError(
			"attraction_schedule_mode must be one of: row_scan, "
			"active_edge_calendar, active_edge_periodic, auto"
		)
	if mode == "auto":
		return "row_scan"
	return mode


def _resolve_attraction_kernel_mode(mode, *, is_gpu, average_degree, warp_min_degree):
	if mode not in ("thread_per_row", "warp_per_row", "auto"):
		raise ValueError(
			"attraction_kernel_mode must be one of: thread_per_row, "
			"warp_per_row, auto"
		)
	if mode == "auto":
		return (
			"warp_per_row"
			if is_gpu and average_degree >= float(warp_min_degree)
			else "thread_per_row"
		)
	if mode == "warp_per_row" and not is_gpu:
		raise NotImplementedError(
			"attraction_kernel_mode='warp_per_row' is CUDA-only"
		)
	return mode


def _warp_per_row_launch(n_vertices, *, threads_per_block=256):
	threads_per_block = int(threads_per_block)
	if threads_per_block < 32 or threads_per_block % 32 != 0:
		raise ValueError("threads_per_block must be a positive multiple of 32")
	warps_per_block = threads_per_block // 32
	blocks = max(1, (int(n_vertices) + warps_per_block - 1) // warps_per_block)
	return (blocks,), (threads_per_block,)


def umap_true_loss_optimization(
	embedding,
	edgesrc,
	edgetgt,
	weights,
	degrees,
	n_epochs,
	n_vertices,
	n_components,
	epochs_per_sample,
	negative_sample_rate,
	attraction_mode,
	repulsion_mode,
	device,
	drfft_params,
):
	if device not in ("cpu", "cuda"):
		raise ValueError(f"Unsupported device for ibumap optimization: {device}")
	if attraction_mode not in ("sampling", "true_loss"):
		raise ValueError(f"Unsupported attraction_mode: {attraction_mode}")
	if repulsion_mode not in ("sampling", "true_loss"):
		raise ValueError(f"Unsupported repulsion_mode: {repulsion_mode}")
	if degrees is None:
		raise ValueError("degrees must be provided for ibumap optimization")

	memory_recorder = drfft_params.get("memory_recorder")
	if memory_recorder is not None:
		memory_recorder.record("optimizer_prepare", "begin")

	is_gpu = device == "cuda"
	numeric_dtype = np.dtype(drfft_params.get("numeric_dtype", np.float32))
	if numeric_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
		raise ValueError("numeric_dtype must be float32 or float64")
	resolved_fft_stages = resolve_fft_schedule(
		n_epochs=int(n_epochs),
		n_interpolation_points=drfft_params['n_interpolation_points'],
		combine_stages=drfft_params.get('tfdp_combine', False),
		interpolation_schedule=drfft_params.get('interpolation_schedule'),
	)
	validate_fft_schedule_execution(
		resolved_fft_stages,
		device=device,
		p2m_mode=drfft_params.get('p2m_mode', 'auto'),
	)
	n_vertices_int = int(n_vertices)
	edge_count_int = int(edgetgt.shape[0])
	attraction_schedule_mode = _resolve_attraction_schedule_mode(
		drfft_params.get("attraction_schedule_mode", "row_scan")
	)
	attraction_warp_min_degree = int(
		drfft_params.get("attraction_warp_min_degree", 8)
	)
	attraction_kernel_mode = _resolve_attraction_kernel_mode(
		drfft_params.get("attraction_kernel_mode", "auto"),
		is_gpu=is_gpu,
		average_degree=(edge_count_int / float(max(n_vertices_int, 1))),
		warp_min_degree=attraction_warp_min_degree,
	)
	if is_gpu and cupy is None:
		raise RuntimeError("GPU optimization requested but cupy/CUDA kernels are unavailable")
	if is_gpu and numeric_dtype == np.dtype(np.float64):
		raise NotImplementedError(
			"dtype='float64' is currently supported on CPU only; CUDA kernels use float32"
		)
	if is_gpu:
		nonfinite_count = int(
			cupy.count_nonzero(~cupy.isfinite(embedding)).item()
		)
	else:
		nonfinite_count = int(
			np.count_nonzero(~np.isfinite(np.asarray(embedding)))
		)
	if nonfinite_count:
		raise FloatingPointError(
			"ibUMAP optimizer input embedding contains "
			f"{nonfinite_count} non-finite value(s)"
		)
	kernel_subsample_mode = drfft_params.get("ibfft_kernel_subsample_mode")
	kernel_subsample_radius_cells = int(
		drfft_params.get("ibfft_kernel_subsample_radius_cells", 2)
	)
	kernel_subsample_points = int(drfft_params.get("ibfft_kernel_subsample_points", 4))
	if is_gpu and kernel_subsample_mode is not None:
		raise NotImplementedError(
			"ibFFT kernel sub-sampling currently supports CPU only"
		)
	local_exact_enabled = bool(drfft_params.get("local_exact_repulsion", False))
	local_density_enabled = bool(drfft_params.get("local_density_pressure", False))
	if is_gpu and (local_exact_enabled or local_density_enabled):
		raise NotImplementedError(
			"local_exact_repulsion and local_density_pressure currently support CPU only"
		)
	local_exact_every = int(drfft_params.get("local_exact_every", 1))
	local_density_every = int(drfft_params.get("local_density_pressure_every", 1))
	local_exact_start_epoch, local_exact_end_epoch = _resolve_local_exact_epoch_bounds(
		drfft_params, n_epochs
	)

	t1 = time()
	initial_alpha = drfft_params['umap_initial_alpha']
	alpha = initial_alpha
	epoch_of_next_sample = epochs_per_sample.copy()

	### Positive and Negative Sampling Effect
	pos_effects = weights
	neg_effect, neg_effects, probabilities = 0.0, None, None
	if repulsion_mode == 'true_loss':
		neg_effects = degrees * 0.5 * negative_sample_rate / n_vertices
	else:
		neg_effect = 10 * negative_sample_rate / n_vertices
		probabilities = degrees / degrees.max()

	### Constraints
	whether_known_points		= drfft_params['whether_known_points']
	known_points_positions		= drfft_params['known_points_positions']
	known_points_reverse_index	= drfft_params['known_points_reverse_index']
	constraint_weight			= drfft_params['constraint_weight']
	soft_constraint				= drfft_params['soft_constraint']
	hard_constraint				= (True in whether_known_points) and (not soft_constraint)
	noise_mode = drfft_params.get('noise_mode', 'none')
	noise_rng = None
	sampling_rng = _make_sampling_rng(
		is_gpu, drfft_params.get("sampling_random_state")
	)
	if attraction_schedule_mode in ("active_edge_calendar", "active_edge_periodic"):
		if is_gpu:
			raise NotImplementedError(
				f"attraction_schedule_mode={attraction_schedule_mode!r} currently "
				"supports CPU only"
			)
		if attraction_mode != "sampling":
			raise NotImplementedError(
				"active-edge attraction schedules currently support "
				"attraction_mode='sampling' only"
			)
		if soft_constraint:
			raise NotImplementedError(
				"active-edge attraction schedules do not yet support soft constraints"
			)

	if not is_gpu:
		embedding = np.asarray(embedding, dtype=numeric_dtype, order="C")
		weights = np.asarray(weights, dtype=numeric_dtype)
		degrees = np.asarray(degrees, dtype=numeric_dtype)
		pos_effects = np.asarray(pos_effects, dtype=numeric_dtype)
		if neg_effects is not None:
			neg_effects = np.asarray(neg_effects, dtype=numeric_dtype)
		if probabilities is not None:
			probabilities = np.asarray(probabilities, dtype=numeric_dtype)
		cpu_workspace = drfft_params.get('optimizer_workspace')
		if cpu_workspace is None:
			attr_force = np.zeros((n_vertices, n_components), dtype=numeric_dtype)
			_ibfft_ws = {}
		else:
			attr_force = cpu_workspace.get_or_alloc(
				'optimizer.attr_force',
				(n_vertices, n_components),
				dtype=numeric_dtype,
			)
			_ibfft_ws = cpu_workspace.buffers
		# H-2 FIX: Shared mutable workspace dict passed to ibFFT_repulsive_sampling.
		# ibFFT will lazily allocate and cache arrays in this dict on first call
		# and reuse them on subsequent calls, eliminating per-epoch heap allocation.
		if local_exact_enabled:
			local_edge_sources, local_edge_targets = _local_edge_sample(edgesrc, edgetgt)
		else:
			local_edge_sources = np.empty(0, dtype=np.int64)
			local_edge_targets = np.empty(0, dtype=np.int64)
		attraction_calendar_build_time = 0.0
		attraction_calendar_build_count_time = 0.0
		attraction_calendar_build_fill_time = 0.0
		attraction_calendar_build_sort_time = 0.0
		attraction_calendar_build_group_time = 0.0
		attraction_calendar_build_schedule_time = 0.0
		attraction_calendar_init_time = 0.0
		attraction_calendar_event_count = 0
		attraction_calendar_active_rows = 0
		attraction_calendar_estimated_bytes = 0
		attraction_calendar_bucket_count = 0
		attraction_calendar_due_pattern_count = 0
		if attraction_schedule_mode == "active_edge_calendar":
			calendar_t0 = perf_counter()
			(
				attraction_calendar_epoch_row_offsets,
				attraction_calendar_rows,
				attraction_calendar_edge_offsets,
				attraction_calendar_edges,
				attraction_calendar_stats,
			) = _build_sampling_attraction_calendar(
				edgesrc,
				epochs_per_sample,
				n_epochs,
				drfft_params.get("attraction_calendar_memory_limit_bytes"),
			)
			init_t0 = perf_counter()
			attr_force.fill(0.0)
			attraction_calendar_init_time = perf_counter() - init_t0
			attraction_calendar_build_time = perf_counter() - calendar_t0
			attraction_calendar_event_count = int(attraction_calendar_stats["events"])
			attraction_calendar_active_rows = int(attraction_calendar_stats["active_rows"])
			attraction_calendar_estimated_bytes = int(
				attraction_calendar_stats["estimated_bytes"]
			)
			attraction_calendar_build_count_time = float(
				attraction_calendar_stats.get("build_count_time", 0.0)
			)
			attraction_calendar_build_fill_time = float(
				attraction_calendar_stats.get("build_fill_time", 0.0)
			)
			attraction_calendar_build_sort_time = float(
				attraction_calendar_stats.get("build_sort_time", 0.0)
			)
			attraction_calendar_build_group_time = float(
				attraction_calendar_stats.get("build_group_time", 0.0)
			)
			attraction_calendar_build_schedule_time = float(
				attraction_calendar_stats.get("build_schedule_time", 0.0)
			)
			attraction_calendar_bucket_count = int(
				attraction_calendar_stats.get("build_bucket_count", 0)
			)
			attraction_calendar_due_pattern_count = int(
				attraction_calendar_stats.get("build_due_pattern_count", 0)
			)
			attraction_periodic_bucket_offsets = np.zeros(1, dtype=np.int64)
			attraction_periodic_edges = np.empty(0, dtype=np.int32)
			attraction_periodic_edge_sources = np.empty(0, dtype=np.int32)
			attraction_periodic_due_epochs = np.zeros((0, int(n_epochs)), dtype=np.uint8)
			attraction_periodic_row_epoch_marks = np.zeros(n_vertices_int, dtype=np.int32)
		elif attraction_schedule_mode == "active_edge_periodic":
			calendar_t0 = perf_counter()
			(
				attraction_periodic_bucket_offsets,
				attraction_periodic_edges,
				attraction_periodic_edge_sources,
				attraction_periodic_due_epochs,
				attraction_calendar_stats,
			) = _build_sampling_attraction_periodic(
				edgesrc,
				epochs_per_sample,
				n_epochs,
				drfft_params.get("attraction_calendar_memory_limit_bytes"),
			)
			attraction_periodic_row_epoch_marks = np.zeros(n_vertices_int, dtype=np.int32)
			attraction_calendar_build_time = perf_counter() - calendar_t0
			attraction_calendar_event_count = int(attraction_calendar_stats["events"])
			attraction_calendar_active_rows = int(attraction_calendar_stats["active_rows"])
			attraction_calendar_estimated_bytes = int(
				attraction_calendar_stats["estimated_bytes"]
			)
			attraction_calendar_build_count_time = float(
				attraction_calendar_stats.get("build_count_time", 0.0)
			)
			attraction_calendar_build_fill_time = float(
				attraction_calendar_stats.get("build_fill_time", 0.0)
			)
			attraction_calendar_build_sort_time = float(
				attraction_calendar_stats.get("build_sort_time", 0.0)
			)
			attraction_calendar_build_group_time = float(
				attraction_calendar_stats.get("build_group_time", 0.0)
			)
			attraction_calendar_build_schedule_time = float(
				attraction_calendar_stats.get("build_schedule_time", 0.0)
			)
			attraction_calendar_bucket_count = int(
				attraction_calendar_stats.get("build_bucket_count", 0)
			)
			attraction_calendar_due_pattern_count = int(
				attraction_calendar_stats.get("build_due_pattern_count", 0)
			)
			attraction_calendar_epoch_row_offsets = np.zeros(1, dtype=np.int64)
			attraction_calendar_rows = np.empty(0, dtype=np.int32)
			attraction_calendar_edge_offsets = np.zeros(1, dtype=np.int64)
			attraction_calendar_edges = np.empty(0, dtype=np.int32)
		else:
			attraction_calendar_epoch_row_offsets = np.zeros(1, dtype=np.int64)
			attraction_calendar_rows = np.empty(0, dtype=np.int32)
			attraction_calendar_edge_offsets = np.zeros(1, dtype=np.int64)
			attraction_calendar_edges = np.empty(0, dtype=np.int32)
			attraction_periodic_bucket_offsets = np.zeros(1, dtype=np.int64)
			attraction_periodic_edges = np.empty(0, dtype=np.int32)
			attraction_periodic_edge_sources = np.empty(0, dtype=np.int32)
			attraction_periodic_due_epochs = np.zeros((0, int(n_epochs)), dtype=np.uint8)
			attraction_periodic_row_epoch_marks = np.zeros(n_vertices_int, dtype=np.int32)
	else:
		# GPU attraction kernels do not consume constraint arrays. Transfer the
		# mask only for hard-constraint post-kernel zeroing.
		gpu_workspace = drfft_params.get('optimizer_workspace')
		whether_known_points = (
			cupy.asarray(whether_known_points, dtype=cupy.bool_)
			if hard_constraint
			else cupy.empty(0, dtype=cupy.bool_)
		)
		if gpu_workspace is None:
			attr_force = cupy.empty((n_vertices_int, n_components), dtype=cupy.float32)
		else:
			attr_force = gpu_workspace.get_or_alloc(
				'optimizer.attr_force',
				(n_vertices_int, n_components),
				dtype=cupy.float32,
			)
		optimizer_grid_dim, optimizer_block_dim = point_launch(n_vertices_int)
		n_vertices = cupy.int32(n_vertices)
		epochs_per_sample = cupy.asarray(epochs_per_sample, dtype=cupy.float32)
		epoch_of_next_sample = cupy.asarray(epoch_of_next_sample, dtype=cupy.float32)
		alpha = cupy.float32(alpha)
		embedding = cupy.asarray(embedding, dtype=cupy.float32)
		edgesrc = cupy.asarray(edgesrc, dtype=cupy.int32)
		edgetgt = cupy.asarray(edgetgt, dtype=cupy.int32)
		pos_effects = cupy.asarray(pos_effects, dtype=cupy.float32)
		neg_effect = cupy.float32(neg_effect)
		if neg_effects is not None: neg_effects = cupy.asarray(neg_effects, dtype=cupy.float32)
		if probabilities is not None: probabilities = cupy.asarray(probabilities, dtype=cupy.float32)
		if attraction_kernel_mode == "warp_per_row":
			attraction_warp_grid_dim, attraction_warp_block_dim = _warp_per_row_launch(
				n_vertices_int
			)
		else:
			attraction_warp_grid_dim = attraction_warp_block_dim = None
		attraction_calendar_build_time = 0.0
		attraction_calendar_build_count_time = 0.0
		attraction_calendar_build_fill_time = 0.0
		attraction_calendar_build_sort_time = 0.0
		attraction_calendar_build_group_time = 0.0
		attraction_calendar_build_schedule_time = 0.0
		attraction_calendar_init_time = 0.0
		attraction_calendar_event_count = 0
		attraction_calendar_active_rows = 0
		attraction_calendar_estimated_bytes = 0
		attraction_calendar_bucket_count = 0
		attraction_calendar_due_pattern_count = 0
		attraction_calendar_epoch_row_offsets = None
		attraction_calendar_rows = None
		attraction_calendar_edge_offsets = None
		attraction_calendar_edges = None
		attraction_periodic_bucket_offsets = None
		attraction_periodic_edges = None
		attraction_periodic_edge_sources = None
		attraction_periodic_due_epochs = None
		attraction_periodic_row_epoch_marks = None

	# Experimental synchronous ibUMAP safeguard: high-degree points can collect
	# very large source-row attraction totals. Compute the static scale once and
	# apply it only after each epoch's edge-level attraction has accumulated.
	damping_enabled = bool(drfft_params.get("attraction_degree_damping", False))
	damping_mode = drfft_params.get(
		"attraction_degree_damping_mode", "weighted_degree"
	)
	damping_ref_type = drfft_params.get("attraction_degree_damping_ref", "p99")
	damping_ref_value = drfft_params.get("attraction_degree_damping_ref_value")
	damping_power = float(drfft_params.get("attraction_degree_damping_power", 0.5))
	damping_min_scale = drfft_params.get("attraction_degree_damping_min_scale")
	damping_diagnostics_requested = bool(
		drfft_params.get("diagnostics_path")
		or (
			drfft_params.get("diagnostics_topk_path")
			and int(drfft_params.get("diagnostics_topk") or 0) > 0
		)
	)
	if damping_enabled or damping_diagnostics_requested:
		(
			damping_degree_values,
			damping_degree_ref,
			attraction_degree_damping_scale,
		) = _compute_attraction_degree_damping_scale(
			edgesrc,
			pos_effects,
			n_vertices_int,
			mode=damping_mode,
			ref=damping_ref_type,
			ref_value=damping_ref_value,
			power=damping_power,
			min_scale=damping_min_scale,
			enabled=damping_enabled,
			is_gpu=is_gpu,
			weighted_degrees=degrees,
		)
	else:
		damping_degree_values = degrees
		damping_degree_ref = np.nan
		attraction_degree_damping_scale = None
	if drfft_params.get("diagnostics_path"):
		attraction_degree_damping_metadata = _attraction_degree_damping_metadata(
			damping_enabled,
			damping_mode,
			damping_ref_type,
			damping_degree_ref,
			damping_power,
			damping_min_scale,
			damping_degree_values,
			attraction_degree_damping_scale,
		)
	else:
		attraction_degree_damping_metadata = None

	if noise_mode != 'none' and float(drfft_params.get('noise_scale', 0.0) or 0.0) > 0.0:
		noise_rng = _make_noise_rng(is_gpu, _resolve_noise_seed(drfft_params))
	if drfft_params.get("diagnostics_timing_path") and local_exact_enabled:
		_dummy_embedding = np.zeros((1, 2), dtype=numeric_dtype)
		_dummy_int = np.zeros(1, dtype=np.int64)
		_dummy_cell = np.zeros(1, dtype=np.int32)
		_local_exact_repulsion_force_profiled(
			_dummy_embedding,
			_dummy_cell,
			_dummy_cell,
			_dummy_int,
			_dummy_int,
			np.ones(1, dtype=np.int64),
			1,
			1,
			1,
			1.0,
			1.0,
			1.0,
			1.0,
			0.001,
			1.0,
			4.0,
			True,
			np.ones(1, dtype=np.int64),
			_dummy_int,
			False,
			0.0,
			False,
			1,
		)

	attr_time = []
	repl_time = []
	p2m_time = []
	ibfft_timing_names = (
		'setup_time_s',
		'kernel_build_time_s',
		'kernel_fft_time_s',
		'point_setup_time_s',
		'mesh_clear_time_s',
		'p2m_time_s',
		'fft_plan_build_time_s',
		'fft_forward_time_s',
		'fft_multiply_time_s',
		'fft_inverse_time_s',
		'm2p_time_s',
		'sampling_restore_time_s',
		'ibfft_other_time_s',
	)
	ibfft_timing_totals = {name: 0.0 for name in ibfft_timing_names}
	p2m_resolved_modes = set()
	fft_kernel_cache_hits = 0
	fft_kernel_cache_misses = 0
	fft_kernel_cache_compulsory_misses = 0
	fft_kernel_cache_eviction_misses = 0
	fft_kernel_cache_evictions = 0
	fft_kernel_cache_evicted_bytes = 0
	fft_kernel_cache_bytes = 0
	fft_kernel_cache_peak_bytes = 0
	fft_kernel_cache_entries = 0
	fft_kernel_cache_peak_entries = 0
	fft_kernel_cache_limit_bytes = None
	fft_kernel_cache_max_entries = None
	fft_kernel_cache_policy = None
	fft_kernel_cache_scope = None
	fft_kernel_cache_store_oversized = 0
	fft_kernel_cache_exact_keys = set()
	fft_plan_cache_hits = 0
	fft_plan_cache_misses = 0
	fft_grid_quantization_levels = set()
	fft_stage_epoch_indices = []
	fused_m2p_update_epochs = 0
	attraction_warp_per_row_epochs = 0
	attraction_thread_per_row_epochs = 0
	attraction_calendar_epochs = 0
	attraction_calendar_execute_time = 0.0
	attraction_calendar_execute_zero_time = 0.0
	attraction_calendar_execute_kernel_time = 0.0
	attraction_calendar_execute_event_count = 0
	attraction_calendar_execute_active_rows = 0
	appl_time = []
	cuda_attr_events = []
	cuda_repl_events = []
	cuda_p2m_events = []
	cuda_ibfft_timing_events = {}
	cuda_appl_events = []
	opt_prep_time = time() - t1
	if memory_recorder is not None:
		memory_recorder.record(
			"optimizer_prepare",
			"end",
			metadata={"n_vertices": int(n_vertices_int), "n_epochs": int(n_epochs)},
		)

	def _cuda_stage_start():
		event = cupy.cuda.Event()
		event.record()
		return event

	def _cuda_stage_end(start_event, event_pairs):
		end_event = cupy.cuda.Event()
		end_event.record()
		event_pairs.append((start_event, end_event))

	diagnostic_path = drfft_params.get("diagnostics_path")
	if diagnostic_path:
		diagnostic_thresholds = drfft_params.get("diagnostics_thresholds") or (
			1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0
		)
		diagnostic_thresholds = tuple(float(value) for value in diagnostic_thresholds)
		diagnostic_file, diagnostic_writer = _open_diagnostic_writer(
			diagnostic_path, diagnostic_thresholds
		)
	else:
		diagnostic_thresholds = None
		diagnostic_file = diagnostic_writer = None
	runtime_diagnostic_path = drfft_params.get("diagnostics_timing_path")
	if runtime_diagnostic_path:
		runtime_diagnostic_file, runtime_diagnostic_writer = (
			_open_runtime_diagnostic_writer(runtime_diagnostic_path)
		)
	else:
		runtime_diagnostic_file = runtime_diagnostic_writer = None
	diagnostic_topk = int(drfft_params.get("diagnostics_topk") or 0)
	topk_path = drfft_params.get("diagnostics_topk_path")
	if topk_path and diagnostic_topk > 0:
		topk_file, topk_writer = _open_topk_writer(topk_path, diagnostic_topk)
		diagnostic_labels = drfft_params.get("diagnostics_labels")
		diagnostic_point_ids = drfft_params.get("diagnostics_point_ids")
	else:
		topk_file = topk_writer = None
		diagnostic_labels = diagnostic_point_ids = None
	degree_percentiles = _degree_percentiles(degrees) if topk_writer is not None else None
	previous_diagnostic_radii = None
	drfft_params["repulsion_clip_epoch_range"] = _validate_clip_epoch_range_for_epochs(
		drfft_params.get("repulsion_clip_epoch_range"),
		n_epochs,
		"repulsion_clip_epoch_range",
	)
	drfft_params["total_update_clip_epoch_range"] = _validate_clip_epoch_range_for_epochs(
		drfft_params.get("total_update_clip_epoch_range"),
		n_epochs,
		"total_update_clip_epoch_range",
	)
	fused_m2p_update_base = (
		repulsion_mode == 'true_loss'
		and not local_exact_enabled
		and not local_density_enabled
		and not hard_constraint
		and not soft_constraint
		and diagnostic_writer is None
		and runtime_diagnostic_writer is None
		and topk_writer is None
		and memory_recorder is None
		and kernel_subsample_mode is None
		and drfft_params.get("total_update_clip_norm") is None
		and not (
			noise_mode != 'none'
			and float(drfft_params.get('noise_scale', 0.0) or 0.0) > 0.0
		)
	)

	### Embedding optimization
	for epoch_itr in range(n_epochs):
		record_epoch_memory = (
			memory_recorder is not None
			and memory_recorder.should_record_epoch(int(epoch_itr))
		)
		epoch_start_time = (
			perf_counter() if runtime_diagnostic_writer is not None else 0.0
		)
		ibfft_epoch_time = 0.0
		local_grid_time = 0.0
		local_exact_total_time = 0.0
		local_prep_time = 0.0
		local_stage_times = None
		local_outer_accum_time = 0.0
		local_array_copy_time = 0.0
		local_density_threshold = 0.0
		local_candidate_counts = None
		local_distance_counts = None
		local_selected_counts = None
		local_eligible = None
		if runtime_diagnostic_writer is not None:
			local_stage_times = np.zeros(5, dtype=np.float64)
			local_candidate_counts = np.empty(0, dtype=np.int64)
			local_distance_counts = np.empty(0, dtype=np.int64)
			local_selected_counts = np.empty(0, dtype=np.int64)
			local_eligible = np.empty(0, dtype=np.uint8)
		local_timing_sample_count = 0
		local_exact_norms = None
		grid_context = None
		repl_force_pre_clip_norms = None
		total_update_pre_clip_norms = None
		attr_norm_pre_degree_damping = None
		attr_norm_post_degree_damping = None
		local_exact_this_epoch = _local_enabled_for_epoch(
			local_exact_enabled,
			local_exact_every,
			epoch_itr,
			local_exact_start_epoch,
			local_exact_end_epoch,
		)
		local_density_this_epoch = _local_enabled_for_epoch(
			local_density_enabled, local_density_every, epoch_itr
		)
		repulsion_clip_norm = _resolve_repulsion_clip_norm(
			drfft_params,
			alpha,
			epoch_itr,
		)
		total_update_clip_norm = _resolve_total_update_clip_norm(
			drfft_params,
			alpha,
			epoch_itr,
		)
		noise_scale = _noise_scale_for_epoch(drfft_params, epoch_itr, n_epochs)
		fused_m2p_update_this_epoch = (
			fused_m2p_update_base
			and total_update_clip_norm is None
			and noise_scale <= 0.0
		)
		need_grid_context = (
			local_exact_this_epoch
			or local_density_this_epoch
			or diagnostic_writer is not None
			or topk_writer is not None
		) and not is_gpu
		local_exact_diagnostics = None
		local_density_diagnostics = None
		if diagnostic_writer is not None:
			local_exact_diagnostics = _local_force_diagnostics(
				"local_exact",
				local_exact_enabled,
				drfft_params.get("local_exact_weight", 0.1),
				None,
				n_vertices,
				extra={
					"local_exact_k": int(drfft_params.get("local_exact_k", 8)),
					"local_exact_radius": np.nan,
				},
			)
			local_density_diagnostics = _local_force_diagnostics(
				"local_density_pressure",
				local_density_enabled,
				drfft_params.get("local_density_pressure_weight", 0.05),
				None,
				n_vertices,
			)
	### Attractive force
		if record_epoch_memory:
			memory_recorder.record("optimizer_attraction", "begin", epoch=epoch_itr)
		if is_gpu:
			attr_event_start = _cuda_stage_start()
		else:
			t1 = time()
		if attraction_mode == 'sampling':
			if not is_gpu:
				if attraction_schedule_mode == "active_edge_calendar":
					if epoch_itr > 0:
						prev_row_start = attraction_calendar_epoch_row_offsets[epoch_itr - 1]
						prev_row_end = attraction_calendar_epoch_row_offsets[epoch_itr]
						calendar_zero_t0 = perf_counter()
						_zero_calendar_attr_rows(
							attr_force,
							attraction_calendar_rows,
							prev_row_start,
							prev_row_end,
						)
						attraction_calendar_execute_zero_time += (
							perf_counter() - calendar_zero_t0
						)
					row_start = attraction_calendar_epoch_row_offsets[epoch_itr]
					row_end = attraction_calendar_epoch_row_offsets[epoch_itr + 1]
					calendar_kernel_t0 = perf_counter()
					UMAP_AttrForce_sampling_calendar(
						attr_force,
						embedding,
						attraction_calendar_rows,
						attraction_calendar_edge_offsets,
						attraction_calendar_edges,
						edgetgt,
						row_start,
						row_end,
						drfft_params['umap_a'],
						drfft_params['umap_b'],
						alpha,
					)
					attraction_calendar_execute_kernel_time += (
						perf_counter() - calendar_kernel_t0
					)
					attraction_calendar_epochs += 1
				elif attraction_schedule_mode == "active_edge_periodic":
					calendar_zero_t0 = perf_counter()
					attr_force.fill(0.0)
					attraction_calendar_execute_zero_time += (
						perf_counter() - calendar_zero_t0
					)
					calendar_kernel_t0 = perf_counter()
					active_events, active_rows = UMAP_AttrForce_sampling_periodic(
						attr_force,
						embedding,
						attraction_periodic_bucket_offsets,
						attraction_periodic_edges,
						attraction_periodic_edge_sources,
						attraction_periodic_due_epochs,
						attraction_periodic_row_epoch_marks,
						edgetgt,
						epoch_itr,
						drfft_params['umap_a'],
						drfft_params['umap_b'],
						alpha,
					)
					attraction_calendar_execute_kernel_time += (
						perf_counter() - calendar_kernel_t0
					)
					attraction_calendar_execute_event_count += int(active_events)
					attraction_calendar_execute_active_rows += int(active_rows)
					attraction_calendar_epochs += 1
				else:
					UMAP_AttrForce_sampling(
						attr_force, embedding, edgesrc, edgetgt, n_vertices,
						drfft_params['umap_a'], drfft_params['umap_b'], alpha,
						epoch_itr, epochs_per_sample, epoch_of_next_sample,
						whether_known_points, known_points_positions,
						known_points_reverse_index, soft_constraint, constraint_weight,
					)
					attraction_thread_per_row_epochs += 1
			else:
				if attraction_kernel_mode == "warp_per_row":
					UMAP_AttrForce_sampling_warp_per_row_cu(
						attraction_warp_grid_dim, attraction_warp_block_dim, (
							attr_force, embedding, edgesrc, edgetgt, n_vertices,
							cupy.float32(drfft_params['umap_a']), cupy.float32(drfft_params['umap_b']), alpha,
							epoch_itr, epochs_per_sample, epoch_of_next_sample,
						)
					)
					attraction_warp_per_row_epochs += 1
				else:
					UMAP_AttrForce_sampling_cu(optimizer_grid_dim, optimizer_block_dim, (
						attr_force, embedding, edgesrc, edgetgt, n_vertices,
						cupy.float32(drfft_params['umap_a']), cupy.float32(drfft_params['umap_b']), alpha,
						epoch_itr, epochs_per_sample, epoch_of_next_sample,
					))
					attraction_thread_per_row_epochs += 1
		else:
			if not is_gpu:
				UMAP_AttrForce(
					attr_force, embedding, edgesrc, edgetgt, pos_effects,
					n_vertices, n_components, drfft_params['umap_a'],
					drfft_params['umap_b'], alpha,
					whether_known_points, known_points_positions,
					known_points_reverse_index, soft_constraint, constraint_weight,
				)
				attraction_thread_per_row_epochs += 1
			else:
				if attraction_kernel_mode == "warp_per_row":
					UMAP_AttrForce_warp_per_row_cu(
						attraction_warp_grid_dim, attraction_warp_block_dim, (
							attr_force, embedding, edgesrc, edgetgt, pos_effects,
							n_vertices, cupy.float32(drfft_params['umap_a']),
							cupy.float32(drfft_params['umap_b']), alpha,
						)
					)
					attraction_warp_per_row_epochs += 1
				else:
					UMAP_AttrForce_cu(optimizer_grid_dim, optimizer_block_dim, (
						attr_force, embedding, edgesrc, edgetgt, pos_effects,
						n_vertices, cupy.float32(drfft_params['umap_a']),
						cupy.float32(drfft_params['umap_b']), alpha,
					))
					attraction_thread_per_row_epochs += 1
		if hard_constraint:
			attr_force[whether_known_points, 0] = 0.0
			attr_force[whether_known_points, 1] = 0.0
		if diagnostic_writer is not None:
			attr_norm_pre_degree_damping = _force_norms_numpy(attr_force, is_gpu)
		if damping_enabled:
			attr_force *= attraction_degree_damping_scale.reshape((-1, 1))
		if diagnostic_writer is not None:
			attr_norm_post_degree_damping = _force_norms_numpy(attr_force, is_gpu)
		if is_gpu:
			_cuda_stage_end(attr_event_start, cuda_attr_events)
			attraction_epoch_time = 0.0
		else:
			t2 = time()
			attraction_epoch_time = t2 - t1
			attr_time.append(attraction_epoch_time)
		if record_epoch_memory:
			memory_recorder.record("optimizer_attraction", "end", epoch=epoch_itr)

		# if cupy.isnan(attr_force).any():
		# 	print('NaN exists in attr_force')
		# 	print(epoch_itr)

	### Repulsive force
		if record_epoch_memory:
			memory_recorder.record("optimizer_repulsion", "begin", epoch=epoch_itr)
		if is_gpu:
			repl_event_start = _cuda_stage_start()
		else:
			t1 = time()
			ibfft_start_time = perf_counter()
		fft_stage_index = fft_stage_index_for_epoch(
			resolved_fft_stages, epoch_itr
		)
		fft_stage_epoch_indices.append(fft_stage_index)
		n_interpolation_points = resolved_fft_stages[
			fft_stage_index
		].n_interpolation_points
		gpu_repl_postprocess = None
		p2m_epoch_diagnostics = {}
		ibfft_epoch_timing = {}

		if repulsion_mode == 'true_loss':
			if not is_gpu:
				fused_update = None
				if fused_m2p_update_this_epoch:
					fused_update = {
						'embedding': embedding,
						'attr_force': attr_force,
						'alpha': alpha,
						'neg_effects': neg_effects,
						'clip_norm': repulsion_clip_norm,
					}
				# H-2 FIX: pass shared workspace dict for buffer reuse
				ibfft_result = ibFFT_repulsive_sampling(
					embedding, n_interpolation_points,
					drfft_params['intervals_per_integer'],
					drfft_params['min_num_intervals'],
					drfft_params['tfdp_gamma'],
					drfft_params['tfdp_paraFactor'],
					None,
					drfft_params['n_boxes_per_dim'],
					kernel_method='UMAP_kernel',
					umap_a=drfft_params['umap_a'],
					umap_b=drfft_params['umap_b'],
					umap_gamma=drfft_params['umap_gamma'],
					umap_epsilon=drfft_params['umap_epsilon'],
					ibfft_kernel_clip=drfft_params['ibfft_kernel_clip'],
					ibfft_kernel_subsample_mode=kernel_subsample_mode,
					ibfft_kernel_subsample_radius_cells=kernel_subsample_radius_cells,
					ibfft_kernel_subsample_points=kernel_subsample_points,
					random_state=sampling_rng,
					deterministic=bool(drfft_params.get('deterministic', False)),
					p2m_mode=drfft_params.get('p2m_mode', 'auto'),
					_workspace=_ibfft_ws,
					workspace_policy=drfft_params.get('workspace_policy', 'auto'),
					workspace_limit_bytes=drfft_params.get('workspace_limit_bytes'),
					fft_kernel_cache=getattr(cpu_workspace, 'fft_kernel_cache', None),
					fft_kernel_cache_policy=drfft_params.get('fft_kernel_cache_policy', 'auto'),
					fft_kernel_cache_limit_bytes=drfft_params.get('fft_kernel_cache_limit_bytes'),
					fft_kernel_cache_max_entries=drfft_params.get('fft_kernel_cache_max_entries'),
					p2m_diagnostics=p2m_epoch_diagnostics,
					timing_diagnostics=ibfft_epoch_timing,
					return_grid_context=need_grid_context,
					fused_update=fused_update,
				)
				if fused_m2p_update_this_epoch:
					repl_force = None
					grid_context = None
					fused_m2p_update_epochs += 1
				elif need_grid_context:
					repl_force, grid_context = ibfft_result
				else:
					repl_force = ibfft_result
					grid_context = None
				if not fused_m2p_update_this_epoch:
					repl_force = tFDP_ReplForce_ibFFT_neg_effect(
						repl_force, alpha, neg_effects
					)
			else:
				fused_update = None
				if fused_m2p_update_this_epoch:
					fused_update = {
						'embedding': embedding,
						'attr_force': attr_force,
						'alpha': alpha,
						'neg_effects': neg_effects,
						'clip_norm': repulsion_clip_norm,
					}
				repl_force = ibFFT_repulsive_sampling_GPU(
					embedding, n_interpolation_points,
					drfft_params['intervals_per_integer'],
					drfft_params['min_num_intervals'],
					drfft_params['tfdp_gamma'],
					drfft_params['tfdp_paraFactor'],
					None,
					drfft_params['n_boxes_per_dim'],
					kernel_method='UMAP_kernel',
					umap_a=cupy.float32(drfft_params['umap_a']),
					umap_b=cupy.float32(drfft_params['umap_b']),
					umap_gamma=cupy.float32(drfft_params['umap_gamma']),
					umap_epsilon=cupy.float32(drfft_params['umap_epsilon']),
					ibfft_kernel_clip=cupy.float32(drfft_params['ibfft_kernel_clip']),
					random_state=sampling_rng,
					deterministic=bool(drfft_params.get('deterministic', False)),
					p2m_mode=drfft_params.get('p2m_mode', 'auto'),
					workspace=gpu_workspace,
					workspace_policy=drfft_params.get('workspace_policy', 'auto'),
					workspace_limit_bytes=drfft_params.get('workspace_limit_bytes'),
					fft_kernel_cache_policy=drfft_params.get('fft_kernel_cache_policy', 'auto'),
					fft_kernel_cache_limit_bytes=drfft_params.get('fft_kernel_cache_limit_bytes'),
					fft_kernel_cache_max_entries=drfft_params.get('fft_kernel_cache_max_entries'),
					p2m_diagnostics=p2m_epoch_diagnostics,
					p2m_event_pairs=cuda_p2m_events,
					timing_event_pairs=cuda_ibfft_timing_events,
					fused_update=fused_update,
				)
				p2m_resolved_modes.add(
					str(p2m_epoch_diagnostics.get('resolved_mode', 'unknown'))
				)
				if fused_m2p_update_this_epoch:
					fused_m2p_update_epochs += 1
				else:
					gpu_repl_postprocess = (True, neg_effects, cupy.float32(0.0))
		else:
			if not is_gpu:
				# H-2 FIX: pass shared workspace dict for buffer reuse.
				# Sampling mode subsets Y so active N <= n_vertices; the workspace dict
				# handles shape mismatch by reallocating only on the first epoch.
				ibfft_result = ibFFT_repulsive_sampling(
					embedding, n_interpolation_points,
					drfft_params['intervals_per_integer'],
					drfft_params['min_num_intervals'],
					drfft_params['tfdp_gamma'],
					drfft_params['tfdp_paraFactor'],
					probabilities,
					drfft_params['n_boxes_per_dim'],
					kernel_method='UMAP_kernel',
					umap_a=drfft_params['umap_a'],
					umap_b=drfft_params['umap_b'],
					umap_gamma=drfft_params['umap_gamma'],
					umap_epsilon=drfft_params['umap_epsilon'],
					ibfft_kernel_clip=drfft_params['ibfft_kernel_clip'],
					ibfft_kernel_subsample_mode=kernel_subsample_mode,
					ibfft_kernel_subsample_radius_cells=kernel_subsample_radius_cells,
					ibfft_kernel_subsample_points=kernel_subsample_points,
					random_state=sampling_rng,
					deterministic=bool(drfft_params.get('deterministic', False)),
					p2m_mode=drfft_params.get('p2m_mode', 'auto'),
					_workspace=_ibfft_ws,
					workspace_policy=drfft_params.get('workspace_policy', 'auto'),
					workspace_limit_bytes=drfft_params.get('workspace_limit_bytes'),
					fft_kernel_cache=getattr(cpu_workspace, 'fft_kernel_cache', None),
					fft_kernel_cache_policy=drfft_params.get('fft_kernel_cache_policy', 'auto'),
					fft_kernel_cache_limit_bytes=drfft_params.get('fft_kernel_cache_limit_bytes'),
					fft_kernel_cache_max_entries=drfft_params.get('fft_kernel_cache_max_entries'),
					p2m_diagnostics=p2m_epoch_diagnostics,
					timing_diagnostics=ibfft_epoch_timing,
					return_grid_context=need_grid_context,
				)
				if need_grid_context:
					repl_force, grid_context = ibfft_result
				else:
					repl_force = ibfft_result
					grid_context = None
				repl_force *= alpha
				repl_force *= neg_effect
			else:
				repl_force = ibFFT_repulsive_sampling_GPU(
					embedding, n_interpolation_points,
					drfft_params['intervals_per_integer'],
					drfft_params['min_num_intervals'],
					drfft_params['tfdp_gamma'],
					drfft_params['tfdp_paraFactor'],
					probabilities,
					drfft_params['n_boxes_per_dim'],
					kernel_method='UMAP_kernel',
					umap_a=cupy.float32(drfft_params['umap_a']),
					umap_b=cupy.float32(drfft_params['umap_b']),
					umap_gamma=cupy.float32(drfft_params['umap_gamma']),
					umap_epsilon=cupy.float32(drfft_params['umap_epsilon']),
					ibfft_kernel_clip=cupy.float32(drfft_params['ibfft_kernel_clip']),
					random_state=sampling_rng,
					deterministic=bool(drfft_params.get('deterministic', False)),
					p2m_mode=drfft_params.get('p2m_mode', 'auto'),
					workspace=gpu_workspace,
					workspace_policy=drfft_params.get('workspace_policy', 'auto'),
					workspace_limit_bytes=drfft_params.get('workspace_limit_bytes'),
					fft_kernel_cache_policy=drfft_params.get('fft_kernel_cache_policy', 'auto'),
					fft_kernel_cache_limit_bytes=drfft_params.get('fft_kernel_cache_limit_bytes'),
					fft_kernel_cache_max_entries=drfft_params.get('fft_kernel_cache_max_entries'),
					p2m_diagnostics=p2m_epoch_diagnostics,
					p2m_event_pairs=cuda_p2m_events,
					timing_event_pairs=cuda_ibfft_timing_events,
				)
				p2m_resolved_modes.add(
					str(p2m_epoch_diagnostics.get('resolved_mode', 'unknown'))
				)
				gpu_repl_postprocess = (False, repl_force, neg_effect)

		if is_gpu:
			ibfft_epoch_time = 0.0
		else:
			p2m_time.append(float(p2m_epoch_diagnostics.get('time_s', 0.0)))
			for timing_name in ibfft_timing_names:
				ibfft_timing_totals[timing_name] += float(
					ibfft_epoch_timing.get(timing_name, 0.0)
				)
			p2m_resolved_modes.add(
				str(p2m_epoch_diagnostics.get('resolved_mode', 'unknown'))
			)
			ibfft_call_time = perf_counter() - ibfft_start_time
			if grid_context is not None:
				local_grid_time = float(grid_context.build_time_s)
			ibfft_epoch_time = max(0.0, ibfft_call_time - local_grid_time)
		cache_hit = p2m_epoch_diagnostics.get('kernel_cache_hit')
		if cache_hit is not None:
			if bool(cache_hit):
				fft_kernel_cache_hits += 1
			else:
				fft_kernel_cache_misses += 1
		cache_key_token = p2m_epoch_diagnostics.get('kernel_cache_key_token')
		if cache_key_token is not None:
			fft_kernel_cache_exact_keys.add(str(cache_key_token))
		cache_miss_type = p2m_epoch_diagnostics.get('kernel_cache_miss_type')
		if cache_miss_type == 'compulsory':
			fft_kernel_cache_compulsory_misses += 1
		elif cache_miss_type == 'eviction':
			fft_kernel_cache_eviction_misses += 1
		fft_kernel_cache_evictions += int(
			p2m_epoch_diagnostics.get('kernel_cache_evictions') or 0
		)
		fft_kernel_cache_evicted_bytes += int(
			p2m_epoch_diagnostics.get('kernel_cache_evicted_bytes') or 0
		)
		if p2m_epoch_diagnostics.get('kernel_cache_store_oversized'):
			fft_kernel_cache_store_oversized += 1
		if 'kernel_cache_bytes' in p2m_epoch_diagnostics:
			fft_kernel_cache_bytes = int(
				p2m_epoch_diagnostics.get('kernel_cache_bytes') or 0
			)
		if 'kernel_cache_entries' in p2m_epoch_diagnostics:
			fft_kernel_cache_entries = int(
				p2m_epoch_diagnostics.get('kernel_cache_entries') or 0
			)
		fft_kernel_cache_peak_bytes = max(
			fft_kernel_cache_peak_bytes,
			int(p2m_epoch_diagnostics.get('kernel_cache_peak_bytes') or 0),
		)
		fft_kernel_cache_peak_entries = max(
			fft_kernel_cache_peak_entries,
			int(p2m_epoch_diagnostics.get('kernel_cache_peak_entries') or 0),
		)
		if p2m_epoch_diagnostics.get('kernel_cache_limit_bytes') is not None:
			fft_kernel_cache_limit_bytes = int(
				p2m_epoch_diagnostics['kernel_cache_limit_bytes']
			)
		if p2m_epoch_diagnostics.get('kernel_cache_max_entries') is not None:
			fft_kernel_cache_max_entries = int(
				p2m_epoch_diagnostics['kernel_cache_max_entries']
			)
		if p2m_epoch_diagnostics.get('kernel_cache_policy') is not None:
			fft_kernel_cache_policy = str(
				p2m_epoch_diagnostics['kernel_cache_policy']
			)
		if p2m_epoch_diagnostics.get('kernel_cache_scope') is not None:
			fft_kernel_cache_scope = str(
				p2m_epoch_diagnostics['kernel_cache_scope']
			)
		plan_cache_hit = p2m_epoch_diagnostics.get('fft_plan_cache_hit')
		if plan_cache_hit is not None:
			if bool(plan_cache_hit):
				fft_plan_cache_hits += 1
			else:
				fft_plan_cache_misses += 1
		grid_level = p2m_epoch_diagnostics.get('grid_quantization_level')
		if grid_level is not None:
			fft_grid_quantization_levels.add(int(grid_level))

		# Experimental local mechanisms are additive near-field corrections to
		# ibFFT's global repulsion and intentionally run before final clipping.
		if local_exact_this_epoch:
			local_exact_start_time = perf_counter()
			local_prep_start_time = perf_counter()
			local_radius = _local_exact_radius(
				embedding,
				local_edge_sources,
				local_edge_targets,
				drfft_params.get("local_exact_radius_factor", 0.5),
				grid_context.cell_size,
			)
			local_density_threshold = _local_exact_density_threshold(
				drfft_params, grid_context
			)
			local_prep_time = perf_counter() - local_prep_start_time
			local_kernel_start_time = perf_counter()
			local_kernel_args = (
				embedding,
				grid_context.point_cell_x,
				grid_context.point_cell_y,
				grid_context.sorted_indices,
				grid_context.cell_start,
				grid_context.cell_end,
				grid_context.n_cells_x,
				grid_context.n_cells_y,
				int(drfft_params.get("local_exact_k", 8)),
				local_radius,
				float(drfft_params['umap_a']),
				float(drfft_params['umap_b']),
				float(drfft_params['umap_gamma']),
				float(drfft_params['umap_epsilon']),
				_scalar_float(alpha),
				float(drfft_params.get("local_exact_clip", 4.0)),
				bool(drfft_params.get("local_exact_symmetric", True)),
				grid_context.density,
				grid_context.flat_cell_id,
				bool(drfft_params.get("local_exact_density_filter", False)),
				float(local_density_threshold),
				bool(drfft_params.get("local_exact_density_include_neighbor_cells", False)),
			)
			if runtime_diagnostic_writer is not None:
				(
					local_exact_force,
					local_candidate_counts,
					local_distance_counts,
					local_selected_counts,
					local_eligible,
					local_stage_ticks,
					local_timing_sample_count,
				) = _local_exact_repulsion_force_profiled(
					*local_kernel_args,
					int(drfft_params.get("local_exact_timing_sample_size", 2048)),
				)
			else:
				local_exact_force = _local_exact_repulsion_force(*local_kernel_args)
			local_kernel_time = perf_counter() - local_kernel_start_time
			if runtime_diagnostic_writer is not None:
				tick_total = float(np.sum(local_stage_ticks))
				if tick_total > 0.0 and _PROFILE_SECONDS_PER_TICK > 0.0:
					local_stage_times = (
						local_stage_ticks.astype(np.float64) / tick_total * local_kernel_time
					)
			local_outer_accum_start_time = perf_counter()
			repl_force += (
				float(drfft_params.get("local_exact_weight", 0.1)) * local_exact_force
			)
			local_outer_accum_time = perf_counter() - local_outer_accum_start_time
			local_exact_total_time = perf_counter() - local_exact_start_time
			local_exact_norms = None
			if diagnostic_writer is not None or runtime_diagnostic_writer is not None:
				local_copy_start_time = perf_counter()
				local_exact_norms = _force_norms_numpy(local_exact_force, False)
				local_array_copy_time += perf_counter() - local_copy_start_time
			if diagnostic_writer is not None:
				local_exact_diagnostics = _local_force_diagnostics(
					"local_exact",
					local_exact_enabled,
					drfft_params.get("local_exact_weight", 0.1),
					local_exact_norms,
					n_vertices,
					extra={
						"local_exact_k": int(drfft_params.get("local_exact_k", 8)),
						"local_exact_radius": float(local_radius),
					},
				)
		if local_density_this_epoch:
			local_density_force = _local_density_pressure_force(
				embedding,
				grid_context,
				int(drfft_params.get("local_density_pressure_min_count", 4)),
				float(drfft_params.get("local_density_pressure_clip", 4.0)),
				float(drfft_params.get("local_density_pressure_power", 1.0)),
			)
			repl_force += (
				float(drfft_params.get("local_density_pressure_weight", 0.05))
				* local_density_force
			)
			local_density_norms = None
			if diagnostic_writer is not None:
				local_density_norms = _force_norms_numpy(local_density_force, False)
			if diagnostic_writer is not None:
				local_density_diagnostics = _local_force_diagnostics(
					"local_density_pressure",
					local_density_enabled,
					drfft_params.get("local_density_pressure_weight", 0.05),
					local_density_norms,
					n_vertices,
				)

		clip_needs_norms = diagnostic_writer is not None or topk_writer is not None
		fused_repulsion_clip = False
		if fused_m2p_update_this_epoch:
			fused_repulsion_clip = repulsion_clip_norm is not None
		elif is_gpu and gpu_repl_postprocess is not None:
			use_vector_neg_effects, neg_effects_arg, neg_effect_scalar = gpu_repl_postprocess
			fused_repulsion_clip = (
				repulsion_clip_norm is not None and not clip_needs_norms
			)
			UMAP_ReplForcePostprocess_cu(optimizer_grid_dim, optimizer_block_dim, (
				repl_force,
				neg_effects_arg,
				cupy.float32(neg_effect_scalar),
				whether_known_points,
				int(use_vector_neg_effects),
				int(hard_constraint),
				int(fused_repulsion_clip),
				cupy.float32(0.0 if repulsion_clip_norm is None else repulsion_clip_norm),
				alpha,
				n_vertices,
			))
		elif hard_constraint:
			repl_force[whether_known_points, 0] = 0.0
			repl_force[whether_known_points, 1] = 0.0
		if (
			not fused_m2p_update_this_epoch
			and repulsion_clip_norm is not None
			and not fused_repulsion_clip
		):
			repl_force_pre_clip_norms = _clip_force_norm_inplace(
				repl_force,
				repulsion_clip_norm,
				is_gpu,
				return_norms=clip_needs_norms,
			)
		elif (
			not fused_m2p_update_this_epoch
			and (diagnostic_writer is not None or topk_writer is not None)
		):
			repl_force_pre_clip_norms = _force_norms_numpy(repl_force, is_gpu)
		if is_gpu:
			_cuda_stage_end(repl_event_start, cuda_repl_events)
		else:
			repl_time.append(perf_counter() - ibfft_start_time)
		if record_epoch_memory:
			memory_recorder.record("optimizer_repulsion", "end", epoch=epoch_itr)

	### Update embedding and alpha
		if record_epoch_memory:
			memory_recorder.record("optimizer_apply", "begin", epoch=epoch_itr)
		if is_gpu:
			appl_event_start = (
				None if fused_m2p_update_this_epoch else _cuda_stage_start()
			)
			update_start_time = 0.0
		else:
			t1 = time()
			update_start_time = (
				0.0 if fused_m2p_update_this_epoch else perf_counter()
			)
		alpha_for_epoch = (
			_scalar_float(alpha)
			if diagnostic_writer is not None or topk_writer is not None
			else 0.0
		)
		if (
			not fused_m2p_update_this_epoch
			and noise_mode == 'force'
			and noise_rng is not None
			and noise_scale > 0.0
		):
			_add_gaussian_noise_inplace(
				attr_force, noise_rng, noise_scale, is_gpu,
				hard_constraint, whether_known_points,
			)
		attr_force_for_diagnostics = attr_force
		repl_force_for_diagnostics = (
			None if fused_m2p_update_this_epoch else repl_force
		)
		total_force_pre_clip = None
		if (
			not fused_m2p_update_this_epoch
			and (diagnostic_writer is not None or topk_writer is not None)
		):
			attr_force_for_diagnostics = _to_numpy_array(attr_force).copy()
			repl_force_for_diagnostics = _to_numpy_array(repl_force).copy()
			total_force_pre_clip = attr_force_for_diagnostics + repl_force_for_diagnostics
		if not fused_m2p_update_this_epoch and total_update_clip_norm is not None:
			total_update_pre_clip_norms = _clip_total_update_norm_inplace(
				attr_force,
				repl_force,
				total_update_clip_norm,
				is_gpu,
				return_norms=(diagnostic_writer is not None),
			)
		elif not fused_m2p_update_this_epoch and diagnostic_writer is not None:
			total_update_pre_clip_norms = _force_norms_numpy(attr_force + repl_force, is_gpu)
		if fused_m2p_update_this_epoch:
			if is_gpu:
				alpha = cupy.float32(
					initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))
				)
			else:
				alpha = initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))
		elif not is_gpu:
			UMAP_ApplyForce(embedding, attr_force, repl_force, n_vertices, n_components)
			if noise_mode == 'embedding' and noise_rng is not None and noise_scale > 0.0:
				_add_gaussian_noise_inplace(
					embedding, noise_rng, noise_scale, is_gpu,
					hard_constraint, whether_known_points,
				)
			alpha = initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))
		else:
			UMAP_ApplyForce_cu(optimizer_grid_dim, optimizer_block_dim, (embedding, attr_force, repl_force, n_vertices))
			if noise_mode == 'embedding' and noise_rng is not None and noise_scale > 0.0:
				_add_gaussian_noise_inplace(
					embedding, noise_rng, noise_scale, is_gpu,
					hard_constraint, whether_known_points,
				)
			alpha = cupy.float32(initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs))))
		if is_gpu:
			update_epoch_time = 0.0
			if appl_event_start is not None:
				_cuda_stage_end(appl_event_start, cuda_appl_events)
		else:
			update_epoch_time = (
				0.0
				if fused_m2p_update_this_epoch
				else perf_counter() - update_start_time
			)
		if record_epoch_memory:
			memory_recorder.record("optimizer_apply", "end", epoch=epoch_itr)
		diagnostics_start_time = (
			perf_counter()
			if diagnostic_writer is not None
			or topk_writer is not None
			or runtime_diagnostic_writer is not None
			else 0.0
		)
		if topk_writer is not None:
			total_force_post_clip = _to_numpy_array(attr_force + repl_force).copy()
			previous_diagnostic_radii = _write_topk_rows(
				topk_writer,
				epoch_itr,
				alpha_for_epoch,
				attr_force_for_diagnostics,
				repl_force_for_diagnostics,
				repl_force_pre_clip_norms,
				total_force_pre_clip,
				total_force_post_clip,
				embedding,
				previous_diagnostic_radii,
				degrees,
				degree_percentiles,
				diagnostic_labels,
				diagnostic_point_ids,
				diagnostic_topk,
				repulsion_clip_norm,
				is_gpu,
				attraction_degree_damping_scale,
			)
		if diagnostic_writer is not None:
			total_force_post_clip = _to_numpy_array(attr_force + repl_force).copy()
			_write_diagnostic_row(
				diagnostic_writer,
				epoch_itr,
				alpha_for_epoch,
				attr_force_for_diagnostics,
				repl_force_for_diagnostics,
				repl_force_pre_clip_norms,
				total_update_pre_clip_norms,
				total_force_post_clip,
				embedding,
				diagnostic_thresholds,
				repulsion_clip_norm,
				total_update_clip_norm,
				local_exact_diagnostics,
				local_density_diagnostics,
				drfft_params['ibfft_kernel_clip'],
				drfft_params['umap_epsilon'],
				kernel_subsample_mode,
				kernel_subsample_radius_cells,
				kernel_subsample_points,
				grid_context,
				n_interpolation_points,
				attr_norm_pre_degree_damping,
				attr_norm_post_degree_damping,
				attraction_degree_damping_metadata,
				is_gpu,
			)
		diagnostics_epoch_time = (
			perf_counter() - diagnostics_start_time
			if diagnostics_start_time
			else 0.0
		)
		if not is_gpu:
			t2 = time()
			appl_time.append(0.0 if fused_m2p_update_this_epoch else t2-t1)
		if runtime_diagnostic_writer is not None:
			if grid_context is None:
				occupied_cells = np.empty(0, dtype=np.float64)
			else:
				occupied_cells = grid_context.density[grid_context.density > 0].astype(
					np.float64, copy=False
				)
			eligible_mask = local_eligible.astype(bool, copy=False)
			eligible_count = int(np.sum(eligible_mask))
			active_count = (
				0 if local_exact_norms is None else int(np.sum(local_exact_norms > 0.0))
			)
			if eligible_count:
				candidate_values = local_candidate_counts[eligible_mask]
				distance_values = local_distance_counts[eligible_mask]
				selected_values = local_selected_counts[eligible_mask]
			else:
				candidate_values = np.empty(0, dtype=np.int64)
				distance_values = np.empty(0, dtype=np.int64)
				selected_values = np.empty(0, dtype=np.int64)
			runtime_row = {
				"epoch": int(epoch_itr),
				"epoch_time_s": perf_counter() - epoch_start_time,
				"attraction_time_s": float(attraction_epoch_time),
				"ibfft_time_s": float(ibfft_epoch_time),
				"local_exact_total_time_s": float(local_exact_total_time),
				"local_grid_time_s": float(local_grid_time),
				"local_prep_time_s": float(local_prep_time),
				"local_candidate_time_s": float(local_stage_times[0]),
				"local_distance_time_s": float(local_stage_times[1]),
				"local_topk_time_s": float(local_stage_times[2]),
				"local_force_time_s": float(local_stage_times[3]),
				"local_accum_time_s": float(local_stage_times[4] + local_outer_accum_time),
				"local_array_copy_time_s": float(local_array_copy_time),
				"update_time_s": float(update_epoch_time),
				"diagnostics_time_s": float(diagnostics_epoch_time),
				"local_exact_ran": int(local_exact_this_epoch),
				"local_exact_density_threshold": float(local_density_threshold),
				"local_exact_eligible_count": eligible_count,
				"local_exact_eligible_pct": 100.0 * eligible_count / float(max(int(n_vertices), 1)),
				"local_exact_active_count": active_count,
				"local_exact_active_pct": 100.0 * active_count / float(max(int(n_vertices), 1)),
				"local_exact_pair_count": int(np.sum(local_selected_counts)),
				"cell_occupancy_mean": 0.0 if occupied_cells.size == 0 else float(np.mean(occupied_cells)),
				"cell_occupancy_p95": 0.0 if occupied_cells.size == 0 else float(np.percentile(occupied_cells, 95.0)),
				"cell_occupancy_p99": 0.0 if occupied_cells.size == 0 else float(np.percentile(occupied_cells, 99.0)),
				"cell_occupancy_max": 0.0 if occupied_cells.size == 0 else float(np.max(occupied_cells)),
				"local_timing_sample_count": int(local_timing_sample_count),
			}
			runtime_row.update(_count_stats("local_exact_candidate_count", candidate_values))
			runtime_row.update(_count_stats("local_exact_distance_count", distance_values))
			runtime_row.update(_count_stats("local_exact_selected_count", selected_values))
			runtime_diagnostic_writer.writerow(runtime_row)

	if diagnostic_file is not None:
		diagnostic_file.close()
	if topk_file is not None:
		topk_file.close()
	if runtime_diagnostic_file is not None:
		runtime_diagnostic_file.close()

	if is_gpu:
		cupy.cuda.Stream.null.synchronize()
		attr_time = [
			cupy.cuda.get_elapsed_time(start_event, end_event) / 1000.0
			for start_event, end_event in cuda_attr_events
		]
		repl_time = [
			cupy.cuda.get_elapsed_time(start_event, end_event) / 1000.0
			for start_event, end_event in cuda_repl_events
		]
		p2m_time = [
			cupy.cuda.get_elapsed_time(start_event, end_event) / 1000.0
			for start_event, end_event in cuda_p2m_events
		]
		for timing_name in ibfft_timing_names:
			event_pairs = cuda_ibfft_timing_events.get(timing_name, ())
			ibfft_timing_totals[timing_name] = sum(
				cupy.cuda.get_elapsed_time(start_event, end_event) / 1000.0
				for start_event, end_event in event_pairs
			)
		appl_time = [
			cupy.cuda.get_elapsed_time(start_event, end_event) / 1000.0
			for start_event, end_event in cuda_appl_events
		]

	fft_stage_repl_times = [0.0] * len(resolved_fft_stages)
	fft_stage_p2m_times = [0.0] * len(resolved_fft_stages)
	fft_stage_first_epoch_repl_times = [0.0] * len(resolved_fft_stages)
	fft_stage_seen = [False] * len(resolved_fft_stages)
	for stage_index, elapsed in zip(fft_stage_epoch_indices, repl_time):
		elapsed = float(elapsed)
		fft_stage_repl_times[stage_index] += elapsed
		if not fft_stage_seen[stage_index]:
			fft_stage_first_epoch_repl_times[stage_index] = elapsed
			fft_stage_seen[stage_index] = True
	for stage_index, elapsed in zip(fft_stage_epoch_indices, p2m_time):
		fft_stage_p2m_times[stage_index] += float(elapsed)

	repl_total_time = float(np.sum(repl_time))
	p2m_total_time = float(np.sum(p2m_time))
	# CUDA's consecutive event ranges cover the complete ibFFT device timeline.
	# CPU additionally reports the small residual not assigned to an explicit
	# substage. Both paths therefore expose an exclusive decomposition.
	ibfft_exclusive_total = float(sum(ibfft_timing_totals.values()))
	ibfft_total_time = ibfft_exclusive_total
	repl_postprocess_time = max(0.0, repl_total_time - ibfft_total_time)
	fft_kernel_cache_lookups = fft_kernel_cache_hits + fft_kernel_cache_misses
	fft_plan_cache_lookups = fft_plan_cache_hits + fft_plan_cache_misses
	attraction_calendar_execute_time = float(
		attraction_calendar_execute_zero_time
		+ attraction_calendar_execute_kernel_time
	)
	if attraction_schedule_mode == "active_edge_calendar":
		attraction_calendar_execute_event_count = int(attraction_calendar_event_count)
		attraction_calendar_execute_active_rows = int(attraction_calendar_active_rows)
	elif attraction_schedule_mode == "active_edge_periodic":
		attraction_calendar_active_rows = int(attraction_calendar_execute_active_rows)
		if attraction_calendar_execute_event_count:
			attraction_calendar_event_count = int(attraction_calendar_execute_event_count)
	scanned_edge_slots = float(max(edge_count_int * int(n_epochs), 1))
	scanned_row_slots = float(max(n_vertices_int * int(n_epochs), 1))
	attraction_active_event_ratio = float(
		attraction_calendar_event_count / scanned_edge_slots
	)
	attraction_active_row_ratio = float(
		attraction_calendar_active_rows / scanned_row_slots
	)
	attraction_events_per_active_row = float(
		attraction_calendar_event_count / max(attraction_calendar_active_rows, 1)
	)
	time_costs = {
		'opt_prep_time': opt_prep_time,
		'attr_time': np.sum(attr_time),
		'repl_time': repl_total_time,
		'p2m_time': p2m_total_time,
		'repl_setup_time': ibfft_timing_totals['setup_time_s'],
		'repl_kernel_build_time': ibfft_timing_totals['kernel_build_time_s'],
		'repl_kernel_fft_time': ibfft_timing_totals['kernel_fft_time_s'],
		'repl_point_setup_time': ibfft_timing_totals['point_setup_time_s'],
		'repl_mesh_clear_time': ibfft_timing_totals['mesh_clear_time_s'],
		'repl_fft_plan_build_time': ibfft_timing_totals['fft_plan_build_time_s'],
		'repl_fft_forward_time': ibfft_timing_totals['fft_forward_time_s'],
		'repl_fft_multiply_time': ibfft_timing_totals['fft_multiply_time_s'],
		'repl_fft_inverse_time': ibfft_timing_totals['fft_inverse_time_s'],
		'repl_m2p_time': ibfft_timing_totals['m2p_time_s'],
		'repl_sampling_restore_time': ibfft_timing_totals['sampling_restore_time_s'],
		'repl_ibfft_other_time': ibfft_timing_totals['ibfft_other_time_s'],
		'repl_ibfft_total_time': ibfft_total_time,
		'repl_postprocess_time': repl_postprocess_time,
		'repl_kernel_cache_hits': int(fft_kernel_cache_hits),
		'repl_kernel_cache_misses': int(fft_kernel_cache_misses),
		'repl_kernel_cache_hit_rate': (
			float(fft_kernel_cache_hits / fft_kernel_cache_lookups)
			if fft_kernel_cache_lookups else 0.0
		),
		'repl_kernel_cache_exact_hits': int(fft_kernel_cache_hits),
		'repl_kernel_cache_exact_misses': int(fft_kernel_cache_misses),
		'repl_kernel_cache_exact_key_count': int(len(fft_kernel_cache_exact_keys)),
		'repl_kernel_cache_compulsory_misses': int(
			fft_kernel_cache_compulsory_misses
		),
		'repl_kernel_cache_eviction_misses': int(
			fft_kernel_cache_eviction_misses
		),
		'repl_kernel_cache_evictions': int(fft_kernel_cache_evictions),
		'repl_kernel_cache_evicted_bytes': int(fft_kernel_cache_evicted_bytes),
		'repl_kernel_cache_bytes': int(fft_kernel_cache_bytes),
		'repl_kernel_cache_peak_bytes': int(fft_kernel_cache_peak_bytes),
		'repl_kernel_cache_entries': int(fft_kernel_cache_entries),
		'repl_kernel_cache_peak_entries': int(fft_kernel_cache_peak_entries),
		'repl_kernel_cache_limit_bytes': fft_kernel_cache_limit_bytes,
		'repl_kernel_cache_max_entries': fft_kernel_cache_max_entries,
		'repl_kernel_cache_policy': (
			'' if fft_kernel_cache_policy is None else str(fft_kernel_cache_policy)
		),
		'repl_kernel_cache_scope': (
			'' if fft_kernel_cache_scope is None else str(fft_kernel_cache_scope)
		),
		'repl_kernel_cache_oversized_stores': int(
			fft_kernel_cache_store_oversized
		),
		'repl_fft_plan_cache_hits': int(fft_plan_cache_hits),
		'repl_fft_plan_cache_misses': int(fft_plan_cache_misses),
		'repl_fft_plan_cache_hit_rate': (
			float(fft_plan_cache_hits / fft_plan_cache_lookups)
			if fft_plan_cache_lookups else 0.0
		),
		'repl_grid_quantization_level_count': len(fft_grid_quantization_levels),
		'fused_m2p_update_epochs': int(fused_m2p_update_epochs),
		'fused_m2p_update_rate': float(fused_m2p_update_epochs / max(n_epochs, 1)),
		'attraction_kernel_mode': str(attraction_kernel_mode),
		'attraction_schedule_mode': str(attraction_schedule_mode),
		'attraction_thread_per_row_epochs': int(attraction_thread_per_row_epochs),
		'attraction_warp_per_row_epochs': int(attraction_warp_per_row_epochs),
		'attraction_calendar_epochs': int(attraction_calendar_epochs),
		'attraction_calendar_build_time': float(attraction_calendar_build_time),
		'attraction_calendar_build_count_time': float(attraction_calendar_build_count_time),
		'attraction_calendar_build_fill_time': float(attraction_calendar_build_fill_time),
		'attraction_calendar_build_sort_time': float(attraction_calendar_build_sort_time),
		'attraction_calendar_build_group_time': float(attraction_calendar_build_group_time),
		'attraction_calendar_build_schedule_time': float(attraction_calendar_build_schedule_time),
		'attraction_calendar_init_time': float(attraction_calendar_init_time),
		'attraction_calendar_execute_time': float(attraction_calendar_execute_time),
		'attraction_calendar_execute_zero_time': float(attraction_calendar_execute_zero_time),
		'attraction_calendar_execute_kernel_time': float(attraction_calendar_execute_kernel_time),
		'attraction_calendar_execute_event_count': int(attraction_calendar_execute_event_count),
		'attraction_calendar_execute_active_rows': int(attraction_calendar_execute_active_rows),
		'attraction_calendar_event_count': int(attraction_calendar_event_count),
		'attraction_calendar_active_rows': int(attraction_calendar_active_rows),
		'attraction_active_event_ratio': attraction_active_event_ratio,
		'attraction_active_row_ratio': attraction_active_row_ratio,
		'attraction_events_per_active_row': attraction_events_per_active_row,
		'attraction_calendar_bucket_count': int(attraction_calendar_bucket_count),
		'attraction_calendar_due_pattern_count': int(attraction_calendar_due_pattern_count),
		'attraction_calendar_estimated_bytes': int(attraction_calendar_estimated_bytes),
		'p2m_resolved_mode': (
			','.join(sorted(p2m_resolved_modes)) if p2m_resolved_modes else None
		),
		'fft_stage_schedule': format_resolved_fft_schedule(resolved_fft_stages),
		'fft_stage_count': int(len(resolved_fft_stages)),
		'fft_stage_transition_count': int(max(0, len(resolved_fft_stages) - 1)),
		'fft_stage_transition_epochs': ','.join(
			str(stage.start_epoch) for stage in resolved_fft_stages[1:]
		),
		'appl_time': np.sum(appl_time),
	}
	for stage_index, stage in enumerate(resolved_fft_stages):
		order = int(stage.n_interpolation_points)
		time_costs[f'fft_stage_epochs_p{order}'] = int(stage.n_epochs)
		time_costs[f'fft_stage_repl_time_p{order}'] = float(
			fft_stage_repl_times[stage_index]
		)
		time_costs[f'fft_stage_p2m_time_p{order}'] = float(
			fft_stage_p2m_times[stage_index]
		)
		time_costs[f'fft_stage_first_epoch_repl_time_p{order}'] = float(
			fft_stage_first_epoch_repl_times[stage_index]
		)

	if is_gpu: embedding = embedding.get()

	return embedding, time_costs


# ---------------------------------------------------------------------------
# Change A (Round-5): AOT Warmup — optimizer-side JIT kernels
#
# The ibfft.py module already warms up its own kernels at import time.
# This function handles the optimizer-side @numba.njit functions:
#   - UMAP_AttrForce_sampling (used in 'true_loss_ibFFT_sampling' path)
#   - UMAP_AttrForce          (used in 'true_loss_ibFFT' exact path)
#   - UMAP_ApplyForce         (used by both paths)
#
# Why sampling mode had ~10s cold start while exact mode only had ~0.8s:
# UMAP_AttrForce_sampling has a more complex type signature than
# UMAP_AttrForce (it includes epoch_itr, epochs_per_sample,
# epoch_of_next_sample arrays), so its JIT specialization takes longer.
# Pre-triggering it here shifts the cost to import time.
# ---------------------------------------------------------------------------

def _warmup_optimizer_jit_kernels() -> None:
	"""Pre-trigger JIT for optimizer-side numba kernels.

	Called once at module import.  Uses minimum-size dummy inputs to trigger
	specialization for float32/int32 types without meaningful computation.
	"""
	import warnings

	N = 4
	NC = 2
	emb_d = np.zeros((N, NC), dtype=np.float32)
	attr_d = np.zeros((N, NC), dtype=np.float32)
	repl_d = np.zeros((N, NC), dtype=np.float32)
	edgesrc_d = np.array([0, 1, 2, 3, 4], dtype=np.int32)  # 4 vertices, 1 edge each
	edgetgt_d = np.array([1, 2, 3, 0], dtype=np.int32)
	pos_eff_d = np.ones(4, dtype=np.float32)
	known_d = np.zeros(N, dtype=np.bool_)
	known_pos_d = np.zeros((0, NC), dtype=np.float32)
	known_rev_d = np.full(N, -1, dtype=np.int32)
	eps_d = np.ones(4, dtype=np.float32)
	eons_d = np.ones(4, dtype=np.float32)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		try:
			# Warm up UMAP_ApplyForce
			UMAP_ApplyForce(emb_d, attr_d, repl_d, N, NC)
			# Warm up UMAP_AttrForce (exact mode)
			UMAP_AttrForce(
				attr_d, emb_d, edgesrc_d, edgetgt_d, pos_eff_d,
				N, NC, np.float32(1.0), np.float32(1.0), np.float32(1.0),
				known_d, known_pos_d, known_rev_d, False, np.float32(0.1),
			)
			# Warm up UMAP_AttrForce_sampling (sampling mode — key for cold-start)
			UMAP_AttrForce_sampling(
				attr_d, emb_d, edgesrc_d, edgetgt_d, N,
				np.float32(1.0), np.float32(1.0), np.float32(1.0),
				np.int32(0), eps_d, eons_d,
				known_d, known_pos_d, known_rev_d, False, np.float32(0.1),
			)
		except Exception:
			pass


try:
	_warmup_optimizer_jit_kernels()
except Exception:
	pass
