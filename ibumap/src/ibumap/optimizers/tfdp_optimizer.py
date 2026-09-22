import platform
from ..kernels.cpu.ibfft import *
from ..fft_schedule import (
	fft_stage_index_for_epoch,
	format_resolved_fft_schedule,
	resolve_fft_schedule,
	validate_fft_schedule_execution,
)
# condition: if current system is Linux, import the GPU version
# if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
cupy = None
if 'Linux' in platform.platform():
	try:
		import cupy
		from ..kernels.gpu.cupy_ibfft import grid_dim, block_dim, cuda_raw_text
		from ..kernels.gpu.cupy_ibfft import ibFFT_repulsive_sampling_GPU
		AttrForce_cu					= cupy.RawKernel(cuda_raw_text, 'AttrForce_cu')
		AttrForce_sampling_effect_cu	= cupy.RawKernel(cuda_raw_text, 'AttrForce_sampling_effect_cu')
		AttrForce_sampling_cu			= cupy.RawKernel(cuda_raw_text, 'AttrForce_sampling_cu')
		# RepulsiveForce_cu				= cupy.RawKernel(cuda_raw_text, 'RepulsiveForce_cu')
		ApplyForce_cu					= cupy.RawKernel(cuda_raw_text, 'ApplyForce_cu')
	except Exception:
		cupy = None


""" Controller """

def tfdp_optimization(
	embedding,
	edgesrc, 
	edgetgt, 
	weights,
	degrees,
	n_epochs,
	n_vertices,
	rng_state,
	epochs_per_sample,
	epochs_per_negative_sample,
	negative_sample_rate,
	tfdp_algo,
	drfft_params,
):
	if "GPU" in tfdp_algo and cupy is None:
		raise RuntimeError("GPU optimization requested but cupy/CUDA kernels are unavailable")
	numeric_dtype = np.dtype(drfft_params.get("numeric_dtype", np.float32))
	if numeric_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
		raise ValueError("numeric_dtype must be float32 or float64")
	if "GPU" in tfdp_algo and numeric_dtype == np.dtype(np.float64):
		raise NotImplementedError(
			"dtype='float64' is currently supported on CPU only; CUDA kernels use float32"
		)

	t1 = time()
	epoch_of_next_sample = epochs_per_sample.copy()
	epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
	alpha		= drfft_params['tfdp_alpha']
	beta		= drfft_params['tfdp_beta']
	gamma		= drfft_params['tfdp_gamma']
	paraFactor	= drfft_params['tfdp_paraFactor']	# repulsion strength
	n_itrp_pts	= drfft_params['n_interpolation_points']
	resolved_fft_stages = resolve_fft_schedule(
		n_epochs=int(n_epochs),
		n_interpolation_points=n_itrp_pts,
		combine_stages=drfft_params.get('tfdp_combine', False),
		interpolation_schedule=drfft_params.get('interpolation_schedule'),
	)
	p2m_mode = drfft_params.get('p2m_mode', 'auto')
	if 'ibFFT' in tfdp_algo:
		validate_fft_schedule_execution(
			resolved_fft_stages,
			device=('cuda' if 'GPU' in tfdp_algo else 'cpu'),
			p2m_mode=p2m_mode,
		)
	itv_int		= drfft_params['intervals_per_integer']
	mn_itv		= drfft_params['min_num_intervals']
	boxes_dim	= drfft_params['n_boxes_per_dim']
	
	if alpha != 0: paraFactor /= alpha	# for keeping a same long range force
	E = edgetgt.shape[0]
	d3alpha = 1.0						# a smaller d3alpha for higher average degrees
	if E/n_vertices >= 15.0: d3alpha /= 10.0
	if E/n_vertices >= 50.0: d3alpha /= 10.0
	d3alphaMin = d3alpha / 100.0
	#TODO: seed?

	d3alpha_step = 0.012955423246736264	# 1 - pow(0.02, 1 / 300)

	### Positive and Negative Sampling Effect
	pos_effects = weights
	neg_effect, neg_effects, probabilities = 0.0, None, None
	if tfdp_algo == 'tFDP_ibFFT_CPU_sampling' or tfdp_algo == 'tFDP_ibFFT_GPU_sampling':
		neg_effect = 10 * negative_sample_rate / n_vertices
		probabilities = degrees / degrees.max()
	elif tfdp_algo == 'tFDP_ibFFT_CPU_sampling_effect' or tfdp_algo == 'tFDP_ibFFT_GPU_sampling_effect':
		neg_effects = degrees * 0.5 * negative_sample_rate / n_vertices

	### Repulsion Kernel
	kernel_method = 'UMAP_gauss' if drfft_params['repl_gauss'] else 'tFDP'

	### Constraints
	whether_known_points		= drfft_params['whether_known_points']
	known_points_positions		= drfft_params['known_points_positions']
	known_points_reverse_index	= drfft_params['known_points_reverse_index']
	constraint_weight			= drfft_params['constraint_weight']
	soft_constraint				= drfft_params['soft_constraint']
	hard_constraint				= (True in whether_known_points) and (not soft_constraint)

	### Parameters for GPU
	if "GPU" not in tfdp_algo:
		embedding = np.asarray(embedding, dtype=numeric_dtype, order="C")
		pos_effects = np.asarray(pos_effects, dtype=numeric_dtype)
		if neg_effects is not None:
			neg_effects = np.asarray(neg_effects, dtype=numeric_dtype)
		if probabilities is not None:
			probabilities = np.asarray(probabilities, dtype=numeric_dtype)
		dC = np.zeros((n_vertices, 2), dtype=numeric_dtype)			# 每个点受到的合力 Directional Changes
		attr_force = np.zeros((n_vertices, 2), dtype=numeric_dtype)	# 每个点受到的的吸引力
		repl_force = np.zeros((n_vertices, 2), dtype=numeric_dtype)	# 每个点受到的排斥力
		bias = np.zeros(edgetgt.shape[0], dtype=numeric_dtype)
	else:
		dC = cupy.zeros((n_vertices, 2), dtype=cupy.float32)
		attr_force = cupy.zeros((n_vertices, 2), dtype=cupy.float32)
		repl_force = cupy.zeros((n_vertices, 2), dtype=cupy.float32)
		n_vertices = cupy.int32(n_vertices)
		epochs_per_sample = cupy.array(epochs_per_sample, dtype=cupy.float32)
		epoch_of_next_sample = cupy.array(epoch_of_next_sample, dtype=cupy.float32)
		embedding = cupy.array(embedding, dtype=cupy.float32)
		edgesrc = cupy.array(edgesrc, dtype=cupy.int32)
		edgetgt = cupy.array(edgetgt, dtype=cupy.int32)
		beta = cupy.float32(beta)
		d3alpha = cupy.float32(d3alpha)
		d3alphaMin = cupy.float32(d3alphaMin)
		d3alpha_step = cupy.float32(d3alpha_step)
		gamma = cupy.float32(gamma)
		paraFactor = cupy.float32(paraFactor)
		pos_effects = cupy.array(pos_effects, dtype=cupy.float32)
		neg_effect = cupy.float32(neg_effect)
		if neg_effects is not None: neg_effects = cupy.array(neg_effects, dtype=cupy.float32)
		if probabilities is not None: probabilities = cupy.array(probabilities, dtype=cupy.float32)


	attr_time = []
	repl_time = []
	appl_time = []
	fft_stage_epoch_indices = []
	opt_prep_time = time() - t1

	for epoch_itr in range(n_epochs):
	### Attractive Force
		t1 = time()
		# if 'GPU' not in tfdp_algo:	attr_force = np.zeros((n_vertices, 2), dtype=np.float32)
		# else:						attr_force = cupy.zeros((n_vertices, 2), dtype=cupy.float32)
		if tfdp_algo == 'tFDP_Exact' or tfdp_algo == 'tFDP_ibFFT_CPU':
			AttrForce(
				attr_force, dC, embedding, edgesrc, edgetgt, bias, n_vertices, beta, d3alpha
			)
		elif tfdp_algo == 'tFDP_ibFFT_GPU':# or tfdp_algo == 'tFDP_ibFFT_GPU_sampling_effect':
			AttrForce_cu(grid_dim, block_dim, (
				attr_force, dC, embedding, edgesrc, edgetgt, n_vertices, beta, d3alpha
			))
		elif tfdp_algo == 'tFDP_ibFFT_CPU_sampling_effect':
			AttrForce_sampling_effect(
				attr_force, dC, embedding, edgesrc, edgetgt, bias, n_vertices, beta, d3alpha, pos_effects
			)
		elif tfdp_algo == 'tFDP_ibFFT_GPU_sampling_effect':
			AttrForce_sampling_effect_cu(grid_dim, block_dim, (
				attr_force, dC, embedding, edgesrc, edgetgt, n_vertices, beta, d3alpha, pos_effects
			))
		elif tfdp_algo == 'tFDP_Exact_sampling' or tfdp_algo == 'tFDP_ibFFT_CPU_sampling':
			AttrForce_sampling(
				attr_force, dC, embedding, edgesrc, edgetgt, bias, n_vertices, beta, d3alpha, 
				epoch_itr, epochs_per_sample, epoch_of_next_sample,
				whether_known_points, known_points_positions, 
				known_points_reverse_index, soft_constraint, constraint_weight,
			)
		#TODO
		elif tfdp_algo == 'tFDP_ibFFT_GPU_sampling':
			AttrForce_sampling_cu(grid_dim, block_dim, (
				attr_force, dC, embedding, edgesrc, edgetgt, n_vertices, beta, d3alpha, 
				epoch_itr, epochs_per_sample, epoch_of_next_sample,
				# whether_known_points, known_points_positions, 
				# known_points_reverse_index, soft_constraint, constraint_weight,
			))
		if hard_constraint:
			attr_force[whether_known_points, 0] = 0.0
			attr_force[whether_known_points, 1] = 0.0
		# if cupy.isnan(attr_force).any():
		# 	print('NaN exists in attr_force')
		# 	print(epoch_itr)
		dC += attr_force
		if 'GPU' in tfdp_algo: 
			dC -= 0.01 * d3alpha * cupy.random.randn(embedding.shape[0], embedding.shape[1], dtype=cupy.float32)
		else: 
			dC -= 0.01 * d3alpha * np.random.normal(0, 1, size=embedding.shape).astype(numeric_dtype)
		attr_time.append(time() - t1)

	### Repulsive Force
		t1 = time()
		fft_stage_index = fft_stage_index_for_epoch(
			resolved_fft_stages, epoch_itr
		)
		fft_stage_epoch_indices.append(fft_stage_index)
		n_itrp_pts = resolved_fft_stages[
			fft_stage_index
		].n_interpolation_points
		if tfdp_algo == 'tFDP_Exact':
			repl_force = np.zeros((n_vertices, 2), dtype=numeric_dtype)
			ReplForce(
				repl_force, embedding, n_vertices, paraFactor, gamma, d3alpha
			)
		elif tfdp_algo == 'tFDP_Exact_sampling':
			repl_force = np.zeros((n_vertices, 2), dtype=numeric_dtype)
			ReplForce_sampling(
				repl_force, embedding, edgesrc, n_vertices, 
				paraFactor, gamma, d3alpha, epoch_itr, rng_state, 
				epochs_per_sample, epoch_of_next_sample,
				epochs_per_negative_sample, epoch_of_next_negative_sample
			)
		elif tfdp_algo == 'tFDP_ibFFT_CPU':
			repl_force = ibFFT_repulsive_sampling(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, None, boxes_dim, p2m_mode=p2m_mode
			)
		elif tfdp_algo == 'tFDP_ibFFT_GPU':
			repl_force = ibFFT_repulsive_sampling_GPU(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, None, boxes_dim, p2m_mode=p2m_mode
			)
		elif tfdp_algo == 'tFDP_ibFFT_CPU_sampling_effect':
			repl_force = ibFFT_repulsive_sampling(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, None, boxes_dim, kernel_method, 
				drfft_params['umap_a'], drfft_params['umap_b'], 
				drfft_params['umap_gamma'], drfft_params['umap_epsilon'],
				drfft_params['gauss_sigma'],
				p2m_mode=p2m_mode,
			)
			repl_force[:, 0] *= neg_effects
			repl_force[:, 1] *= neg_effects
		elif tfdp_algo == 'tFDP_ibFFT_GPU_sampling_effect':
			repl_force = ibFFT_repulsive_sampling_GPU(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, None, boxes_dim, kernel_method, 
				drfft_params['umap_a'], drfft_params['umap_b'], 
				drfft_params['umap_gamma'], drfft_params['umap_epsilon'],
				drfft_params['gauss_sigma'],
				p2m_mode=p2m_mode,
			)
			repl_force[:, 0] *= neg_effects
			repl_force[:, 1] *= neg_effects
		elif tfdp_algo == 'tFDP_ibFFT_CPU_sampling':
			repl_force = ibFFT_repulsive_sampling(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, probabilities, boxes_dim, kernel_method, 
				drfft_params['umap_a'], drfft_params['umap_b'], 
				drfft_params['umap_gamma'], drfft_params['umap_epsilon'],
				drfft_params['gauss_sigma'],
				p2m_mode=p2m_mode,
			)
			repl_force *= neg_effect
		elif tfdp_algo == 'tFDP_ibFFT_GPU_sampling':
			repl_force = ibFFT_repulsive_sampling_GPU(
				embedding, n_itrp_pts, itv_int, mn_itv, gamma, 
				paraFactor, probabilities, boxes_dim, kernel_method, 
				drfft_params['umap_a'], drfft_params['umap_b'], 
				drfft_params['umap_gamma'], drfft_params['umap_epsilon'],
				drfft_params['gauss_sigma'],
				p2m_mode=p2m_mode,
			)
			repl_force *= neg_effect
		if hard_constraint:
			repl_force[whether_known_points, 0] = 0.0
			repl_force[whether_known_points, 1] = 0.0
		# check if NaN exists
		# if cupy.isnan(repl_force).any():
		# 	print('NaN exists in repl_force')
		# 	print(epoch_itr)
		dC += d3alpha * repl_force
		repl_time.append(time() - t1)

	### Apply Force
		t1 = time()
		if "GPU" not in tfdp_algo:
			ApplyForce(embedding, dC, n_vertices)
		else:
			ApplyForce_cu(grid_dim, block_dim, (dC, embedding, n_vertices))
		d3alpha += (d3alphaMin - d3alpha) * d3alpha_step		# 1 - pow(0.02, 1 / 300)
		# pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min())/2	# move to center
		# pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min())/2
		dC *= 0.6
		appl_time.append(time() - t1)

	time_costs = {
		'opt_prep_time': opt_prep_time,
		'attr_time': np.sum(attr_time),
		'repl_time': np.sum(repl_time),
		'appl_time': np.sum(appl_time),
		'fft_stage_schedule': format_resolved_fft_schedule(resolved_fft_stages),
		'fft_stage_count': int(len(resolved_fft_stages)),
		'fft_stage_transition_count': int(max(0, len(resolved_fft_stages) - 1)),
		'fft_stage_transition_epochs': ','.join(
			str(stage.start_epoch) for stage in resolved_fft_stages[1:]
		),
	}
	for stage_index, stage in enumerate(resolved_fft_stages):
		order = int(stage.n_interpolation_points)
		stage_repl_time = sum(
			float(elapsed)
			for index, elapsed in zip(fft_stage_epoch_indices, repl_time)
			if index == stage_index
		)
		time_costs[f'fft_stage_epochs_p{order}'] = int(stage.n_epochs)
		time_costs[f'fft_stage_repl_time_p{order}'] = float(stage_repl_time)
	
	if 'GPU' in tfdp_algo: embedding = embedding.get()

	return embedding, time_costs
