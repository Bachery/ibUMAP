// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, the ibUMAP authors.
//
// Parts of this file are adapted from t-FDP (https://github.com/Ideas-Laboratory/t-fdp)
// with the permission of its author, and follow FIt-SNE's nbodyfft (MIT).
// See THIRD_PARTY_NOTICES.md.
//
// t-FDP utils

extern "C" __global__ void BoundsFinitePartials_cu(
	const float *Y,
	const unsigned char *active_mask,
	float *partials,
	int N,
	int use_active_mask)
{
	extern __shared__ float shared[];
	float *shared_min = shared;
	float *shared_max = shared + blockDim.x;
	float *shared_nonfinite = shared + 2 * blockDim.x;

	int tid = threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	float local_min = 3.402823466e+38F;
	float local_max = -3.402823466e+38F;
	float local_nonfinite = 0.0F;

	for (int i = blockIdx.x * blockDim.x + tid; i < N; i += stride)
	{
		float x = Y[2 * i];
		float y = Y[2 * i + 1];
		bool finite_x = isfinite(x);
		bool finite_y = isfinite(y);
		if (!finite_x) local_nonfinite += 1.0F;
		if (!finite_y) local_nonfinite += 1.0F;

		if (!use_active_mask || active_mask[i])
		{
			if (finite_x)
			{
				local_min = fminf(local_min, x);
				local_max = fmaxf(local_max, x);
			}
			if (finite_y)
			{
				local_min = fminf(local_min, y);
				local_max = fmaxf(local_max, y);
			}
		}
	}

	shared_min[tid] = local_min;
	shared_max[tid] = local_max;
	shared_nonfinite[tid] = local_nonfinite;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			shared_min[tid] = fminf(shared_min[tid], shared_min[tid + offset]);
			shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + offset]);
			shared_nonfinite[tid] += shared_nonfinite[tid + offset];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		int out = 3 * blockIdx.x;
		partials[out] = shared_min[0];
		partials[out + 1] = shared_max[0];
		partials[out + 2] = shared_nonfinite[0];
	}
}

extern "C" __global__ void BoundsFiniteFinalize_cu(
	const float *partials,
	float *result,
	int n_partials)
{
	extern __shared__ float shared[];
	float *shared_min = shared;
	float *shared_max = shared + blockDim.x;
	float *shared_nonfinite = shared + 2 * blockDim.x;

	int tid = threadIdx.x;
	float local_min = 3.402823466e+38F;
	float local_max = -3.402823466e+38F;
	float local_nonfinite = 0.0F;

	for (int i = tid; i < n_partials; i += blockDim.x)
	{
		int in = 3 * i;
		local_min = fminf(local_min, partials[in]);
		local_max = fmaxf(local_max, partials[in + 1]);
		local_nonfinite += partials[in + 2];
	}

	shared_min[tid] = local_min;
	shared_max[tid] = local_max;
	shared_nonfinite[tid] = local_nonfinite;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			shared_min[tid] = fminf(shared_min[tid], shared_min[tid + offset]);
			shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + offset]);
			shared_nonfinite[tid] += shared_nonfinite[tid + offset];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		result[0] = shared_min[0];
		result[1] = shared_max[0];
		result[2] = shared_nonfinite[0];
	}
}

extern "C" __global__ void AttrForce_cu(
	float *attr_force, float *dC, const float *pos, 
	const int *edgesrc, const int *edgetgt, int N, float beta, float d3alpha)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	// int cntTgt;
	float bias = 0.0;
	float mv0, mv1, d, R;
	for (int i = tid; i < N; i += xStride)
	{
		// int cntSrc = edgesrc[i + 1] - edgesrc[i];
		attr_force[2 * i] = 0.0;
		attr_force[2 * i + 1] = 0.0;
		for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
		{
			int j = edgetgt[k];
			// cntTgt = edgesrc[j + 1] - edgesrc[j];
			// bias = 1.0f * cntTgt / (cntTgt + cntSrc);
			mv0 = pos[2 * i] + dC[2 * i] - pos[2 * j] - dC[2 * j];
			mv1 = pos[2 * i + 1] + dC[2 * i + 1] - pos[2 * j + 1] - dC[2 * j + 1];
			d = sqrtf(mv0 * mv0 + mv1 * mv1);
			R = d3alpha * (1.0f * beta * powf(1.0f + d * d, -1.0f) + bias);
			// float dsqare = d * d;
			// float rr = beta / (1.0f + dsqare) + bias;
			// R = d3alpha * rr;
			// if (isinf(R) || isnan(R))
			// {
			// 	printf("R is nan or inf\n");
			// 	printf("d: %f, dsqare: %f, rr: %f\n", d, dsqare, rr);
			// 	printf("d3alpha: %f, beta: %f, bias: %f\n", d3alpha, beta, bias);
			// 	printf("R: %f\n", R);
			// }
			attr_force[2 * i] -= R * mv0;
			attr_force[2 * i + 1] -= R * mv1;
		}
		mv0 = attr_force[2 * i];
		mv1 = attr_force[2 * i+1];
		d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-12;
		R = d < 2000.0f ? d : 2000.0f;
		attr_force[2 * i] =  mv0 / d * R;
		attr_force[2 * i + 1] =  mv1 / d * R;
	}
}

extern "C" __global__ void AttrForce_sampling_effect_cu(
	float *attr_force, float *dC, const float *pos, 
	const int *edgesrc, const int *edgetgt, 
	int n_vertices, float beta, float d3alpha, float *pos_effects)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	// int cntTgt;
	float bias = 0.0;
	float mv0, mv1, d, R;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		// int cntSrc = edgesrc[i + 1] - edgesrc[i];
		attr_force[2 * i] = 0.0;
		attr_force[2 * i + 1] = 0.0;
		for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
		{
			int j = edgetgt[k];
			// cntTgt = edgesrc[j + 1] - edgesrc[j];
			// bias = 1.0f * cntTgt / (cntTgt + cntSrc);
			mv0 = pos[2 * i] + dC[2 * i] - pos[2 * j] - dC[2 * j];
			mv1 = pos[2 * i + 1] + dC[2 * i + 1] - pos[2 * j + 1] - dC[2 * j + 1];
			d = sqrtf(mv0 * mv0 + mv1 * mv1);
			R = d3alpha * (1.0f * beta * powf(1.0f + d * d, -1.0f) + bias);
			attr_force[2 * i] -= R * mv0 * pos_effects[k];
			attr_force[2 * i + 1] -= R * mv1 * pos_effects[k];
		}
		mv0 = attr_force[2 * i];
		mv1 = attr_force[2 * i+1];
		d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-12;
		R = d < 2000.0f ? d : 2000.0f;
		attr_force[2 * i] =  mv0 / d * R;
		attr_force[2 * i + 1] =  mv1 / d * R;
	}
}

extern "C" __global__ void AttrForce_sampling_cu(
	float *attr_force, float *dC, const float *pos, const int *edgesrc, const int *edgetgt, 
	int n_vertices, float beta, float d3alpha, 
	int epoch_itr, float *epochs_per_sample, float *epoch_of_next_sample)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	// int cntTgt;
	float bias = 0.0;
	float mv0, mv1, d, R;
	int ii, jj;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		ii = 2 * i;
		attr_force[ii] = 0.0;
		attr_force[ii + 1] = 0.0;
		for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
		{
			if (epoch_of_next_sample[k] <= epoch_itr)
			{
				epoch_of_next_sample[k] += epochs_per_sample[k];
				int j = edgetgt[k];
				jj = 2 * j;
				// cntTgt = edgesrc[j + 1] - edgesrc[j];
				// bias = 1.0f * cntTgt / (cntTgt + cntSrc);
				mv0 = pos[ii] + dC[ii] - pos[jj] - dC[jj];
				mv1 = pos[ii + 1] + dC[ii + 1] - pos[jj + 1] - dC[jj + 1];
				d = sqrtf(mv0 * mv0 + mv1 * mv1);
				R = d3alpha * (1.0f * beta * powf(1.0f + d * d, -1.0f) + bias);
				attr_force[ii] -= R * mv0;
				attr_force[ii + 1] -= R * mv1;
			}
		}
		mv0 = attr_force[ii];
		mv1 = attr_force[ii + 1];
		d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-12;
		R = d < 2000.0f ? d : 2000.0f;
		attr_force[ii] =  mv0 / d * R;
		attr_force[ii + 1] =  mv1 / d * R;
	}
}

extern "C" __global__ void RepulsiveForce_cu(float *dC, const float *pos, int N, float a, float b, float c, float alpha)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float mv0,mv1,dsqare,R;
	for (int i = tid; i < N; i += xStride)
	{
		for (int j = 0; j < N; j++)
		{
			mv0 = pos[2 * i] - pos[2 * j];
			mv1 = pos[2 * i + 1] - pos[2 * j + 1];
			dsqare = (mv0 * mv0 + mv1 * mv1) + 1e-24;
			R = alpha * a * powf(1.0 + dsqare * b, -c);
			dC[2 * i] += R * mv0;
			dC[2 * i + 1] += R * mv1;
		}
	}
}

extern "C" __global__ void ApplyForce_cu(const float *dC, float *pos, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i += xStride)
	{
		float mv0 = dC[2 * i];
		float mv1 = dC[2 * i + 1];
		float d = sqrtf(mv0 * mv0 + mv1 * mv1) + 1e-16;
		float R = d < 1.0f ? d : 1.0f;
		pos[2 * i] += mv0 / d * R;
		pos[2 * i + 1] += mv1 / d * R;
	}
}

// ibFFT utils

extern "C" __global__ void Y_in_box_cu(float *y_in_box, const float *Y, const int *box_idx, float *box_width, float *min_coord, int n_boxes_per_dim, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;

	for (int i = tid; i < N; i += xStride)
	{
		int j = box_idx[2 * i];
		int k = box_idx[2 * i + 1];
		y_in_box[2 * i] = (Y[2 * i] - *box_width * j - *min_coord);
		y_in_box[2 * i + 1] = (Y[2 * i + 1] - *box_width * k - *min_coord);
	}
}

extern "C" __global__ void Y_in_box_cu_bak(float *y_in_box, const float *Y, const long long *box_idx, const float *box_lower_bounds, int n_boxes_per_dim, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i += xStride)
	{
		int j = box_idx[2 * i];
		int k = box_idx[2 * i + 1];
		y_in_box[2 * i] = (Y[2 * i] - box_lower_bounds[2 * (j * n_boxes_per_dim + k)]);
		y_in_box[2 * i + 1] = (Y[2 * i + 1] - box_lower_bounds[2 * (j * n_boxes_per_dim + k) + 1]);
	}
}

extern "C" __global__ void Interpolate_cu(const float *y_in_box, const float *y_tilde_spacings, const float *denominator, float *interpolated_values, int n_interpolation_points, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points; j++)
		{
			interpolated_values[2 * (i * n_interpolation_points + j)] = 1.0;
			interpolated_values[2 * (i * n_interpolation_points + j) + 1] = 1.0;
			for (int k = 0; k < n_interpolation_points; k++)
			{
				if (j != k)
				{
					interpolated_values[2 * (i * n_interpolation_points + j)] *= y_in_box[2 * i] - y_tilde_spacings[k];
					interpolated_values[2 * (i * n_interpolation_points + j) + 1] *= y_in_box[2 * i + 1] - y_tilde_spacings[k];
				}
			}
			interpolated_values[2 * (i * n_interpolation_points + j)] /= denominator[j];
			interpolated_values[2 * (i * n_interpolation_points + j) + 1] /= denominator[j];
		}
	}
}

extern "C" __global__ void Denominator_cu(float *denominator, const float *y_tilde_spacings, int n_interpolation_points)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	denominator[tid] = 1;
	for (int j = 0; j < n_interpolation_points; j++)
	{
		if (tid != j)
			denominator[tid] *= y_tilde_spacings[tid] - y_tilde_spacings[j];
	}
}

extern "C" __global__ void Compute_w_coeff_cu(float *w_coefficients, const int *box_idx, const float *chargesQij, const float *interpolated_values, int n_interpolation_points, int n_boxes_per_dim, int n_terms, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int w_coeff_len_per_dim = n_interpolation_points * n_boxes_per_dim;
	for (int i = tid; i < N; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points; j++)
		{
			for (int k = 0; k < n_interpolation_points; k++)
			{
				int idj = box_idx[2 * i];
				int idk = box_idx[2 * i + 1];
				int point = (idj * n_interpolation_points + j) * w_coeff_len_per_dim + idk * n_interpolation_points + k;
				float prob = interpolated_values[2 * (i * n_interpolation_points + j)] * interpolated_values[2 * (i * n_interpolation_points + k) + 1];
				for (int term = 0; term < n_terms; term++)
				{
					atomicAdd(&(w_coefficients[n_terms * point + term]), prob * chargesQij[n_terms * i + term]);
				}
			}
		}
	}
}

extern "C" __global__ void Compute_w_coeff_p1_cu(float *w_coefficients, const int *box_idx, const float *chargesQij, int n_boxes_per_dim, int n_terms, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i += xStride)
	{
		int idj = box_idx[2 * i];
		int idk = box_idx[2 * i + 1];
		int point = idj * n_boxes_per_dim + idk;
		for (int term = 0; term < n_terms; term++)
		{
			atomicAdd(&(w_coefficients[n_terms * point + term]), chargesQij[n_terms * i + term]);
		}
	}
}

extern "C" __global__ void Compute_mat_w_p1_cu(float *mat_w, const int *box_idx, const float *Y, int n_fft_coeffs, int n_terms, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int term_stride = n_fft_coeffs * n_fft_coeffs;
	for (int i = tid; i < N; i += xStride)
	{
		int idj = box_idx[2 * i];
		int idk = box_idx[2 * i + 1];
		int point = idj * n_fft_coeffs + idk;
		atomicAdd(&(mat_w[point]), Y[2 * i]);
		if (n_terms > 1)
		{
			atomicAdd(&(mat_w[term_stride + point]), Y[2 * i + 1]);
		}
		if (n_terms > 2)
		{
			atomicAdd(&(mat_w[2 * term_stride + point]), 1.0f);
		}
	}
}

extern "C" __global__ void Compute_mat_w_p1_block_atomic_cu(float *mat_w, const int *box_idx, const float *Y, int n_fft_coeffs, int n_terms, int N)
{
	extern __shared__ unsigned char shared_raw[];
	int *keys = reinterpret_cast<int *>(shared_raw);
	float *values_x = reinterpret_cast<float *>(keys + blockDim.x);
	float *values_y = values_x + blockDim.x;
	int local = threadIdx.x;
	int i = blockIdx.x * blockDim.x + local;
	int key = -1;
	float value_x = 0.0f;
	float value_y = 0.0f;
	if (i < N)
	{
		int idj = box_idx[2 * i];
		int idk = box_idx[2 * i + 1];
		key = idj * n_fft_coeffs + idk;
		value_x = Y[2 * i];
		value_y = Y[2 * i + 1];
	}
	keys[local] = key;
	values_x[local] = value_x;
	values_y[local] = value_y;
	__syncthreads();

	if (key >= 0)
	{
		bool leader = true;
		for (int other = 0; other < local; ++other)
		{
			if (keys[other] == key) { leader = false; break; }
		}
		if (leader)
		{
			float total_x = value_x;
			float total_y = value_y;
			float total_count = 1.0f;
			for (int other = local + 1; other < blockDim.x; ++other)
			{
				if (keys[other] == key)
				{
					total_x += values_x[other];
					total_y += values_y[other];
					total_count += 1.0f;
				}
			}
			int term_stride = n_fft_coeffs * n_fft_coeffs;
			atomicAdd(&(mat_w[key]), total_x);
			if (n_terms > 1) atomicAdd(&(mat_w[term_stride + key]), total_y);
			if (n_terms > 2) atomicAdd(&(mat_w[2 * term_stride + key]), total_count);
		}
	}
}

extern "C" __global__ void Compute_mat_w_p1_segmented_cu(
	float *mat_w,
	const float *Y,
	const long long *ordered_points,
	const long long *segment_cells,
	const long long *segment_starts,
	const long long *segment_ends,
	int n_segments,
	int n_fft_coeffs,
	int n_terms)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int term_stride = n_fft_coeffs * n_fft_coeffs;
	for (int segment = tid; segment < n_segments; segment += stride)
	{
		long long cell = segment_cells[segment];
		float total_x = 0.0f;
		float total_y = 0.0f;
		float total_count = 0.0f;
		for (long long pos = segment_starts[segment]; pos < segment_ends[segment]; ++pos)
		{
			long long point = ordered_points[pos];
			total_x += Y[2 * point];
			total_y += Y[2 * point + 1];
			total_count += 1.0f;
		}
		mat_w[cell] = total_x;
		if (n_terms > 1) mat_w[term_stride + cell] = total_y;
		if (n_terms > 2) mat_w[2 * term_stride + cell] = total_count;
	}
}

extern "C" __global__ void Compute_mat_w_segmented_cu(
	float *mat_w,
	const float *Y,
	const float *interpolated_values,
	const long long *ordered_points,
	const long long *segment_boxes,
	const long long *segment_starts,
	const long long *segment_ends,
	int n_segments,
	int n_interpolation_points,
	int n_boxes_per_dim,
	int n_fft_coeffs,
	int n_terms)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int nodes_per_box = n_interpolation_points * n_interpolation_points;
	int n_outputs = n_segments * nodes_per_box;
	int term_stride = n_fft_coeffs * n_fft_coeffs;
	for (int output_index = tid; output_index < n_outputs; output_index += stride)
	{
		int segment = output_index / nodes_per_box;
		int local_node = output_index - segment * nodes_per_box;
		int j = local_node / n_interpolation_points;
		int k = local_node - j * n_interpolation_points;
		long long box = segment_boxes[segment];
		int box_x = static_cast<int>(box / n_boxes_per_dim);
		int box_y = static_cast<int>(box - static_cast<long long>(box_x) * n_boxes_per_dim);
		int mesh_x = box_x * n_interpolation_points + j;
		int mesh_y = box_y * n_interpolation_points + k;
		int mesh_cell = mesh_x * n_fft_coeffs + mesh_y;
		float total_x = 0.0f;
		float total_y = 0.0f;
		float total_weight = 0.0f;
		for (long long pos = segment_starts[segment]; pos < segment_ends[segment]; ++pos)
		{
			long long point = ordered_points[pos];
			long long x_weight_index = 2LL * (point * n_interpolation_points + j);
			long long y_weight_index = 2LL * (point * n_interpolation_points + k) + 1;
			// Match the reference atomic kernel's two rounded operations rather
			// than allowing an FMA to change the last bits of the reduction.
			float probability = __fmul_rn(
				interpolated_values[x_weight_index], interpolated_values[y_weight_index]);
			total_x = __fadd_rn(total_x, __fmul_rn(probability, Y[2LL * point]));
			total_y = __fadd_rn(total_y, __fmul_rn(probability, Y[2LL * point + 1]));
			total_weight = __fadd_rn(total_weight, probability);
		}
		mat_w[mesh_cell] = total_x;
		if (n_terms > 1) mat_w[term_stride + mesh_cell] = total_y;
		if (n_terms > 2) mat_w[2 * term_stride + mesh_cell] = total_weight;
	}
}

extern "C" __global__ void PotentialsQij_cu(float *PotentialsQij, const int *box_idx, const float *interpolated_values, const float *y_tilde_values, int n_interpolation_points, int n_boxes_per_dim, int n_terms, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim;

	for (int i = tid; i < N; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points; j++)
		{
			for (int k = 0; k < n_interpolation_points; k++)
			{
				int idj = box_idx[2 * i];
				int idk = box_idx[2 * i + 1];
				int point = (idj * n_interpolation_points + j + Qij_len_per_dim) * 2 * Qij_len_per_dim + Qij_len_per_dim + idk * n_interpolation_points + k;
				float prob = interpolated_values[2 * (i * n_interpolation_points + j)] * interpolated_values[2 * (i * n_interpolation_points + k) + 1];
				for (int term = 0; term < n_terms; term++)
				{
					PotentialsQij[i * n_terms + term] += prob * y_tilde_values[term * 4 * Qij_len_per_dim * Qij_len_per_dim + point];
				}
			}
		}
	}
}

extern "C" __global__ void PotentialsQij_p1_cu(float *PotentialsQij, const int *box_idx, const float *y_tilde_values, int n_boxes_per_dim, int n_terms, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int Qij_len_per_dim = n_boxes_per_dim;
	int y_tilde_stride = 4 * Qij_len_per_dim * Qij_len_per_dim;

	for (int i = tid; i < N; i += xStride)
	{
		int idj = box_idx[2 * i] + Qij_len_per_dim;
		int idk = box_idx[2 * i + 1] + Qij_len_per_dim;
		int point = idj * 2 * Qij_len_per_dim + idk;
		for (int term = 0; term < n_terms; term++)
		{
			PotentialsQij[i * n_terms + term] += y_tilde_values[term * y_tilde_stride + point];
		}
	}
}

extern "C" __global__ void NegF_p1_cu(float *neg_f, const int *box_idx, const float *Y, const float *y_tilde_values, int n_boxes_per_dim, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int Qij_len_per_dim = n_boxes_per_dim;
	int y_tilde_stride = 4 * Qij_len_per_dim * Qij_len_per_dim;

	for (int i = tid; i < N; i += xStride)
	{
		int idj = box_idx[2 * i] + Qij_len_per_dim;
		int idk = box_idx[2 * i + 1] + Qij_len_per_dim;
		int point = idj * 2 * Qij_len_per_dim + idk;
		float potential_x = y_tilde_values[point];
		float potential_y = y_tilde_values[y_tilde_stride + point];
		float potential_common = y_tilde_values[2 * y_tilde_stride + point];
		neg_f[2 * i] = potential_common * Y[2 * i] - potential_x;
		neg_f[2 * i + 1] = potential_common * Y[2 * i + 1] - potential_y;
	}
}

__device__ __forceinline__ void apply_fused_m2p_update(
	float *embedding,
	const float *attr_force,
	int i,
	float raw_x,
	float raw_y,
	float alpha,
	float neg_effect,
	int clip_enabled,
	float clip_norm)
{
	float force_x = raw_x * alpha;
	float force_y = raw_y * alpha;
	force_x *= neg_effect;
	force_y *= neg_effect;
	if (clip_enabled)
	{
		float norm = sqrtf(force_x * force_x + force_y * force_y);
		float scale = fminf(1.0f, clip_norm / (norm + 1e-12f));
		force_x *= scale;
		force_y *= scale;
	}
	embedding[2 * i] += attr_force[2 * i] + force_x;
	embedding[2 * i + 1] += attr_force[2 * i + 1] + force_y;
}

extern "C" __global__ void NegF_p1_fused_update_cu(
	float *embedding,
	const float *attr_force,
	const int *box_idx,
	const float *Y,
	const float *y_tilde_values,
	int n_boxes_per_dim,
	int N,
	float alpha,
	const float *neg_effects,
	int clip_enabled,
	float clip_norm)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int y_tilde_stride = 4 * n_boxes_per_dim * n_boxes_per_dim;
	for (int i = tid; i < N; i += xStride)
	{
		int idj = box_idx[2 * i] + n_boxes_per_dim;
		int idk = box_idx[2 * i + 1] + n_boxes_per_dim;
		int point = idj * 2 * n_boxes_per_dim + idk;
		float common = y_tilde_values[2 * y_tilde_stride + point];
		float raw_x = common * Y[2 * i] - y_tilde_values[point];
		float raw_y = common * Y[2 * i + 1] - y_tilde_values[y_tilde_stride + point];
		apply_fused_m2p_update(
			embedding, attr_force, i, raw_x, raw_y, alpha,
			neg_effects[i], clip_enabled, clip_norm);
	}
}

extern "C" __global__ void PotentialsQij_fused_update_cu(
	float *embedding,
	const float *attr_force,
	const int *box_idx,
	const float *Y,
	const float *interpolated_values,
	const float *y_tilde_values,
	int n_interpolation_points,
	int n_boxes_per_dim,
	int N,
	float alpha,
	const float *neg_effects,
	int clip_enabled,
	float clip_norm)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	int Qij_len_per_dim = n_interpolation_points * n_boxes_per_dim;
	int y_tilde_stride = 4 * Qij_len_per_dim * Qij_len_per_dim;
	for (int i = tid; i < N; i += xStride)
	{
		float potential_x = 0.0f;
		float potential_y = 0.0f;
		float potential_common = 0.0f;
		for (int j = 0; j < n_interpolation_points; j++)
		{
			for (int k = 0; k < n_interpolation_points; k++)
			{
				int idj = box_idx[2 * i] * n_interpolation_points + j + Qij_len_per_dim;
				int idk = box_idx[2 * i + 1] * n_interpolation_points + k + Qij_len_per_dim;
				int point = idj * 2 * Qij_len_per_dim + idk;
				float prob = interpolated_values[2 * (i * n_interpolation_points + j)]
					* interpolated_values[2 * (i * n_interpolation_points + k) + 1];
				potential_x += prob * y_tilde_values[point];
				potential_y += prob * y_tilde_values[y_tilde_stride + point];
				potential_common += prob * y_tilde_values[2 * y_tilde_stride + point];
			}
		}
		float raw_x = potential_common * Y[2 * i] - potential_x;
		float raw_y = potential_common * Y[2 * i + 1] - potential_y;
		apply_fused_m2p_update(
			embedding, attr_force, i, raw_x, raw_y, alpha,
			neg_effects[i], clip_enabled, clip_norm);
	}
}

extern "C" __global__ void Circulant_kernel_tilde_cu(
	float *circulant_kernel_tilde, float *box_width, int n_interpolation_points, 
	int n_interpolation_points_1d, int n_fft_coeffs, int n_boxes_per_dim, int N, float a, float c)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float whsquare = *box_width / n_interpolation_points;
	float tmp;
	whsquare = whsquare * whsquare;
	
	for (int i = tid; i < n_interpolation_points_1d; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points_1d; j++)
		{
			tmp = a * powf(1.0f + (i * i + j * j) * whsquare, -c);
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
		}
	}
}

extern "C" __global__ void Circulant_kernel_tilde_UMAP_cu(
	float *circulant_kernel_tilde, float *box_width, int n_interpolation_points, 
	int n_interpolation_points_1d, int n_fft_coeffs, 
	float umap_a, float umap_b, float umap_gamma, float umap_epsilon,
	float ibfft_kernel_clip)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float tmp, tmp1;
	float whsquare = *box_width / n_interpolation_points;
	whsquare = whsquare * whsquare;
	
	for (int i = tid; i < n_interpolation_points_1d; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points_1d; j++)
		{
			// tmp = a * powf(1.0f + (i * i + j * j) * whsquare, -c);
			tmp1 = (i * i + j * j) * whsquare;
			tmp = (2.0f * umap_gamma * umap_b) / ((umap_epsilon + tmp1) * (umap_a * powf(tmp1, umap_b) + 1.0f));
			// clip
			tmp = max(-ibfft_kernel_clip, min(ibfft_kernel_clip, tmp));
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
		}
	}
}

extern "C" __global__ void Collision_kernel_tilde_cu(float *circulant_kernel_tilde, float *box_width, int n_interpolation_points, int n_interpolation_points_1d, int n_fft_coeffs, int n_boxes_per_dim, int N, float Collision_size)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float whsquare = *box_width / n_interpolation_points;
	whsquare *= whsquare;

	for (int i = tid; i < n_interpolation_points_1d; i += xStride)
	{
		for (int j = 0; j < n_interpolation_points_1d; j++)
		{
			// float tmp = powf(1.0 + (i * i + j * j) * whsquare, -c);
			float tmp = sqrtf((i * i + j * j) * whsquare) + 1e-3;
			tmp = max(Collision_size - tmp,0.0) / tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
			circulant_kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
		}
	}
}

extern "C" __global__ void Box_idx_cu(int *box_idx, const float *Y, float *box_width, float *min_coord, int n_boxes_per_dim, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;

	for (int i = tid; i < N; i += xStride)
	{
		box_idx[2 * i] = max(0, min(int((Y[2 * i] - *min_coord) / (*box_width)), n_boxes_per_dim - 1));
		box_idx[2 * i + 1] = max(0, min(int((Y[2 * i + 1] - *min_coord) / (*box_width)), n_boxes_per_dim - 1));
	}
}

// UMAP utils

__device__ float clip(float val)
{
	if (val > 4.0) return 4.0;
	if (val < -4.0) return -4.0;
	return val;
}

extern "C" __global__ void UMAP_AttrForce_cu(
	float *attr_force, const float *embedding, const int *edgesrc, const int *edgetgt, 
	float *pos_effects, int n_vertices, float a, float b, float alpha)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float mv0, mv1, dsqare;
	float grad_coeff;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		float temp0 = 0.0, temp1 = 0.0;
		for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
		{
			int j = edgetgt[k];
			// bias = 1.0f * cntTgt / (cntTgt + cntSrc);
			mv0 = embedding[2 * i] - embedding[2 * j];
			mv1 = embedding[2 * i + 1] - embedding[2 * j + 1];
			dsqare = mv0 * mv0 + mv1 * mv1;
			if (dsqare > 0.0)
			{
				const float distance_power_b = powf(dsqare, b);
				grad_coeff = -2.0f * a * b * (distance_power_b / dsqare);
				grad_coeff /= a * distance_power_b + 1.0f;
				grad_coeff *= pos_effects[k];
			}
			else
			{
				grad_coeff = 0.0;
			}
			temp0 += alpha * clip(grad_coeff * mv0);
			temp1 += alpha * clip(grad_coeff * mv1);
		}
		attr_force[2 * i] = temp0;
		attr_force[2 * i + 1] = temp1;
	}
}

extern "C" __global__ void UMAP_AttrForce_sampling_cu(
	float *attr_force, const float *embedding, const int *edgesrc, const int *edgetgt, 
	int n_vertices, float a, float b, float alpha,
	int epoch_itr, float *epochs_per_sample, float *epoch_of_next_sample)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	float mv0, mv1, dsqare;
	float grad_coeff;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		float temp0 = 0.0, temp1 = 0.0;
		for (int k = edgesrc[i]; k < edgesrc[i + 1]; k++)
		{
			if (epoch_of_next_sample[k] <= epoch_itr)
			{
				epoch_of_next_sample[k] += epochs_per_sample[k];
				int j = edgetgt[k];
				// bias = 1.0f * cntTgt / (cntTgt + cntSrc);
				mv0 = embedding[2 * i] - embedding[2 * j];
				mv1 = embedding[2 * i + 1] - embedding[2 * j + 1];
				dsqare = mv0 * mv0 + mv1 * mv1;
				if (dsqare > 0.0)
				{
					const float distance_power_b = powf(dsqare, b);
					grad_coeff = -2.0f * a * b * (distance_power_b / dsqare);
					grad_coeff /= a * distance_power_b + 1.0f;
				}
				else
				{
					grad_coeff = 0.0;
				}
				temp0 += alpha * clip(grad_coeff * mv0);
				temp1 += alpha * clip(grad_coeff * mv1);
				// if (isinf(temp0) || isnan(temp0) || isinf(temp1) || isnan(temp1))
				// {
				// 	printf("temp0 or temp1 is nan or inf\n");
				// 	printf("temp0: %f, temp1: %f\n", temp0, temp1);
				// 	printf("grad_coeff: %f, mv0: %f, mv1: %f\n", grad_coeff, mv0, mv1);
				// 	printf("dsqare: %f\n", dsqare);
				// 	printf("a: %f, b: %f\n", a, b);
				// 	printf("alpha: %f\n", alpha);
				// }
			}
		}
		attr_force[2 * i] = temp0;
		attr_force[2 * i + 1] = temp1;
	}
}

__inline__ __device__ float ibumap_warp_sum(float value)
{
	for (int offset = 16; offset > 0; offset >>= 1)
	{
		value += __shfl_down_sync(0xffffffffu, value, offset);
	}
	return value;
}

extern "C" __global__ void UMAP_AttrForce_warp_per_row_cu(
	float *attr_force, const float *embedding, const int *edgesrc, const int *edgetgt,
	const float *pos_effects, int n_vertices, float a, float b, float alpha)
{
	const int lane = threadIdx.x & 31;
	const int warp_in_block = threadIdx.x >> 5;
	const int warps_per_block = blockDim.x >> 5;
	const int row_stride = gridDim.x * warps_per_block;
	int row = blockIdx.x * warps_per_block + warp_in_block;

	for (int i = row; i < n_vertices; i += row_stride)
	{
		float temp0 = 0.0f;
		float temp1 = 0.0f;
		for (int k = edgesrc[i] + lane; k < edgesrc[i + 1]; k += 32)
		{
			const int j = edgetgt[k];
			const float mv0 = embedding[2 * i] - embedding[2 * j];
			const float mv1 = embedding[2 * i + 1] - embedding[2 * j + 1];
			const float dsqare = mv0 * mv0 + mv1 * mv1;
			float grad_coeff;
			if (dsqare > 0.0f)
			{
				const float distance_power_b = powf(dsqare, b);
				grad_coeff = -2.0f * a * b * (distance_power_b / dsqare);
				grad_coeff /= a * distance_power_b + 1.0f;
				grad_coeff *= pos_effects[k];
			}
			else
			{
				grad_coeff = 0.0f;
			}
			temp0 += alpha * clip(grad_coeff * mv0);
			temp1 += alpha * clip(grad_coeff * mv1);
		}
		temp0 = ibumap_warp_sum(temp0);
		temp1 = ibumap_warp_sum(temp1);
		if (lane == 0)
		{
			attr_force[2 * i] = temp0;
			attr_force[2 * i + 1] = temp1;
		}
	}
}

extern "C" __global__ void UMAP_AttrForce_sampling_warp_per_row_cu(
	float *attr_force, const float *embedding, const int *edgesrc, const int *edgetgt,
	int n_vertices, float a, float b, float alpha,
	int epoch_itr, float *epochs_per_sample, float *epoch_of_next_sample)
{
	const int lane = threadIdx.x & 31;
	const int warp_in_block = threadIdx.x >> 5;
	const int warps_per_block = blockDim.x >> 5;
	const int row_stride = gridDim.x * warps_per_block;
	int row = blockIdx.x * warps_per_block + warp_in_block;

	for (int i = row; i < n_vertices; i += row_stride)
	{
		float temp0 = 0.0f;
		float temp1 = 0.0f;
		for (int k = edgesrc[i] + lane; k < edgesrc[i + 1]; k += 32)
		{
			if (epoch_of_next_sample[k] <= epoch_itr)
			{
				epoch_of_next_sample[k] += epochs_per_sample[k];
				const int j = edgetgt[k];
				const float mv0 = embedding[2 * i] - embedding[2 * j];
				const float mv1 = embedding[2 * i + 1] - embedding[2 * j + 1];
				const float dsqare = mv0 * mv0 + mv1 * mv1;
				float grad_coeff;
				if (dsqare > 0.0f)
				{
					const float distance_power_b = powf(dsqare, b);
					grad_coeff = -2.0f * a * b * (distance_power_b / dsqare);
					grad_coeff /= a * distance_power_b + 1.0f;
				}
				else
				{
					grad_coeff = 0.0f;
				}
				temp0 += alpha * clip(grad_coeff * mv0);
				temp1 += alpha * clip(grad_coeff * mv1);
			}
		}
		temp0 = ibumap_warp_sum(temp0);
		temp1 = ibumap_warp_sum(temp1);
		if (lane == 0)
		{
			attr_force[2 * i] = temp0;
			attr_force[2 * i + 1] = temp1;
		}
	}
}

extern "C" __global__ void UMAP_ApplyForce_cu(
	float *embedding, const float *attr_force, const float *repl_force, const int n_vertices)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		embedding[2 * i] += attr_force[2 * i] + repl_force[2 * i];
		embedding[2 * i + 1] += attr_force[2 * i + 1] + repl_force[2 * i + 1];
	}
}

extern "C" __global__ void UMAP_ReplForcePostprocess_cu(
	float *repl_force,
	const float *neg_effects,
	float neg_effect_scalar,
	const bool *whether_known_points,
	int use_vector_neg_effects,
	int use_hard_mask,
	int use_clip,
	float clip_norm,
	float alpha,
	int n_vertices)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int xStride = blockDim.x * gridDim.x;
	for (int i = tid; i < n_vertices; i += xStride)
	{
		float scale = alpha;
		if (use_vector_neg_effects)
		{
			scale *= neg_effects[i];
		}
		else
		{
			scale *= neg_effect_scalar;
		}
		float fx = repl_force[2 * i] * scale;
		float fy = repl_force[2 * i + 1] * scale;
		if (use_hard_mask && whether_known_points[i])
		{
			fx = 0.0f;
			fy = 0.0f;
		}
		if (use_clip)
		{
			float norm = sqrtf(fx * fx + fy * fy);
			float clip_scale = fminf(1.0f, clip_norm / (norm + 1e-12f));
			fx *= clip_scale;
			fy *= clip_scale;
		}
		repl_force[2 * i] = fx;
		repl_force[2 * i + 1] = fy;
	}
}
