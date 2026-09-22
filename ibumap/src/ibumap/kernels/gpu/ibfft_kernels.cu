extern "C" {

// ------------------------------------------------------------------
// Helper: Compute Lagrange interpolation weights (Registers)
// ------------------------------------------------------------------
__device__ void compute_interpolation_weights(
	float* weights,          // Output weights (length n_interp)
	float relative_coord,    // Relative coordinate in box
	const float* spacings,   // y_tilde_spacings
	const float* denominators, // denominators
	int n_interp
) {
	for (int j = 0; j < n_interp; ++j) {
		weights[j] = 1.0f;
		for (int k = 0; k < n_interp; ++k) {
			if (j != k) {
				weights[j] *= (relative_coord - spacings[k]);
			}
		}
		weights[j] /= denominators[j];
	}
}

// ------------------------------------------------------------------
// 1. Kernel Generation: UMAP Circulant Kernel
// ------------------------------------------------------------------
__global__ void Circulant_kernel_tilde_UMAP_cu(
	float* kernel,           // Output (2*H, 2*W)
	float box_width,
	int n_interp,
	int grid_dim,            // n_interpolation_points_1d (H)
	int fft_dim,             // 2 * H
	float umap_a,
	float umap_b,
	float umap_gamma,
	float umap_epsilon,
	float ibfft_kernel_clip
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= grid_dim || idy >= grid_dim) return;

	// Calculate squared distance between grid points
	float whsquare = (box_width / n_interp) * (box_width / n_interp);
	float dist_sq = (idx * idx + idy * idy) * whsquare;

	// UMAP Kernel Formula
	float val = (2.0f * umap_gamma * umap_b) / 
				((umap_epsilon + dist_sq) * (umap_a * powf(dist_sq, umap_b) + 1.0f));
	
	// Clipping to prevent numerical instability
	val = fmaxf(-ibfft_kernel_clip, fminf(ibfft_kernel_clip, val));

	// Fill the four quadrants (Symmetric padding)
	// Right-Bottom (Original)
	kernel[(grid_dim + idx) * fft_dim + (grid_dim + idy)] = val;
	// Right-Top (Vertical Flip)
	kernel[(grid_dim - idx) * fft_dim + (grid_dim + idy)] = val;
	// Left-Bottom (Horizontal Flip)
	kernel[(grid_dim + idx) * fft_dim + (grid_dim - idy)] = val;
	// Left-Top (Full Flip)
	kernel[(grid_dim - idx) * fft_dim + (grid_dim - idy)] = val;
}

// ------------------------------------------------------------------
// 2. P2M Fused: Particle -> Mesh (BoxIdx + Interpolate + Scatter)
// ------------------------------------------------------------------
__global__ void P2M_Fused_cu(
	float* grid_aos,             // Output Grid (grid_dim, grid_dim, 3) - AoS Layout
	const float* Y,              // Particle Coords (N, 2)
	const float* y_tilde_spacings,
	const float* denominators,
	int N,
	int n_interp,
	int n_boxes,
	float box_width,
	float min_coord,
	int grid_dim                 // n_boxes * n_interp
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	// 1. Load Particle (Registers)
	float px = Y[2 * i];
	float py = Y[2 * i + 1];

	// 2. Compute Box Index
	int bx = (int)((px - min_coord) / box_width);
	int by = (int)((py - min_coord) / box_width);
	bx = max(0, min(bx, n_boxes - 1));
	by = max(0, min(by, n_boxes - 1));

	// 3. Compute Relative Position in Box
	float dx = px - min_coord - bx * box_width;
	float dy = py - min_coord - by * box_width;

	// 4. Compute Interpolation Weights (Registers)
	float w_x[10]; 
	float w_y[10];
	compute_interpolation_weights(w_x, dx, y_tilde_spacings, denominators, n_interp);
	compute_interpolation_weights(w_y, dy, y_tilde_spacings, denominators, n_interp);

	// 5. Scatter to Grid (Atomic Add)
	// Charge is [px, py, 1.0]
	for (int j = 0; j < n_interp; ++j) {
		int gx = bx * n_interp + j;
		for (int k = 0; k < n_interp; ++k) {
			int gy = by * n_interp + k;
			float prob = w_x[j] * w_y[k];

			// Grid Address (AoS: 3 channels adjacent)
			int addr = (gx * grid_dim + gy) * 3;

			atomicAdd(&grid_aos[addr + 0], prob * px);
			atomicAdd(&grid_aos[addr + 1], prob * py);
			atomicAdd(&grid_aos[addr + 2], prob * 1.0f);
		}
	}
}

// ------------------------------------------------------------------
// 3. M2P Fused: Mesh -> Particle (Interpolate + Force Calculation)
// ------------------------------------------------------------------
__global__ void M2P_Fused_cu(
	float* forces,               // Output Forces (N, 2)
	const float* Y,              // Particle Coords
	const float* potentials_soa, // Input Potentials (3, 2*grid_dim, 2*grid_dim) - SoA Layout
	const float* y_tilde_spacings,
	const float* denominators,
	int N,
	int n_interp,
	int n_boxes,
	float box_width,
	float min_coord,
	int grid_dim,                // n_boxes * n_interp
	int fft_dim                  // 2 * grid_dim
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float px = Y[2 * i];
	float py = Y[2 * i + 1];

	// Recompute Box Index and Weights
	int bx = (int)((px - min_coord) / box_width);
	int by = (int)((py - min_coord) / box_width);
	bx = max(0, min(bx, n_boxes - 1));
	by = max(0, min(by, n_boxes - 1));

	float dx = px - min_coord - bx * box_width;
	float dy = py - min_coord - by * box_width;

	float w_x[10];
	float w_y[10];
	compute_interpolation_weights(w_x, dx, y_tilde_spacings, denominators, n_interp);
	compute_interpolation_weights(w_y, dy, y_tilde_spacings, denominators, n_interp);

	// Gather Potentials
	float pot_x = 0.0f;
	float pot_y = 0.0f;
	float pot_c = 0.0f;

	// Potentials Layout: (3, fft_dim, fft_dim)
	// The FFT calculation typically results in a SoA (Sort of Areas) layout.
	// Channel 0: X potentials
	// Channel 1: Y potentials
	// Channel 2: Common (Charge 1) potentials
	
	// Offset applied in python/CPU version (Qij_len_per_dim) is handled here naturally
	// because we map particle to [0, grid_dim] inside the padded [0, fft_dim] array.
	// However, the convolution result keeps the "valid" part in the corners or center?
	// t-FDP logic: The result aligns such that index [i] in space corresponds to [i] in potential.
	// Note: Python code used offset `Qij_len_per_dim` when reading back. 
	// Usually standard conv requires reading from [grid_dim, 2*grid_dim].
	// Let's assume standard cyclic conv padding logic:
	// With `circulant` kernel construction, the valid potential is at indices [n, 2n].
	// Let's match the Python logic: `idj + Qij_len_per_dim`.
	
	int read_offset = grid_dim; 

	for (int j = 0; j < n_interp; ++j) {
		int gx = bx * n_interp + j + read_offset;
		for (int k = 0; k < n_interp; ++k) {
			int gy = by * n_interp + k + read_offset;
			
			float prob = w_x[j] * w_y[k];
			int idx_in_plane = gx * fft_dim + gy;
			int plane_stride = fft_dim * fft_dim;

			// Read 3 channels (SoA)
			pot_x += potentials_soa[0 * plane_stride + idx_in_plane] * prob;
			pot_y += potentials_soa[1 * plane_stride + idx_in_plane] * prob;
			pot_c += potentials_soa[2 * plane_stride + idx_in_plane] * prob;
		}
	}

	// Force Calculation: neg_f = PotentialsCom * Y - PotentialsXY
	float fx = pot_c * px - pot_x;
	float fy = pot_c * py - pot_y;

	forces[2 * i + 0] = fx;
	forces[2 * i + 1] = fy;
}

}
