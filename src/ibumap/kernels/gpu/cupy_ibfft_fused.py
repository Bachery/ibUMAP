import cupy as cp
import numpy as np
import os

# 编译或加载 CUDA Kernel
curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_dir, 'ibfft_kernels.cu'), 'r', encoding='utf-8') as f:
	cuda_source = f.read()

module = cp.RawModule(code=cuda_source)
kernel_gen = module.get_function('Circulant_kernel_tilde_UMAP_cu')
p2m_kernel = module.get_function('P2M_Fused_cu')
m2p_kernel = module.get_function('M2P_Fused_cu')

def ibFFT_repulsive_forces_GPU(
	Y,                          # (N, 2) cupy array or numpy array
	n_interpolation_points=3,
	intervals_per_integer=1.0,
	min_num_intervals=50,
	n_boxes_per_dim=None,       # Optional manual override
	umap_a=1.57,
	umap_b=0.895,
	umap_gamma=1.0,
	umap_epsilon=0.001,
	ibfft_kernel_clip=1.0,
):
	"""
	Computes UMAP repulsive forces using GPU-accelerated ibFFT.
	Returns: forces (N, 2) as cupy array.
	"""
	# 0. 确保数据在 GPU 上
	if not isinstance(Y, cp.ndarray):
		Y = cp.asarray(Y, dtype=cp.float32)
	
	N = Y.shape[0]
	min_coord = float(cp.min(Y))
	max_coord = float(cp.max(Y))
	
	# 1. 计算网格参数
	# 如果没指定 n_boxes，使用 heuristic 计算
	if n_boxes_per_dim is None:
		boxes_1 = np.sqrt(16 * N)
		boxes_2 = np.sqrt(4 * N / np.log(N))
		boxes_3 = (max_coord - min_coord) / intervals_per_integer
		n_temp = int(max(min_num_intervals, boxes_3, min(boxes_1, boxes_2)))
		
		# 对齐到 2/3/5 的倍数以便 FFT 高效计算 (简化版：直接对齐到2的倍数或常见尺寸)
		# 这里为了简单，向上取偶数，CuPy FFT 对任意尺寸都很快，但 2^n 最好
		if n_temp % 2 != 0: n_temp += 1
		n_boxes_per_dim = n_temp
	
	box_width = (max_coord - min_coord) / n_boxes_per_dim
	grid_dim = n_boxes_per_dim * n_interpolation_points
	fft_dim = 2 * grid_dim
	
	# 2. 准备插值参数 (Spacings & Denominators)
	h = box_width / n_interpolation_points
	# y_tilde_spacings = [h/2, h + h/2, ...]
	y_tilde_spacings = cp.arange(n_interpolation_points, dtype=cp.float32) * h + h/2.0
	
	# 计算 Denominators (仅需计算一次，量很小)
	denominators = cp.ones(n_interpolation_points, dtype=cp.float32)
	# 简单的 CPU 计算再传 GPU 也可以，因为 n_interp 很小
	spacings_cpu = cp.asnumpy(y_tilde_spacings)
	denom_cpu = np.ones(n_interpolation_points, dtype=np.float32)
	for j in range(n_interpolation_points):
		for k in range(n_interpolation_points):
			if j != k:
				denom_cpu[j] *= (spacings_cpu[j] - spacings_cpu[k])
	denominators = cp.asarray(denom_cpu)

	# 3. 生成核 (Kernel Generation)
	# kernel 需要是 (fft_dim, fft_dim)
	kernel_tilde = cp.zeros((fft_dim, fft_dim), dtype=cp.float32)
	
	threads_per_block_2d = (16, 16)
	blocks_x = (grid_dim + 15) // 16
	blocks_y = (grid_dim + 15) // 16
	
	kernel_gen((blocks_x, blocks_y), threads_per_block_2d, (
		kernel_tilde, 
		cp.float32(box_width), 
		np.int32(n_interpolation_points), 
		np.int32(grid_dim), 
		np.int32(fft_dim),
		cp.float32(umap_a), cp.float32(umap_b), 
		cp.float32(umap_gamma), cp.float32(umap_epsilon),
		cp.float32(ibfft_kernel_clip),
	))
	
	# FFT Kernel
	fft_kernel = cp.fft.rfft2(kernel_tilde)

	# 4. P2M: 粒子 -> 网格
	# Grid AoS: (Grid, Grid, 3)
	grid_aos = cp.zeros((grid_dim, grid_dim, 3), dtype=cp.float32)
	
	threads_per_block = 128
	blocks = (N + 127) // 128
	
	p2m_kernel((blocks,), (threads_per_block,), (
		grid_aos, Y, y_tilde_spacings, denominators,
		np.int32(N), np.int32(n_interpolation_points), 
		np.int32(n_boxes_per_dim), cp.float32(box_width), cp.float32(min_coord),
		np.int32(grid_dim)
	))
	
	# 5. Convolution (FFT)
	# 转置为 SoA: (3, Grid, Grid) 适合 Batch FFT
	grid_soa = grid_aos.transpose(2, 0, 1)
	
	# 这里的 pad 是为了匹配 kernel 的尺寸 (fft_dim, fft_dim)
	# 我们需要在右下角 pad 0 到 (2*Grid, 2*Grid)
	# cupy.fft.rfft2 可以自动 pad，通过指定 s 参数
	fft_grid = cp.fft.rfft2(grid_soa, s=(fft_dim, fft_dim), axes=(1, 2))
	
	# 频域乘法 (Broadcasting: (3, H, W) * (H, W))
	fft_grid *= fft_kernel
	
	# Inverse FFT
	# irfft2 输出 (3, fft_dim, fft_dim)
	potentials_soa = cp.fft.irfft2(fft_grid, s=(fft_dim, fft_dim), axes=(1, 2))
	
	# 6. M2P: 网格 -> 粒子 & 力计算
	forces = cp.empty((N, 2), dtype=cp.float32)
	
	m2p_kernel((blocks,), (threads_per_block,), (
		forces, Y, potentials_soa, y_tilde_spacings, denominators,
		np.int32(N), np.int32(n_interpolation_points), 
		np.int32(n_boxes_per_dim), cp.float32(box_width), cp.float32(min_coord),
		np.int32(grid_dim), np.int32(fft_dim)
	))
	
	return forces

# 使用示例
if __name__ == "__main__":
	# 模拟数据
	N = 10000
	Y_cpu = np.random.rand(N, 2).astype(np.float32) * 10.0
	
	# 预热 GPU
	ibFFT_repulsive_forces_GPU(Y_cpu[:100])
	
	import time
	t0 = time.time()
	forces = ibFFT_repulsive_forces_GPU(Y_cpu)
	cp.cuda.Stream.null.synchronize()
	print(f"GPU Time: {time.time() - t0:.4f}s")
	print("Forces shape:", forces.shape)
	print("First 5 forces:\n", forces[:5])
