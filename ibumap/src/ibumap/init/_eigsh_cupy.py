"""CuPy ``eigsh`` adapter with explicit CSR SpMV algorithm selection.

The implementation follows CuPy 13.3's thick-restart Lanczos solver while
making the cuSPARSE SpMV algorithm explicit. The deterministic production
route uses CSR ALG2, while the faster route explicitly uses CSR ALG1 instead
of relying on ``CUSPARSE_MV_ALG_DEFAULT``.

The code is intentionally kept private to the CUDA spectral initializer. It
relies on CuPy's low-level CUDA bindings and should be exercised by the
fixed-Laplacian compatibility experiment whenever the supported CuPy/CUDA
matrix changes.

Adapted from CuPy v13.3.0 ``cupyx.scipy.sparse.linalg._eigen``:

Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from cupy import cublas
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx import cusparse
from cupyx.scipy import sparse as cupyx_sparse


_CSR_ALG2_NAMES = (
    "CUSPARSE_SPMV_CSR_ALG2",
    "CUSPARSE_CSRMV_ALG2",
)
_CSR_ALG1_NAMES = (
    "CUSPARSE_SPMV_CSR_ALG1",
    "CUSPARSE_CSRMV_ALG1",
)


def _resolve_csr_spmv_algorithm(
    names: tuple[str, ...],
    *,
    description: str,
) -> tuple[int, str]:
    for name in names:
        if hasattr(_cusparse, name):
            return int(getattr(_cusparse, name)), name
    expected = ", ".join(names)
    raise RuntimeError(
        "The installed CuPy/cuSPARSE bindings do not expose "
        f"{description} ({expected})."
    )


def deterministic_csr_spmv_algorithm() -> tuple[int, str]:
    """Return CuPy's available deterministic CSR SpMV enum and its name."""

    return _resolve_csr_spmv_algorithm(
        _CSR_ALG2_NAMES,
        description="a deterministic CSR SpMV algorithm",
    )


def fast_csr_spmv_algorithm() -> tuple[int, str]:
    """Return CuPy's available non-deterministic CSR ALG1 enum and its name."""

    return _resolve_csr_spmv_algorithm(
        _CSR_ALG1_NAMES,
        description="the fast CSR ALG1 SpMV algorithm",
    )


def eigsh_csr_alg2(
    matrix,
    k: int = 6,
    *,
    which: str = "LM",
    v0=None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    return_eigenvectors: bool = True,
    _spmv_algorithm: tuple[int, str] | None = None,
):
    """Solve a real symmetric CSR eigenproblem with deterministic SpMV.

    This is the subset of :func:`cupyx.scipy.sparse.linalg.eigsh` required by
    ibUMAP's normalized-Laplacian initialization. The Lanczos protocol matches
    CuPy 13.3, but every sparse matrix-vector multiplication explicitly uses
    cuSPARSE CSR ALG2.
    """

    if not cupyx_sparse.isspmatrix_csr(matrix):
        raise TypeError("explicit-algorithm CUDA eigsh requires a CSR matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if matrix.dtype.char not in "fd":
        raise TypeError(
            "explicit-algorithm CUDA eigsh supports real float32/float64 "
            "matrices, "
            f"got {matrix.dtype}"
        )

    n = int(matrix.shape[0])
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")
    if k >= n:
        raise ValueError(f"k must be smaller than matrix size {n}, got {k}")
    if which not in ("LM", "LA", "SA"):
        raise ValueError(f"which must be 'LM', 'LA', or 'SA', got {which!r}")

    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(int(ncv), k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    else:
        maxiter = int(maxiter)
    if tol == 0:
        tol = float(np.finfo(matrix.dtype).eps)
    else:
        tol = float(tol)

    alpha = cp.zeros((ncv,), dtype=matrix.dtype)
    beta = cp.zeros((ncv,), dtype=matrix.dtype)
    lanczos_vectors = cp.empty((ncv, n), dtype=matrix.dtype)

    if v0 is None:
        initial = cp.random.random((n,)).astype(matrix.dtype)
    else:
        initial = cp.asarray(v0, dtype=matrix.dtype)
        if initial.shape != (n,):
            raise ValueError(f"v0 must have shape {(n,)}, got {initial.shape}")
    work = cp.array(initial, dtype=matrix.dtype, copy=True)
    lanczos_vectors[0] = work / cublas.nrm2(work)

    if _spmv_algorithm is None:
        _spmv_algorithm = deterministic_csr_spmv_algorithm()

    lanczos = _lanczos_fast_csr_alg2(
        matrix,
        n,
        ncv,
        spmv_algorithm=_spmv_algorithm,
    )
    lanczos(
        matrix,
        lanczos_vectors,
        work,
        alpha,
        beta,
        0,
        ncv,
    )

    iteration = ncv
    eigenvalues, ritz_vectors = _solve_ritz(alpha, beta, None, k, which)
    eigenvectors = lanczos_vectors.T @ ritz_vectors
    beta_k = beta[-1] * ritz_vectors[-1, :]
    residual = cublas.nrm2(beta_k)
    projection = cp.empty((k,), dtype=matrix.dtype)

    while float(residual) > tol and iteration < maxiter:
        beta[:k] = 0
        alpha[:k] = eigenvalues
        lanczos_vectors[:k] = eigenvectors.T

        cublas.gemv(
            _cublas.CUBLAS_OP_C,
            1,
            lanczos_vectors[:k].T,
            work,
            0,
            projection,
        )
        cublas.gemv(
            _cublas.CUBLAS_OP_N,
            -1,
            lanczos_vectors[:k].T,
            projection,
            1,
            work,
        )
        lanczos_vectors[k] = work / cublas.nrm2(work)

        _spmv_csr_alg2(
            matrix,
            lanczos_vectors[k],
            work,
            spmv_algorithm=_spmv_algorithm,
        )
        cublas.dotc(lanczos_vectors[k], work, out=alpha[k])
        work -= alpha[k] * lanczos_vectors[k]
        work -= lanczos_vectors[:k].T @ beta_k
        cublas.nrm2(work, out=beta[k])
        lanczos_vectors[k + 1] = work / beta[k]

        lanczos(
            matrix,
            lanczos_vectors,
            work,
            alpha,
            beta,
            k + 1,
            ncv,
        )

        iteration += ncv - k
        eigenvalues, ritz_vectors = _solve_ritz(
            alpha,
            beta,
            beta_k,
            k,
            which,
        )
        eigenvectors = lanczos_vectors.T @ ritz_vectors
        beta_k = beta[-1] * ritz_vectors[-1, :]
        residual = cublas.nrm2(beta_k)

    if return_eigenvectors:
        order = cp.argsort(eigenvalues)
        return eigenvalues[order], eigenvectors[:, order]
    return cp.sort(eigenvalues)


def eigsh_csr_alg1(
    matrix,
    k: int = 6,
    *,
    which: str = "LM",
    v0=None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    return_eigenvectors: bool = True,
):
    """Solve a real symmetric CSR eigenproblem with explicit CSR ALG1 SpMV."""

    return eigsh_csr_alg2(
        matrix,
        k,
        which=which,
        v0=v0,
        ncv=ncv,
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=return_eigenvectors,
        _spmv_algorithm=fast_csr_spmv_algorithm(),
    )


def _spmv_csr_alg2(
    matrix,
    vector,
    output,
    *,
    spmv_algorithm: tuple[int, str] | None = None,
) -> None:
    """Compute ``output = matrix @ vector`` with an explicit CSR algorithm."""

    cusparse_handle = device.get_cusparse_handle()
    operation = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    alpha = np.array(1.0, matrix.dtype)
    beta = np.array(0.0, matrix.dtype)
    cuda_dtype = _dtype.to_cuda_dtype(matrix.dtype)
    if spmv_algorithm is None:
        spmv_algorithm = deterministic_csr_spmv_algorithm()
    algorithm, _ = spmv_algorithm
    matrix_descriptor = cusparse.SpMatDescriptor.create(matrix)
    vector_descriptor = cusparse.DnVecDescriptor.create(vector)
    output_descriptor = cusparse.DnVecDescriptor.create(output)
    buffer_size = _cusparse.spMV_bufferSize(
        cusparse_handle,
        operation,
        alpha.ctypes.data,
        matrix_descriptor.desc,
        vector_descriptor.desc,
        beta.ctypes.data,
        output_descriptor.desc,
        cuda_dtype,
        algorithm,
    )
    buffer = cp.empty(buffer_size, cp.int8)
    _cusparse.spMV(
        cusparse_handle,
        operation,
        alpha.ctypes.data,
        matrix_descriptor.desc,
        vector_descriptor.desc,
        beta.ctypes.data,
        output_descriptor.desc,
        cuda_dtype,
        algorithm,
        buffer.data.ptr,
    )


def _lanczos_fast_csr_alg2(
    matrix,
    n: int,
    ncv: int,
    *,
    spmv_algorithm: tuple[int, str] | None = None,
):
    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
    if matrix.dtype.char == "f":
        dotc = _cublas.sdot
        nrm2 = _cublas.snrm2
        gemv = _cublas.sgemv
        axpy = _cublas.saxpy
    elif matrix.dtype.char == "d":
        dotc = _cublas.ddot
        nrm2 = _cublas.dnrm2
        gemv = _cublas.dgemv
        axpy = _cublas.daxpy
    else:  # guarded by eigsh_csr_alg2
        raise TypeError(f"unsupported dtype {matrix.dtype}")

    if not cusparse.check_availability("spmv"):
        raise RuntimeError(
            "The installed CUDA toolkit does not provide generic cuSPARSE SpMV"
        )

    cusparse_handle = device.get_cusparse_handle()
    spmv_operation = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    spmv_alpha = np.array(1.0, matrix.dtype)
    spmv_beta = np.array(0.0, matrix.dtype)
    spmv_cuda_dtype = _dtype.to_cuda_dtype(matrix.dtype)
    if spmv_algorithm is None:
        spmv_algorithm = deterministic_csr_spmv_algorithm()
    spmv_algorithm_enum, _ = spmv_algorithm

    vector = cp.empty((n,), dtype=matrix.dtype)
    projection = cp.empty((ncv,), dtype=matrix.dtype)
    correction = cp.empty((n,), dtype=matrix.dtype)
    previous_beta = cp.empty((), dtype=matrix.dtype)
    one = np.array(1.0, dtype=matrix.dtype)
    zero = np.array(0.0, dtype=matrix.dtype)
    minus_one = np.array(-1.0, dtype=matrix.dtype)

    def run(
        active_matrix,
        lanczos_vectors,
        work,
        alpha,
        beta,
        start: int,
        end: int,
    ) -> None:
        if active_matrix is not matrix:
            raise ValueError(
                "Lanczos matrix changed during explicit-algorithm eigsh"
            )

        matrix_descriptor = cusparse.SpMatDescriptor.create(matrix)
        vector_descriptor = cusparse.DnVecDescriptor.create(vector)
        work_descriptor = cusparse.DnVecDescriptor.create(work)
        buffer_size = _cusparse.spMV_bufferSize(
            cusparse_handle,
            spmv_operation,
            spmv_alpha.ctypes.data,
            matrix_descriptor.desc,
            vector_descriptor.desc,
            spmv_beta.ctypes.data,
            work_descriptor.desc,
            spmv_cuda_dtype,
            spmv_algorithm_enum,
        )
        spmv_buffer = cp.empty(buffer_size, cp.int8)

        vector[...] = lanczos_vectors[start]
        for index in range(start, end):
            _cusparse.spMV(
                cusparse_handle,
                spmv_operation,
                spmv_alpha.ctypes.data,
                matrix_descriptor.desc,
                vector_descriptor.desc,
                spmv_beta.ctypes.data,
                work_descriptor.desc,
                spmv_cuda_dtype,
                spmv_algorithm_enum,
                spmv_buffer.data.ptr,
            )

            _cublas.setPointerMode(
                cublas_handle,
                _cublas.CUBLAS_POINTER_MODE_DEVICE,
            )
            try:
                dotc(
                    cublas_handle,
                    n,
                    vector.data.ptr,
                    1,
                    work.data.ptr,
                    1,
                    alpha.data.ptr + index * alpha.itemsize,
                )
            finally:
                _cublas.setPointerMode(
                    cublas_handle,
                    cublas_pointer_mode,
                )

            correction.fill(0)
            _cublas.setPointerMode(
                cublas_handle,
                _cublas.CUBLAS_POINTER_MODE_DEVICE,
            )
            try:
                axpy(
                    cublas_handle,
                    n,
                    alpha.data.ptr + index * alpha.itemsize,
                    vector.data.ptr,
                    1,
                    correction.data.ptr,
                    1,
                )
                if index > 0:
                    previous_beta[...] = beta[index - 1]
                    axpy(
                        cublas_handle,
                        n,
                        previous_beta.data.ptr,
                        lanczos_vectors[index - 1].data.ptr,
                        1,
                        correction.data.ptr,
                        1,
                    )
            finally:
                _cublas.setPointerMode(
                    cublas_handle,
                    cublas_pointer_mode,
                )
            axpy(
                cublas_handle,
                n,
                minus_one.ctypes.data,
                correction.data.ptr,
                1,
                work.data.ptr,
                1,
            )

            gemv(
                cublas_handle,
                _cublas.CUBLAS_OP_C,
                n,
                index + 1,
                one.ctypes.data,
                lanczos_vectors.data.ptr,
                n,
                work.data.ptr,
                1,
                zero.ctypes.data,
                projection.data.ptr,
                1,
            )
            gemv(
                cublas_handle,
                _cublas.CUBLAS_OP_N,
                n,
                index + 1,
                minus_one.ctypes.data,
                lanczos_vectors.data.ptr,
                n,
                projection.data.ptr,
                1,
                one.ctypes.data,
                work.data.ptr,
                1,
            )
            alpha[index] += projection[index]

            _cublas.setPointerMode(
                cublas_handle,
                _cublas.CUBLAS_POINTER_MODE_DEVICE,
            )
            try:
                nrm2(
                    cublas_handle,
                    n,
                    work.data.ptr,
                    1,
                    beta.data.ptr + index * beta.itemsize,
                )
            finally:
                _cublas.setPointerMode(
                    cublas_handle,
                    cublas_pointer_mode,
                )

            if index >= end - 1:
                break
            _normalize_kernel(
                work,
                beta,
                index,
                n,
                vector,
                lanczos_vectors,
            )

    return run


_normalize_kernel = cp.ElementwiseKernel(
    "T work, raw S beta, int32 index, int32 n",
    "T vector, raw T lanczos_vectors",
    "vector = work / beta[index]; "
    "lanczos_vectors[i + (index + 1) * n] = vector;",
    "umap_fft_eigsh_normalize",
)


def _solve_ritz(alpha, beta, beta_k, k: int, which: str):
    alpha_cpu = cp.asnumpy(alpha)
    beta_cpu = cp.asnumpy(beta)
    tridiagonal = np.diag(alpha_cpu)
    tridiagonal += np.diag(beta_cpu[:-1], k=1)
    tridiagonal += np.diag(beta_cpu[:-1], k=-1)
    if beta_k is not None:
        beta_k_cpu = cp.asnumpy(beta_k)
        tridiagonal[k, :k] = beta_k_cpu
        tridiagonal[:k, k] = beta_k_cpu

    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    if which == "LA":
        order = np.argsort(eigenvalues)[-k:]
    elif which == "LM":
        order = np.argsort(np.abs(eigenvalues))[-k:]
    else:
        order = np.argsort(eigenvalues)[:k]
    return cp.asarray(eigenvalues[order]), cp.asarray(eigenvectors[:, order])
