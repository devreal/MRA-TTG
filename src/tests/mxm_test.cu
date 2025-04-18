#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cassert>


#include "mra/ops/mxm.h"

// Kernel for matrix multiplication (mTxm)
__global__ void mTxmKernel(const double* A, const double* B, double* C, int M, int N, int K) {
    mra::mTxmq(M, N, K, C, A, B);
}

void test_mTxm() {
    // Matrix dimensions
    const int M = 64; // Rows of A^T and C
    const int N = 8;  // Columns of A^T and rows of B
    const int K = 8;  // Columns of B and C

    // Host matrices
    std::vector<double> h_A(M*K);      // A is MxK
    std::vector<double> h_B(K*N);      // B is NxK
    std::vector<double> h_C(M * N, -1.0); // C is MxN
    std::vector<double> h_C_blas(M * N, -1.0);

    for (int i = 0; i < M; ++i) {
        h_A[i*K] = 2.0;
        for (int j = 1; j < K; ++j) {
            h_A[i*K+j] = 1.0; // Initialize A with 1.0
        }
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0; // Initialize B with 1.0
    }

    // Device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(double));
    cudaMalloc(&d_B, N * K * sizeof(double));
    cudaMalloc(&d_C, M * N * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDims = mra::max_thread_dims(K);
    mTxmKernel<<<1, blockDims, mra::mTxmq_shmem_size<double>(K)>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Use cuBLAS for comparison
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0f;
    const double beta = 0.0f;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, d_A, M, d_B, N, &beta, d_C, M);
    cudaMemcpy(h_C_blas.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Validate results
    bool correct = true;
    for (int i = 0; i < M; ++i) {
	for (int j = 0; j < N; ++j) {
        if (fabs(h_C[i*N+j] - h_C_blas[j*N+i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << h_C[i] << " != " << h_C_blas[i] << std::endl;
            correct = false;
            assert(false);
        }
	}
    }

    if (correct) std::cout << "Test passed!" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    test_mTxm();
    return 0;
}
