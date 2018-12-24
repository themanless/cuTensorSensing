#include "initialY.h"
void cuGemv(float* A, float* X, float* y, int m, int n){
    float *d_X = NULL;
    float *d_A = NULL;
    float *d_y = NULL;
    const float alpha = 1;
    const float beta = 0;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) *m*n);
    cudaStat2 = cudaMalloc((void**)&d_X, sizeof(float) *n);
    cudaStat3 = cudaMalloc((void**)&d_y, sizeof(float) *m);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) *m*n, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_X, X, sizeof(float) *n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasSgemv(
            cublasH,
            CUBLAS_OP_N,
            m,
            n,
            &alpha,
            d_A,
            m,
            d_X,
            1,
            &beta,
            d_y,
            1
            );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    if (d_A    ) cudaFree(d_A);
    if (d_y    ) cudaFree(d_y);
    if (d_X    ) cudaFree(d_X);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();
   
}
void cuGemm(float* A, float* B, float* C, int m, int n, int k){
    float *d_B = NULL;
    float *d_A = NULL;
    float *d_C = NULL;
    const float alpha = 1;
    const float beta = 0;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) *m*k);
    cudaStat2 = cudaMalloc((void**)&d_B, sizeof(float) *n*k);
    cudaStat3 = cudaMalloc((void**)&d_C, sizeof(float) *m*n);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) *m*k, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) *n*k, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasSgemm(
            cublasH,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            d_A,
            k,
            d_B,
            k,
            &beta,
            d_C,
            m
            );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(C, d_C, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    if (d_A    ) cudaFree(d_A);
    if (d_C    ) cudaFree(d_C);
    if (d_B    ) cudaFree(d_B);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();
   
}
