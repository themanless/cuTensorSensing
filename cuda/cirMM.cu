#include "cirMM.h"

void cirMM(float* W, float* A, float* U, int m, int n, int r, int k, int d){
    float* Um = new float[m*k*r*k];
    for (int i=0; i<r; i++)
        for (int j=0; j<m; j++)
            for (int a=0; a<k; a++)
                for (int b=0; b<k; b++){
                    Um[(i*k+a)*m*k+j*k+b] = U[(i*m+j)*k+(b-a+k)%k];
                }
    float *d_U = NULL;
    float *d_A = NULL;
    float *d_W = NULL;
    const float alpha = 1;
    const float beta = 0;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    int K = m*k, M = d, N = r*k, batch = n;
    cout << "Um is  \n";
    for (int j=0; j<K; j++){
        for (int i=0; i<N; i++){
            cout << Um[i*K+j] << " ";
        }
        cout << endl;
    }
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) *M*K*batch);
    cudaStat2 = cudaMalloc((void**)&d_U, sizeof(float) *K*N);
    cudaStat3 = cudaMalloc((void**)&d_W, sizeof(float) *M*N*batch);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) *M*K*batch, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_U, Um, sizeof(float) *N*K, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasSgemmStridedBatched(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M,
            N,
            K,
            &alpha,
            d_A,
            M,
            M*K,
            d_U,
            K,
            0,
            &beta,
            d_W,
            M,
            M*N,
            batch
            );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(float)*M*N*batch, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (d_U    ) cudaFree(d_U);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();
}
