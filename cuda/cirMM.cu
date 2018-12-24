#include "cirMM.h"

void cirMM(float* W, float* At, float* V, int M, int N, int K, int batch){
    float *d_V = NULL;
    float *d_At = NULL;
    float *d_W = NULL;
    const float alpha = 1;
    const float beta = 0;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    /*
    cout << "V is  \n";
    for (int j=0; j<K; j++){
        for (int i=0; i<N; i++){
            cout << V[i*K+j] << " ";
        }
        cout << endl;
    }
    */
    cudaStat1 = cudaMalloc((void**)&d_At, sizeof(float) *M*K*batch);
    cudaStat2 = cudaMalloc((void**)&d_V, sizeof(float) *K*N);
    cudaStat3 = cudaMalloc((void**)&d_W, sizeof(float) *M*N*batch);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpy(d_At, At, sizeof(float) *M*K*batch, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_V, V, sizeof(float) *N*K, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasSgemmStridedBatched(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            M,
            N,
            K,
            &alpha,
            d_At,
            M,
            M*K,
            d_V,
            N,
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

    if (d_At    ) cudaFree(d_At);
    if (d_W    ) cudaFree(d_W);
    if (d_V    ) cudaFree(d_V);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();
}
