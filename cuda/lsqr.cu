#include "lsqr.h"
//solve A*x = b where size(A) m*n size(x) n*1 size(b) m*1
void lsqr(float* A, float* X, float* B, int m, int n){
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors


    /*float A[lda*m] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
//    float X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
    float B[ldb*nrhs] = { 6.0, 15.0, 4.0};
    float XC[ldb*nrhs]; // solution matrix from GPU
    */

    float *d_A = NULL; // linear memory of GPU
    float *d_tau = NULL; // linear memory of GPU
    float *d_B  = NULL;
    int *devInfo = NULL; // info in gpu (device copy)
    float *d_work = NULL;
    int  lwork = 0;

    const float one = 1;
// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float) * lda * n);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float) * n);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(float) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

     
// step 3: query working space of geqrf and ormqr
    cusolver_status = cusolverDnSgeqrf_bufferSize(
        cusolverH, 
        m, 
        n, 
        d_A, 
        lda, 
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
 
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
    cusolver_status = cusolverDnSgeqrf(
        cusolverH, 
        m, 
        n, 
        d_A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    //cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    //assert(cudaSuccess == cudaStat1);

// step 5: compute Q^T*B
    cusolver_status= cusolverDnSormqr(
        cusolverH, 
        CUBLAS_SIDE_LEFT, 
        CUBLAS_OP_T,
        m, 
        nrhs, 
        n, 
        d_A, 
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

// step 6: compute x = R \ Q^T*B

    cublas_status = cublasStrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);
    //transfer data back to CPU from GPU？
    cudaStat1 = cudaMemcpy(X, d_B, sizeof(float)*n*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);


    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();
}
