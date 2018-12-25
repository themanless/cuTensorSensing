#include <string>
#include "warmup.h"
#include "../cirMM.h"
#include "../lsqr.h"
#include "initialY.h"
#include <iostream>
#include <ctime>
#include <cstdio>
#include <cstdlib>

using namespace std;
void TS(float* X, float* A, float* At, float* y, int d, int IterNum, int m, int n, int r, int k){
    float* Um = (float*)malloc(m*r*k*sizeof(float));
    float* V = (float*)malloc(n*r*k*sizeof(float));
    //float* Um = new float[m*r*k];
    //float* V = new float[n*r*k];
    for(int i=0; i<m*r*k; i++){
        Um[i] = rand()%(1000) / (float) (1000);
    }
    float* Ut = (float*)malloc(m*k*r*k*sizeof(float));
    //float* Ut = new float[m*k*r*k];
    for (int i=0; i<m; i++)
        for (int j=0; j<r; j++)
            for (int a=0; a<k; a++)
                for (int b=0; b<k; b++){
                    Ut[(i*k+a)*r*k+j*k+b] = Um[(i*r+j)*k+(a-b+k)%k];
                }

    //malloc the prime data on device
    float* d_A = NULL;
    float* d_W = NULL;
    float* d_y = NULL;
    float* d_yL = NULL;
    float* d_At = NULL;
    const float alpha = 1;
    const float beta = 0;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    /*
    cout << "V is  \n";
    for (int j=0; j<m*k; j++){
        for (int i=0; i<r*k; i++){
            cout << V[i*m*k+j] << " ";
        }
        cout << endl;
    }
    */
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) *d*m*k*n);
    cudaStat3 = cudaMalloc((void**)&d_At, sizeof(float) *d*m*k*n);
    cudaStat2 = cudaMalloc((void**)&d_y, sizeof(float) *d);
    cudaStat4 = cudaMalloc((void**)&d_yL, sizeof(float) *d);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    cudaStat2 = cudaMemcpy(d_yL, y, sizeof(float) * d, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) *d*m*k*n, cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(d_At, At, sizeof(float) *d*m*k*n, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_y, Ut, sizeof(float) *m*k*r*k, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMalloc((void**)&d_W, sizeof(float) *d*m*r*k*k);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat4);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    int *devInfo = NULL; // info in gpu (device copy)
    float *d_work = NULL;
    float *d_tau = NULL; // linear memory of GPU
    int  lwork = 0;

    const float one = 1;
// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float) * m*r*k*k);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat4);

    float* Xr = (float*)malloc(n*m*k*sizeof(float));
    for (int i=0; i<IterNum; i++){
        //malloc d_W in LS_V
        cublas_status = cublasSgemmStridedBatched(
                cublasH,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                d,
                r*k,
                m*k,
                &alpha,
                d_A,
                d,
                d*m*k,
                d_y,
                r*k,
                0,
                &beta,
                d_W,
                d,
                d*r*k,
                n
                );
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        assert(cudaSuccess == cudaStat1);

        cusolver_status = cusolverDnSgeqrf_bufferSize(
            cusolverH,
            d,
            r*k*n,
            d_W,
            d,
            &lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
        assert(cudaSuccess == cudaStat1);

    // step 4: compute QR factorization
    cudaStat2 = cudaMemcpy(d_y, d_yL, sizeof(float) * d, cudaMemcpyDeviceToDevice);
    assert(cudaSuccess == cudaStat2);
        cusolver_status = cusolverDnSgeqrf(
            cusolverH,
            d,
            r*k*n,
            d_W,
            d,
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
            d,
            1,
            r*k*n,
            d_W,
            d,
            d_tau,
            d_y,
            d,
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
             r*k*n,
             1,
             &one,
             d_W,
             d,
             d_y,
             d);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        assert(cudaSuccess == cudaStat1);
        if (i==IterNum-1) {
            cudaStat1 = cudaMemcpy(V, d_y, sizeof(float)*n*r*k, cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);
        }
        cudaFree(d_work);
        //LS_U
        cublas_status = cublasSgemmStridedBatched(
                cublasH,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                d,
                r*k,
                n,
                &alpha,
                d_At,
                d,
                d*n,
                d_y,
                r*k,
                0,
                &beta,
                d_W,
                d,
                d*r*k,
                m*k
                );
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        assert(cudaSuccess == cudaStat1);

        cusolver_status = cusolverDnSgeqrf_bufferSize(
            cusolverH,
            d,
            m*r*k*k,
            d_W,
            d,
            &lwork);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
        assert(cudaSuccess == cudaStat1);

    // step 4: compute QR factorization
    cudaStat2 = cudaMemcpy(d_y, d_yL, sizeof(float) * d, cudaMemcpyDeviceToDevice);
    assert(cudaSuccess == cudaStat2);
        cusolver_status = cusolverDnSgeqrf(
            cusolverH,
            d,
            m*r*k*k,
            d_W,
            d,
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
            d,
            1,
            m*r*k*k,
            d_W,
            d,
            d_tau,
            d_y,
            d,
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
             m*r*k*k,
             1,
             &one,
             d_W,
             d,
             d_y,
             d);
        cudaStat1 = cudaDeviceSynchronize();
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        assert(cudaSuccess == cudaStat1);
        if (i==IterNum-1) {
            cudaStat1 = cudaMemcpy(Ut, d_y, sizeof(float)*m*k*r*k, cudaMemcpyDeviceToHost);
            assert(cudaSuccess == cudaStat1);
        }
        cudaFree(d_work);

    }
    //float* Xr = new float[m*n*k];
    if(d_W) cudaFree(d_W);
    if(d_A) cudaFree(d_A);
    if(d_At) cudaFree(d_At);
    if(d_y) cudaFree(d_y);
    if(d_yL) cudaFree(d_yL);
    if(d_tau) cudaFree(d_tau);
    if(devInfo) cudaFree(devInfo);
    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    free(Ut);
    free(V);
    free(Um);
    cudaDeviceReset();
}

int main(){
    int loopNum;
    cin >> loopNum;
    int m, n, r, k, d, IterNum;
    for (int it=0; it<loopNum; it++){
	    cin >> m >> n >> r >> k >> d >> IterNum;
	    //float* A = new float[m*n*k*d];
	    float*A = (float*)malloc(m*n*k*d*sizeof(float));
	    float*At = (float*)malloc(m*n*k*d*sizeof(float));
	    float*X = (float*)malloc(m*n*k*sizeof(float));
	    float*y = (float*)malloc(d*sizeof(float));
	    //float* X = new float[m*n*k];
	    //float* y = new float[d];
	    for (int i=0; i<m*n*k; i++){
		X[i] = rand()%(1000) / (float) (1000);
	    }
	    //cout << "X is \n";
	    //printTensor(X, m*k, n, 1);
	    for (int i=0; i<m*n*k*d; i++){
		A[i] = rand()%(1000) / (float) (1000);
	    }
	    //cout << "A is \n";
	    //printTensor(A, d, m*n*k, 1);
	    for (int i=0; i<m*n*k*d; i++){
		At[i] = rand()%(1000) / (float) (1000);
	    }
	    //cout << "At is \n";
	    //printTensor(At, d, m*n*k, 1);
		//initial y by Ax
		//cuGemv(A, X, y, d, m*n*k);
	    for (int i=0; i<d; i++){
		y[i] = rand()%(1000) / (float) (1000);
	    }
	    //cout << "y is \n";
	    //printTensor(y, d, 1, 1);
	    //print Amt
        warmup();
	    clock_t t1 = clock();
	    TS(X, A, At, y, d, IterNum, m, n, r, k);
	    printf("size %d takes time %f s\n", m, (clock() - t1)*1.0 / CLOCKS_PER_SEC *1000);
	    free(A);
	    free(At);
	    free(X);
	    free(y);
    }
    return 0;
}
