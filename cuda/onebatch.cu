#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>

using namespace std;

void onebatch(float* A, float* B, float* C, int m, int n, int k, int batch){
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
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) *m*k*batch);
    cudaStat2 = cudaMalloc((void**)&d_B, sizeof(float) *k*n);
    cudaStat3 = cudaMalloc((void**)&d_C, sizeof(float) *m*n*batch);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) *m*k*batch, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) *n*k, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cublas_status = cublasSgemmStridedBatched(
            cublasH,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            d_A,
            m,
            m*k,
            d_B,
            k,
            0,
            &beta,
            d_C,
            m,
            m*n,
            batch
            );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(C, d_C, sizeof(float)*m*n*batch, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_C    ) cudaFree(d_C);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();

}
void printTensor(float* ar, int row, int col, int tuple){
    for (int z=0; z<tuple; z++){
        for (int j=0; j<row; j++){
            for (int i=0; i<col; i++){
                cout << ar[z*row*col+i*row+j] << " ";
            }
            cout << endl;
        }
        cout << "----------------\n";
    }
}

int main(){
    int m, n, k, batch;
    cin >> m >> n >> k >> batch;
    float* A = new float[m*k*batch];
    float* B = new float[k*n];
    float* C = new float[m*n*batch];
    for (int i=0; i<m*k*batch; i++){
        cin >> A[i];
    }
    for (int i=0; i<k*n; i++){
        cin >> B[i];
    }
    cout << "A is \n";
    printTensor(A, m, k, batch);
    cout << "B is \n";
    printTensor(B, k, n, 1);
    onebatch(A, B, C, m, n, k, batch);
    cout << "C is \n";
    printTensor(C, m, n, batch);
    delete A;
    delete C;
    delete B;
    return 0;
}
