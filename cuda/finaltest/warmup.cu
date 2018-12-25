void warmup(){
    int n = 100;
    float a[100], b[100];
    for (int i=0; i<n; i++>){
        a[i] = i;
        b[i] = i;
    }
    float *d_a, *d_b;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void**)&d_a, sizeof(float)*n);
    cudaMalloc((void**)&d_b, sizeof(float)*n);
    float alpha = 1.0;
    cublasSetVector(n, sizeof(float), a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), b, 1, d_b, 1);
    cublasSaxpy_v2(handle, n, &alpha, d_a,1,d_b,1);
    cublasGetVector(n, sizeof(float), d_b, 1, a, 1);
    cudaFree(d_a);
    cudaFree(d_b);
    cublasDestroy(handle);
}
