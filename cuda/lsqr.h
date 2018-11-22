#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void lsqr(float* A, float* X, float* B, int m, int n);
