#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
using namespace std;

void cirMM(float* W, float* A, float* V, int M, int N, int K, int batch);
