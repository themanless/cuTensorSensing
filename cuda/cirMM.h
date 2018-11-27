#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
using namespace std;

void cirMM(float* W, float* A, float* U, int m, int n, int r, int k, int d);
