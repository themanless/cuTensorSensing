#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
using namespace std;

void cuGemv(float* A, float* X, float* y, int m, int n);
