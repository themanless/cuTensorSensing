#include <ctime>
#include <cstdlib>
void TS(float* X, float* A, float* y, int d, int IterNum, int m, int n, int r, int k){
    float* U = new float[m*r*k];
    float* V = new float[n*r*k];
    randInit(U, m*r*k);
    for (int i=0; i<IterNum; i++){
        LS(A, y, m, n, r, k, d);
    }
}

void randInit(float* ar, int len){
    for(int i=0; i<len; i++){
        ar[i] = rand()%(1000) / (float) (1000)；
    }
}

void LS(float* A, float* y, float* U, float* V, int m, int n, int r, int k, int d){
    float* W = new float[n*r*k*d];
    cirMM(W, A, U, m, n, r, k, d);
    lsqr(W, y, V, m, n, r, k, d);
}
