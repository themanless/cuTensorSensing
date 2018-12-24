#include <ctime>
#include <cstdlib>
void TS(float* X, float* A, float* y, int d, int IterNum, int m, int n, int r, int k){
    float* Um = new float[m*r*k];
    float* V = new float[n*r*k];
    for(int i=0; i<m*r*k; i++){
        Um[i] = rand()%(1000) / (float) (1000);
    }
    float* U = new float[m*k*r*k];
    for (int i=0; i<m; i++)
        for (int j=0; j<r; j++)
            for (int a=0; a<k; a++)
                for (int b=0; b<k; b++){
                    Ut[(i*k+a)*r*k+j*k+b] = Um[(i*r+j)*k+(b-a+k)%k];
                }

    for (int i=0; i<IterNum; i++){
        LS_V(A, y, Ut, V, m, n, r, k, d);
        LS_U(A, y, Ut, V, m, n, r, k, d);
        //compute RMSE between U*V and X

    }
}


void LS_V(float* A, float* y, float* Ut, float* V, int m, int n, int r, int k, int d){
    float* W = new float[n*r*k*d];
    int K = m*k, M = d, N = r*k, batch = n;
    cirMM(W, A, Ut, M, N, K, batch);
    lsqr(W, y, V, d, r*k*n);
    delete W;
}
void LS_U(float* At, float* y, float* Ut, float* V, int m, int n, int r, int k, int d){
    float* W = new float[n*r*k*d];
    int K = n, M = d, N = r*k, batch = m*k;
    cirMM(W, At, V, M, N, K, batch);
    //compute U_
    lsqr(W, y, Ut, d, m*r*k*n);
    delete W;
}
