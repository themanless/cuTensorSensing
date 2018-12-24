#include <iostream>
#include "../TS.cu"

using namespace std;

int main(){
    int m, n, r, k, d, IterNum;
    cin >> m >> n >> r >> k >> d;
    float* A = new float[m*n*k*d];
    float* X = new float[m*n*k];
    float* y = new float[d];
    for (int i=0; i<m*n*k*d; i++){
            cin >> A[i];
    }
    for (int i=0; i<m*n*k; i++){
    cin >> X[i];
    }
    for (int i=0; i<d; i++){
        cin >> y[i];
    }
    TS(X, A, y, d, IterNum, m, n, r, k);
}
