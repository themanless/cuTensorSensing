#include "../lsqr.h"
#include "../cirMM.h"
#include <string>
#include "initialY.h"
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;
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
void LS_V(float* A, float* y, float* Ut, float* V, int m, int n, int r, int k, int d){
    float* W = (float*)malloc(n*r*k*d*sizeof(float));
    int K = m*k, M = d, N = r*k, batch = n;
    cirMM(W, A, Ut, M, N, K, batch);
    //cout << "Wv is \n";
    //printTensor(W, d, r*k*n, 1);
    lsqr(W, V, y, d, r*k*n);
    //cout << "LV is \n";
    //printTensor(V, 1, r*k*n, 1);
    free(W);
}
void LS_U(float* At, float* y, float* Ut, float* V, int m, int n, int r, int k, int d){
    //float* W = new float[m*k*r*k*d];
    float* W = (float*)malloc(m*k*r*k*d*sizeof(float));
    int K = n, M = d, N = r*k, batch = m*k;
    cirMM(W, At, V, M, N, K, batch);
    //cout << "WU is \n";
    //printTensor(W, d, r*k*m*k, 1);
    //compute U_
    lsqr(W, Ut,y,  d, m*r*k*k);
    //cout << "LUt is \n";
    //printTensor(Ut, 1, r*k*m*k, 1);
    free(W);
}

float rseCPU(float* A, float* B, int len){
    float VB=0.0, VM=0.0;
    for (int i=0; i<len; i++){
        VB += B[i]*B[i];
        VM += (B[i] -A[i])*(B[i]-A[i]);
    }
    return VM/VB;
}
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


    float* Xr = (float*)malloc(n*m*k*sizeof(float));
    for (int i=0; i<IterNum; i++){
        LS_V(A, y, Ut, V, m, n, r, k, d);
        LS_U(At, y, Ut, V, m, n, r, k, d);
        //compute RMSE between U*V and X
        cuGemm(Ut, V, Xr, m*k, n, r*k);
        float rse = rseCPU(Xr, X, m*r*n);
        cout << "res is ";
        cout << rse << endl;
    }
    //float* Xr = new float[m*n*k];
    free(Xr);
    free(Ut);
    free(V);
    free(Um);
}

int main(){
    int m, n, r, k, d, IterNum;
    cin >> m >> n >> r >> k >> d >> IterNum;
    //float* A = new float[m*n*k*d];
    float*A = (float*)malloc(m*n*k*d*sizeof(float));
    float*At = (float*)malloc(m*n*k*d*sizeof(float));
    float*X = (float*)malloc(m*n*k*sizeof(float));
    float*y = (float*)malloc(d*sizeof(float));
    //float* X = new float[m*n*k];
    //float* y = new float[d];
    string str;
    cin >> str;
    for (int i=0; i<m*n*k; i++){
		cin >> X[i];
    }
    //cout << "X is \n";
    //printTensor(X, m*k, n, 1);
    cin >> str;
    for (int i=0; i<m*n*k*d; i++){
		cin >> A[i];
    }
    //cout << "A is \n";
    //printTensor(A, d, m*n*k, 1);
    cin >> str;
    for (int i=0; i<m*n*k*d; i++){
		cin >> At[i];
    }
    //cout << "At is \n";
    //printTensor(At, d, m*n*k, 1);
	//initial y by Ax
	//cuGemv(A, X, y, d, m*n*k);
    cin >> str;
    for (int i=0; i<d; i++){
		cin >> y[i];
    }
    //cout << "y is \n";
    //printTensor(y, d, 1, 1);
    //print Amt
    TS(X, A, At, y, d, IterNum, m, n, r, k);
    free(A);
    free(X);
    free(y);
    return 0;
}
