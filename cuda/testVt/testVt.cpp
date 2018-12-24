#include "../lsqr.h"
#include "../cirMM.h"
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
void LS_V(float* A, float* y, float* U, float* V, int m, int n, int r, int k, int d){
    float* W = new float[n*r*k*d];
    int K = m*k, M = d, N = r*k, batch = n;
    cirMM(W, A, U, M, N, K, batch);
	cout << "W is  \n";
    printTensor(W, d, r*n*k, 1);
    lsqr(W, y, V, d, r*k*n);
    delete W;
}

void TS(float* X, float* A, float* y, int d, int IterNum, int m, int n, int r, int k){
    float* Um = new float[m*r*k];
    float* V = new float[n*r*k];
    for(int i=0; i<m*r*k; i++){
        Um[i] = 1.0;
    }
    float* U = new float[m*k*r*k];
    for (int i=0; i<r; i++)
        for (int j=0; j<m; j++)
            for (int a=0; a<k; a++)
                for (int b=0; b<k; b++){
                    U[(i*k+a)*m*k+j*k+b] = Um[(i*m+j)*k+(b-a+k)%k];
                }

    LS_V(A, y, U, V, m, n, r, k, d);
    printTensor(V, r*k, n, 1);
    delete Um;
    delete U;
    delete V;
}


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
	//initial y by Ax
	cuGemv(A, X, y, d, m*n*k);
    //print Amt
    cout << "A is \n";
    printTensor(A, d, m*k, n);
    cout << "X is \n";
    printTensor(X, m, n, k);
    cout << "y is \n";
    printTensor(y, d, 1, 1);

    TS(X, A, y, d, IterNum, m, n, r, k);
    delete A;
    delete X;
    delete y;
    return 0;
}
