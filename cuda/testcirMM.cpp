#include "cirMM.h"
#include <iostream>
#include <cstdio>
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

int main(){
    int m, n, r, k, d;
    cin >> m >> n >> r >> k >> d;
    float* A = new float[m*n*k*d];
    float* U = new float[m*r*k];
    float* W = new float[r*n*k*d];
    for (int i=0; i<m*n*k*d; i++){
        cin >> A[i];
    }
    for (int i=0; i<m*r*k; i++){
        cin >> U[i];
    }
    cout << "A is \n";
    printTensor(A, d, m*k, n);
    cout << "U is \n";
    printTensor(U, m, r, k);
    cirMM(W, A, U, m, n, r, k, d);
    cout << "W is \n";
    printTensor(W, d, r*k, n);
    delete A;
    delete W;
    delete U;
    return 0;
}

