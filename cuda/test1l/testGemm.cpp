#include "initialY.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>

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
    int m, n, k;
    cin >> m >> n>> k;
    float* A = (float*)malloc(m*k*sizeof(float));
    float* B = (float*)malloc(n*k*sizeof(float));
    float* C = (float*)malloc(m*n*sizeof(float));
    for (int i=0; i<m*k; i++)
        cin >> A[i];
    cout << "A is \n";
    printTensor(A, k, m, 1);
    for (int i=0; i<n*k; i++)
        cin >> B[i];
    cout << "B is \n";
    printTensor(B, k, n, 1);
    cuGemm(A, B, C, m, n, k);
    cout << "C is \n";
    printTensor(C, m, n, 1);
    free(A);
    free(B);
    free(C);
    return 0;
}
