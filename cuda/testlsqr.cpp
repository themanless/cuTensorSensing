#include "lsqr.h"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
using namespace std;
/*
void randInit(float* ar, int len){
    for(int i=0; i<len; i++){
        ar[i] = rand()%(1000) / (float) (1000)；
    }
}
*/
void printMatrix(float* ar, int row, int col){
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            cout << ar[i*col+j] << " ";
        }
        cout << endl;
    }
}

int main(){
    int m, n;
    cin >> m >> n;
    float* A = new float[m*n];
    float* x = new float[n];
    float* b = new float[m];
    for (int i=0; i<m*n; i++)
        cin >> A[i];
    for (int i=0; i<m; i++)
        cin >> b[i];
    cout << "A is \n";
    printMatrix(A, m, n);
    printMatrix(b, m, 1);
    cout << "b is \n";
    lsqr(A, x, b, m, n);
    cout << "x is \n";
    printMatrix(x, n, 1);
    delete A;
    delete x;
    delete b;
    return 0;
}
