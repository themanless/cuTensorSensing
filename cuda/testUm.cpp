#include <iostream>
using namespace std;

void printMatrix(float* ar, int row, int col){
    for (int j=0; j<row; j++){
        for (int i=0; i<col; i++){
            cout << ar[i*row+j] << " ";
        }
        cout << endl;
    }
}

int main(){
    int m, k, r;
    cin >> m >> k >> r;
    float* U = new float[m*r*k];
    for (int i=0; i<m*k*r; i++){
        cin >> U[i];
    }
    cout << "U is \n";
    printMatrix(U, m*k, r);
    float* Um = new float[m*k*r*k];
    for (int i=0; i<r; i++)
        for (int j=0; j<m; j++)
            for (int a=0; a<k; a++)
                for (int b=0; b<k; b++){
                    Um[(i*k+a)*m*k+j*k+b] = U[(i*m+j)*k+(b-a+k)%k];
                }
    cout << "Um is \n";
    printMatrix(Um, m*k, r*k);
    delete U;
    delete Um;
    return 0;
}

