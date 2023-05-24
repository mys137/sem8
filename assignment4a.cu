#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <stdio.h>

using namespace std;

#define TILE_WIDTH 32

__global__ void matrixMult(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// serial matrix multiplication 
void mulMat(vector<vector<int>> mat1, vector<vector<int>> mat2)
{   
    int R1 = mat1.size();
    int C1 = mat1.size();
    int R2 = mat2.size();
    int C2 = mat2.size();

    int rslt[R1][C2];
  
    cout << "Multiplication of given two matrices is:\n";
  
    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            rslt[i][j] = 0;
  
            for (int k = 0; k < R2; k++) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
  
            cout << rslt[i][j] << "\t";
        }
  
        cout << endl;
    }
}



int main()
{
    int n;
    n=10;

    // allocate memory for matrices on host
    int *a = new int[n * n];
    int *b = new int[n * n];
    int *c = new int[n * n];

    // initialize matrices with random values
    std::srand(std::time(0));
    for (int i = 0; i < n * n; ++i) {
        a[i] = std::rand() % 10;
        b[i] = std::rand() % 10;
    }

    std::vector<std::vector<int>> aa;
    std::vector<std::vector<int>> bb;

    for(int i=0;i<n;i++){
        std::vector<int> temp(n);
        for(int j=0;j<n;j++){
            temp[j] = a[i * n + j];
        }
        aa.push_back(temp);
    }

     for(int i=0;i<n;i++){
        std::vector<int> temp(n);
        for(int j=0;j<n;j++){
            temp[j] = b[i * n + j];
        }
        bb.push_back(temp);
    }
    std::cout<<" ------------------------------------------------------------------------\n";
    std::cout<<"----------------serial computing------------------------------------\n";
    auto starts = omp_get_wtime();
    mulMat(aa,bb);
    auto ends = omp_get_wtime();
    cout<<" time taken by serial: "<<ends-starts<<endl;

    // allocate memory for matrices on device
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, n * n * sizeof(int));
    cudaMalloc(&dev_b, n * n * sizeof(int));
    cudaMalloc(&dev_c, n * n * sizeof(int));

    // copy matrices from host to device
    cudaMemcpy(dev_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    auto start = omp_get_wtime();
    // launch kernel
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (n - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);
    auto end = omp_get_wtime();

    // copy result matrix from device to host
    cudaMemcpy(c, dev_c, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // print result matrix
     std::cout<<"----------- parallel computing---------------------------------\n";
    std::cout<<"time taken : "<<end-start<<endl;

//  std::cout << "Matrix A:\n";
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << a[i * n + j] << " ";
//         }
//         std::cout << "\n";
//     }
//  std::cout << "Matrix B :\n";
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << b[i * n + j] << " ";
//         }
//         std::cout << "\n";
//     }
    std::cout << "Result matrix:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << c[i * n + j] << " ";
        }
        std::cout << "\n";
    }


    // free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // free memory on host
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
