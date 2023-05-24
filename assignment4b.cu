#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <stdio.h>

using namespace std;

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// function to add the vector serially 
void matAdd(vector<int> a,vector<int> b){
    vector<int> c(a.size());

    std::cout<<" addition using serial : ";
    for(int i=0;i<a.size();i++){
        c[i] = a[i]+b[i];
    }
    cout<<endl;
}

int main() {
    const int n = 1000;  // Length of vectors
    std::vector<int> a(n), b(n), c(n);

    // Initialize vectors with random values
    std::srand(std::time(nullptr));
    for (int i = 0; i < n; ++i) {
        a[i] = std::rand() % 100;
        b[i] = std::rand() % 100;
    }


    std::cout<<" ------------------------------------------------------------------------\n";
    std::cout<<"----------------serial computing------------------------------------\n";
    auto starts = omp_get_wtime();
    matAdd(a,b);
    auto ends = omp_get_wtime();
    cout<<" time taken by serial: "<<ends-starts<<endl;

    // Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    auto start = omp_get_wtime();
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    vector_add<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);
     auto end = omp_get_wtime();

    // Copy output data from device to host
    cudaMemcpy(c.data(), d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Print results
    std::cout<<"---------------------------------------------------------------\n";
    std::cout<<"----------- parallel computing---------------------------------\n";
    std::cout<<"time taken : "<<end-start<<endl;
    return 0;
}

