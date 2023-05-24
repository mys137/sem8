#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <stdio.h>

typedef struct {
    int weight;
    int value;
} Item;

void knapsackSerial(Item* items, int num_items, int capacity) {
    int** dp = (int**)malloc((num_items + 1) * sizeof(int*));
    for (int i = 0; i <= num_items; i++) {
        dp[i] = (int*)malloc((capacity + 1) * sizeof(int));
    }

    // Initialize the first row and column of the table to 0
    for (int i = 0; i <= num_items; i++) {
        dp[i][0] = 0;
    }
    for (int j = 0; j <= capacity; j++) {
        dp[0][j] = 0;
    }

    // Fill in the table using dynamic programming
    for (int i = 1; i <= num_items; i++) {
        for (int j = 1; j <= capacity; j++) {
            // Check if the current item can be included in the knapsack
            if (items[i - 1].weight <= j) {
                dp[i][j] = max(items[i - 1].value + dp[i - 1][j - items[i - 1].weight], dp[i - 1][j]);
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    // Backtrack to find the selected items
    int i = num_items;
    int j = capacity;
    while (i > 0 && j > 0) {
        if (dp[i][j] != dp[i - 1][j]) {
            printf("Item %d: weight = %d, value = %d\n", i, items[i - 1].weight, items[i - 1].value);
            j -= items[i - 1].weight;
        }
        i--;
    }

    // Get the maximum value
    int max_value = dp[num_items][capacity];

    // Free memory
    for (int i = 0; i <= num_items; i++) {
        free(dp[i]);
    }
    free(dp);

    printf("Maximum value: %d\n", max_value);
}


__global__ void knapsackDP(const Item* items, int num_items, int capacity, int* dp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= capacity) {
        dp[idx] = 0;
    }
    __syncthreads();

    for (int i = 0; i < num_items; i++) {
        if (idx >= items[i].weight) {
            int newValue = dp[idx - items[i].weight] + items[i].value;
            if (newValue > dp[idx]) {
                dp[idx] = newValue;
            }
        }
        __syncthreads();
    }
}

void knapsack(Item* items, int num_items, int capacity) {
    Item* dev_items;
    int* dev_dp;

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_items, num_items * sizeof(Item));
    cudaMalloc((void**)&dev_dp, (capacity + 1) * sizeof(int));

    // Copy items to the GPU
    cudaMemcpy(dev_items, items, num_items * sizeof(Item), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int grid_size = (capacity + block_size - 1) / block_size;
    knapsackDP<<<grid_size, block_size>>>(dev_items, num_items, capacity, dev_dp);

    // Copy result back to the CPU
    int* dp = (int*)malloc((capacity + 1) * sizeof(int));
    cudaMemcpy(dp, dev_dp, (capacity + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the maximum value
    int max_value = dp[capacity];

    // Find selected items
    int remaining_capacity = capacity;
    for (int i = num_items - 1; i >= 0; i--) {
        if (remaining_capacity >= items[i].weight && dp[remaining_capacity] == dp[remaining_capacity - items[i].weight] + items[i].value) {
            printf("Item %d: weight = %d, value = %d\n", i + 1, items[i].weight, items[i].value);
            remaining_capacity -= items[i].weight;
        }
    }

    // Free memory
    cudaFree(dev_items);
    cudaFree(dev_dp);
    free(dp);

    printf("Maximum value: %d\n", max_value);
}


int main()
{
    Item items[] = {
        {2, 12},
        {1, 10},
        {3, 20},
        {2, 15},
        {4, 5},
        {5, 9},
        {1, 14},
        {3, 10},
        {5, 18},
        {2, 13},
        {5, 11},
        {6, 13},
        {7, 14},
        {8, 11}
    };

    int numItems = sizeof(items) / sizeof(items[0]);
    int capacity = 10;


    std::cout<<"------------------------------------------------------------------\n";
    auto startserial = omp_get_wtime();
    knapsackSerial(items,numItems,capacity); 
    auto endserial = omp_get_wtime();

    std::cout<<"Time taken by serial : "<<endserial-startserial<<"\n";    

    std::cout<<"------------------------------------------------------------------\n";

    auto start = omp_get_wtime();
    knapsack(items, numItems, capacity);
    auto end = omp_get_wtime();


    //printf("Max Value: %d\n", maxValue);

    std::cout<<"Time taken by parallel : "<<end-start<<"\n";

    return 0;
}
