#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// GPU kernel for matrix multiplication
__global__ void matrixMultiplicationGPU(int* a, int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

// CPU function for matrix multiplication
void matrixMultiplicationCPU(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    srand(time(nullptr));
    const int N = 1000; // Matrix size
    const int numInstances = 10; // Number of instances to run
    float averageCPUTime = 0.00;
    float averageGPUTime = 0.00;

    for (int instance = 0; instance < numInstances; instance++) {
        int *a, *b, *cGPU, *cCPU; // Host Matrices (CPU)
        int *da, *db, *dc; // Device Matrices (GPU)

        // Memory Allocation for Host Devices
        a = new int[N * N];
        b = new int[N * N];
        cGPU = new int[N * N];
        cCPU = new int[N * N];

        // Load Host Matrices with Random Values between 0-100
        for (int i = 0; i < N * N; i++) {
            a[i] = rand() % 101;
            b[i] = rand() % 101;
        }

        // Allocate Memory for Device Matrices
        cudaMalloc((void**)&da, N * N * sizeof(int));
        cudaMalloc((void**)&db, N * N * sizeof(int));
        cudaMalloc((void**)&dc, N * N * sizeof(int));

        // Copy Host Matrices to Device Matrices
        cudaMemcpy(da, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

        // Define Grid/Block Dimensions for Threads
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Record start time for GPU
        clock_t startGPU = clock();

        // Launch GPU Kernel
        matrixMultiplicationGPU <<<numBlocks, threadsPerBlock>>> (da, db, dc, N);

        // Synchronize GPU
        cudaDeviceSynchronize();

        // Record end time for GPU
        clock_t endGPU = clock();
        float millisecondsGPU = 1000.0 * (endGPU - startGPU) / CLOCKS_PER_SEC;
        averageGPUTime += millisecondsGPU;

        // Copy GPU Results back to CPU
        cudaMemcpy(cGPU, dc, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        // Record start time for CPU
        clock_t startCPU = clock();

        // Run CPU matrix multiplication
        matrixMultiplicationCPU(a, b, cCPU, N);

        // Record end time for CPU
        clock_t endCPU = clock();
        float millisecondsCPU = 1000.0 * (endCPU - startCPU) / CLOCKS_PER_SEC;
        averageCPUTime += millisecondsCPU;

        // Print elapsed time for GPU and CPU
        cout << "Instance " << instance + 1 << " - GPU Time: " << millisecondsGPU << " milliseconds" << endl;
        cout << "Instance " << instance + 1 << " - CPU Time: " << millisecondsCPU << " milliseconds" << endl;

        // Free Memory for this instance
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);

        delete[] a;
        delete[] b;
        delete[] cGPU;
        delete[] cCPU;
    }
    cout << "Average CPU Runtime: " << averageCPUTime / numInstances << " milliseconds" << endl;
    cout << "Average GPU Runtime: " << averageGPUTime / numInstances << " milliseconds" << endl;
    cout << "Speedup Percentage: " << (averageCPUTime) / (averageGPUTime)  << " times faster than CPU" << endl;

    return 0;
}