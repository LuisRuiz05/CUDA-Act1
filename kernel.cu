#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>
#include <iostream>
#include <ctime>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void mulMatrixGPU(int* a, int* b, int* c, int size);

int main()
{
    //3D
    //dim3 grid(2, 2, 2);
    //dim3 block(2, 2, 2);
    //idxCalc3D << <grid, block >> > ();

    const int n = 2;
    int size = n * sizeof(n);

    // Declare 2x2 Matrix
    int a[n][n];
    int b[n][n];
    int c[n][n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n; i++) {
        for (int j = 0; i < n; i++) {
            a[i][j] = rand() % 226;
            b[i][j] = rand() % 226;
        }
    }

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, size, cudaMemcpyHostToDevice);

    // Solve Operation
    dim3 grid(8, 4, 4);
    dim3 block(8, 4, 4);
    mulMatrixGPU << <grid, block >> > (devA, devB, devC, size);

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void mulMatrixGPU(int* a, int* b, int* c, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; //thread ID + offset

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
        printf("[DEVICE] %d + %d = %d\n\r", a[gid], b[gid], c[gid]);
    }
}