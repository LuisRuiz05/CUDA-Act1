
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void idxCalc(int* input);
__global__ void idxCalc2D(int* input);
__global__ void idxCalc2Dx2D(int* input);

int main()
{
    const int n = 16;
    int size = n * sizeof(n);

    int a[n] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int* devA = 0;

    cudaMalloc((void**)&devA, size);
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);

    //1Dim
    //idxCalc << <1, n >> > (devA);
    //idxCalc << <2, 8 >> > (devA);
    //idxCalc << <4, 4 >> > (devA);

    //2dGrid
    //dim3 grid(2,2);
    //dim3 block(4);
    //idxCalc2D << <grid, block >> > (devA);

    //2dGrid & 2dBlock
    dim3 grid(2,2);
    dim3 block(2,2);
    idxCalc2Dx2D << <grid, block >> > (devA);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    
    return 0;
}

__global__ void idxCalc(int* input) {
    int tid = threadIdx.x;

    // Offset
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;

    printf("[DEVICE] blockIdx.x %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}

__global__ void idxCalc2D(int* input) {
    int tid = threadIdx.x;

    // Offset
    int offsetBlock = blockIdx.x * blockDim.x;
    int offsetRow = blockIdx.y * blockDim.x * gridDim.y;
    int gid = tid + offsetBlock + offsetRow;

    printf("[DEVICE] blockIdx.x %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

__global__ void idxCalc2Dx2D(int* input) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Offset
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * gridDim.x * blockDim.y;
    int gid = tid + offsetBlock + offsetRow;

    printf("[DEVICE] blockIdx.x %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}
