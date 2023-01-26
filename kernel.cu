
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void multiplication(int *a, int *b, int *c);

int main() {
    const int n = 3;
    int size = n * sizeof(n);

    int a[n] = { 2, 7, 10 };
    int b[n] = { 4, 0, 1 };
    int c[n] = { 0, 0, 0 };

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);

    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, size, cudaMemcpyHostToDevice);

    multiplication << <1, n >> > (devA, devB, devC);
    cudaDeviceSynchronize();

    cudaMemcpy(c, devC, size, cudaMemcpyDeviceToHost);
    printf("{%d, %d, %d}", c[0], c[1], c[2]);
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void multiplication(int* a, int* b, int* c) {
    int id = threadIdx.x;
    c[id] = a[id] * b[id];
}

/*__global__ void indexes();

int main()
{
    dim3 block(2, 2, 2);
    dim3 grid(4 / block.x, 4 / block.y, 4 / block.z);
    indexes << <grid, block >> > ();

    return 0;
}

__global__ void indexes() {
    printf("threadIdx %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);

}
*/