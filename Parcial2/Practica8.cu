#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

 void sort(int* a, int n);
__global__ void sortGPU(int* a, int n);


int main()
{
    const int n = 8;
    int size = n * sizeof(int);
    
    int* a, * ans;

    // Allocate space for local variables
    a = (int*)malloc(size);
    ans = (int*)malloc(size);

    // Assign and print a random value to every position
    printf("Assigned values\n");
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 256;
        printf("%d ", a[i]);
    }

    // Print separation line
    printf("\n");

    int* devA;

    // Allocate Memory
    cudaMalloc(&devA, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);

    // Declare grid
    dim3 block(n);

    // Solve operations
    sort(a, n);
    sortGPU << <1, block >> > (devA, n);
    cudaMemcpy(ans, devA, size, cudaMemcpyDeviceToHost);

    // Print solution CPU
    printf("Sorted CPU\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }

    // Print separation line
    printf("\n");

    // Print solution GPU
    printf("Sorted GPU\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", ans[i]);
    }

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);

    return 0;
}

void sort(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}

__global__ void sortGPU(int* a, int n) {
    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {
        int offset = i % 2;
        int left = 2 * tid + offset;
        int right = left + 1;

        if (right < n) {
            if (a[left] > a[right]) {
                int aux = a[left];
                a[left] = a[right];
                a[right] = aux;
            }
        }
        __syncthreads();
    }
}
