#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void arrayFlip(int* input, int* output, int numElements)
{
    int i = threadIdx.x;
    output[numElements - i - 1] = input[i];
}

int main()
{
    //initialize arrays
    const int SIZE = 128000;
    int A[SIZE];
    int B[sizeof(A) / sizeof(int)] = { 0 };

    // fill array A
    for (int i = 0; i < SIZE; i++) {

        A[i] = rand();
    }

    //pointers to GPU memory
    int* gpuA = 0;
    int* gpuB = 0;

    //allocate memory for arrays
    cudaMalloc(&gpuA, sizeof(A));
    cudaMalloc(&gpuB, sizeof(B));

    //copy arrays to GPU
    cudaMemcpy(gpuA, A, sizeof(A), cudaMemcpyHostToDevice);

    //call GPU function
    arrayFlip <<< 1, sizeof(a)/sizeof(int) >>> (cudaA, cudaB, SIZE);

    //copy result into B
    cudaMemcpy(B, gpuB, sizeof(B), cudaMemcpyDeviceToHost);

    return 0;
}