#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
    //allocate memory fore array A and B
    const int SIZE = 5000000000;
    size_t size = SIZE * sizeof(int);
    int* A = (int*)malloc(size);
    int* B = (int*)malloc(size);

    // Initialize array A and fill it up with random ints
    for (int i = 0; i < SIZE; ++i) {
        A[i] = rand();
    }

    //flip array A into array B
    for (int i = 0; i < SIZE; i++)
    {
        B[i] = A[SIZE - 1 - i];
    }

    // Verify that the result vector is correct
    for (int i = 0; i < SIZE; ++i)
    {
        if (B[i] != A[SIZE - i - 1])
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    return 0;
}