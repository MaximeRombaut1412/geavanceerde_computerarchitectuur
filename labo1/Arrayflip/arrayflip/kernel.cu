#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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