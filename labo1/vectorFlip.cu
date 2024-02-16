#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorFlip(const float *input, float *output, int numElements) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) 
  {
    output[numElements-i-1] = input[i];  
  }
}

int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector flip of %d elements]\n", numElements);

  // Allocate the host input
  float *h_input = (float *)malloc(size);

  // Allocate the host output
  float *h_output = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_input == NULL || h_output == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vector
  for (int i = 0; i < numElements; ++i) {
    h_input[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector
  float *d_input = NULL;
  err = cudaMalloc((void **)&d_input, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector
  float *d_output = NULL;
  err = cudaMalloc((void **)&d_output, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vector in host memory to the device input
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector flip CUDA Kernel
  int threadsPerBlock = 256;
  //int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGrid = 196;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorFlip<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //print input vector
  /*for(int i = 0; i < numElements; i++)
  {
    printf("%.6f ", h_input[i]);
  }

  printf("\nflipped vector \n");
  //print output vector
  for(int i = 0; i < numElements; i++)
  {
    printf("%.6f ", h_output[i]);
  }

  printf("\n");*/

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) 
  {
    if (h_output[i] != h_input[numElements-i-1]) 
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_input);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_output);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_input);
  free(h_output);

  printf("Done\n");
  return 0;
}