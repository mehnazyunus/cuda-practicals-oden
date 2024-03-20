////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

__constant__ int power_of_2;

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float *idata, int len)
{
  float sum = 0.0f;
  for (int i = 0; i < len; i++)
    sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
  // dynamically allocated shared memory

  extern __shared__ float temp[];

  int tid = threadIdx.x;

  // first, each thread loads data into shared memory

  temp[tid] = g_idata[tid + blockIdx.x*blockDim.x];

  if (tid % blockDim.x == 0)
    printf("Extra elements : %d\n", blockDim.x - power_of_2);

  if (tid < blockDim.x - power_of_2)
  {
    // printf("Adding extra: %d\n", tid);
    temp[tid] += temp[tid + power_of_2];
  }

  __syncthreads();

  // next, we perform binary tree reduction

  for (int d = power_of_2 / 2; d > 0; d = d / 2)
  {
    __syncthreads(); // ensure previous step completed
    if (tid < d)
      temp[tid] += temp[tid + d];
  }

  // finally, first thread puts result into global memory

  if (tid % blockDim.x == 0)
    g_odata[blockIdx.x] = temp[0];
}

////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;
  int h_power_of_2; // highest power of 2 that is closest to num_threads

  float *h_data, *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks = 3; // start with only 1 thread block
  num_threads = 8;
  num_elements = num_blocks * num_threads;
  mem_size = sizeof(float) * num_elements;

  for (h_power_of_2 = 1; h_power_of_2 < num_threads; h_power_of_2 *= 2)
  {
  }
  if(h_power_of_2 != num_threads) // if num_threads is not already a power of 2
    h_power_of_2 >>= 1;
  printf("Highest power of 2 : %d\n", h_power_of_2);

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float *)malloc(mem_size);

  for (int i = 0; i < num_elements; i++)
    h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));

  // compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, sizeof(float)*num_blocks));

  // copy host memory to device input array
  checkCudaErrors(cudaMemcpyToSymbol(power_of_2, &h_power_of_2, sizeof(h_power_of_2)));
  checkCudaErrors(cudaMemcpy(d_idata, h_data, mem_size,
                             cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_data, d_odata, sizeof(float)*num_blocks,
                             cudaMemcpyDeviceToHost));

  // sum over partial sums
  for(int i=1; i<num_blocks; ++i) {
    h_data[0] += h_data[i];
  }

  // check results

  printf("reduction error = %f\n", h_data[0] - sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
