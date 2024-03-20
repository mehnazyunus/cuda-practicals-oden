//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float* y)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  printf("tid - %d\n", tid);

  // x[tid] = (float) threadIdx.x;
  x[tid] += (float) y[tid];

}


//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x, *h_y, *d_y;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  for (int i=0; i<nsize; i++) 
    h_x[i] = i;

  h_y = (float *)malloc(nsize*sizeof(float));
  for (int i=0; i<nsize; i++) 
    h_y[i] = i;

  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, nsize*sizeof(float)));

  checkCudaErrors( cudaMemcpy(d_x,h_x,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_y,h_y,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );


  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_y);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );
  // checkCudaErrors( cudaMemcpy(h_y,d_y,nsize*sizeof(float),
                //  cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
