
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2, a, b, c;


////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void expression_calc(float *d_z, float *d_avg) {
  float sum = 0.0f;
  int ind = threadIdx.x + N*blockIdx.x*blockDim.x, z;

  for (int n=0; n<N; n++) {
    z = d_z[ind];
    sum += a*z*z + b+z + c;
    ind += blockDim.x;
  }

  d_avg[threadIdx.x + blockIdx.x*blockDim.x] = sum/N;
}


__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x; // 2*N because each iteration uses two random numbers

  // version 2
  // ind = 2*N*threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    // version 1
    ind += blockDim.x;      // shift pointer to next element
    // version 2
    // ind += 1; 

    y2   = rho*y1 + alpha*d_z[ind];
    // version 1
    ind += blockDim.x;      // shift pointer to next element
    // version 2
    // ind += 1; 

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int     NPATH=960000, h_N=200;
  int K = 2*h_N*NPATH;

  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2, h_a, h_b, h_c;
  float  *h_v, *h_avg, *d_v, *d_z, *d_avg;
  double  sum1, sum2; // sum is done in double precision for accuracy

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);
  h_avg = (float *)malloc(sizeof(float)*(K/200));

  // checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_avg, sizeof(float)*(K/200)) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  h_a = 1.0f;
  h_a = 2.0f;
  h_a = 3.0f;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, 2.0*h_N*NPATH/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);
  // pathcalc<<<NPATH/128, 128>>>(d_z, d_v);
  expression_calc<<<K/(200*128), 128>>>(d_z, d_avg);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("pathcalc execution failed\n");
  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  // checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
  //                  cudaMemcpyDeviceToHost) );
  
  checkCudaErrors( cudaMemcpy(h_avg, d_avg, sizeof(float)*(K/200),
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) { //sum is done on host. this can be done on device
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  for (int i=0; i<(K/200); i++) {
    sum1 += h_avg[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f\n\n",
	 sum1/(K/200));

  // printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	//  sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  // checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );
  checkCudaErrors( cudaFree(d_avg) );


  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
