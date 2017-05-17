// NOTE: h_ prefix means "host"   (CPU)
//       d_ prefix means "device" (GPU)
#include <iostream>
#include <fstream>
#include <cuda.h>
using namespace std;

// GPU code
__global__ void soft_to_hard(double* soft, int* hard) {
  int ix = (blockDim.x * blockIdx.x) + threadIdx.x;

  hard[ix] = (soft[ix] > 0);  // XXX: Is this faster than an 'if'?
}


// CPU code
int main(int argc, char* argv[]) {
  if (argc != 2) {
    cout << "Wrong number of arguments. Expected 1, got " << (argc-1)
         << endl;
    return 1;
  }

  // Read values from input file
  ifstream file(argv[1]);
  int softCount;
  file >> softCount;

  double* h_soft = new double[softCount];

  for (int i = 0; i < softCount; ++i) {
    file >> h_soft[i];
  }

  file.close();

  // Copy values to device memory
  double* d_soft;
  cudaMalloc((void**)&d_soft, softCount*sizeof(double));
  cudaMemcpy(d_soft, h_soft, softCount*sizeof(double), cudaMemcpyHostToDevice);

  int* d_hard;
  cudaMalloc((void**)&d_hard, softCount*sizeof(int));

  // Set up the computational grid
  int threadsPerBlock = 2;
  int blocksPerGrid   = (softCount + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  soft_to_hard<<<blocksPerGrid, threadsPerBlock>>>(d_soft, d_hard);

  // Copy results to host memory
  int* h_hard = new int[softCount];
  cudaMemcpy(h_hard, d_hard, softCount*sizeof(int), cudaMemcpyDeviceToHost);

  // Print results
  for (int i = 0; i < softCount; ++i) {
    cout << h_hard[i] << ' ';
  }
  cout << endl;

  cudaFree(d_soft);
  cudaFree(d_hard);
  delete[] h_soft;
  delete[] h_hard;

  return 0;
}

