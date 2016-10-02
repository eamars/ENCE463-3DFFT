#include "cuda.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

int run_cuda(int N)
{
	// fft parameters
	const size_t N0 = N, N1 = N, N2 = N;
	double *X;

	size_t matrix_dim[3] = { N0, N1, N2 };

	// create 3d fft plan
	cufftHandle plan_handle;
	cufftComplex *cu_buffer;
	
	// allocate memory
	size_t buffer_size = N0 * N1 * N2;
	cudaMalloc((void**)&cu_buffer, buffer_size * sizeof(cufftComplex));

	// create a 3d fft plan
	cufftPlan3d(&plan_handle, N0, N1, N2, CUFFT_C2C);

	// transform
	cufftExecC2C(plan_handle, cu_buffer, cu_buffer, CUFFT_FORWARD);

	// cleanup
	cufftDestroy(plan_handle);

	cudaFree(cu_buffer);

	return 0;

}