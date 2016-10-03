#include "cuda.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

#include "utils.hpp"

float run_cuda(int N)
{
	// cuda perform profiler
	cudaEvent_t start, stop;
	float time_interval;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 3d transform
	int rank = 3;
	int batch_size = 1;
	int profile_count = 1;
	int direction = CUFFT_FORWARD;
	cufftType type = CUFFT_Z2Z;

	// 3 dimensions
	int n[3] = { N, N, N };

	// total length
	size_t vectorLength = N * N * N;

	// allocate host memory
	size_t mem_size = sizeof(double) * vectorLength * 2; //real:imaginary
	double * host_signal = (double *)malloc(mem_size);

	// allocate device memory
	double * device_signal;
	cudaMalloc((void **)&device_signal, mem_size);

	// create cufft plan
	cufftHandle plan;
	cufftPlanMany(&plan, rank, n, NULL, 1, (int)vectorLength,
		NULL, 1, (int)vectorLength, type, batch_size);

	// transform signal and kernel
	// forward cufft
	// copy from memory
	matrix_generator(host_signal, n);

	// copy host memory to device
	cudaMemcpy(device_signal, host_signal, mem_size, cudaMemcpyHostToDevice);

	// start the timer
	cudaEventRecord(start, 0);

	// perform z2z fft
	cufftExecZ2Z(plan, (cufftDoubleComplex *)device_signal, (cufftDoubleComplex *)device_signal, direction);

	// stop the timer
	cudaEventRecord(stop, 0);

	// wait for the execution
	cudaDeviceSynchronize();

	// print performance
	cudaEventElapsedTime(&time_interval, start, stop);
	// printf("Total time taken for performing FFT on GPU (CUDA): %0.6f\n", time_interval);

	// copy device memory to host
	cudaMemcpy(host_signal, device_signal, mem_size, cudaMemcpyDeviceToHost);

	printf("CUDA result\n");
	for (size_t i = 0; i<n[0]; ++i) {
		for (size_t j = 0; j<n[1]; ++j) {
			for (size_t k = 0; k<n[2]; ++k) {
				size_t idx = 2 * (k + j*n[2] + i*n[1] * n[2]);
				printf("(%f, %f) ", host_signal[idx], host_signal[idx + 1]);
			}
			printf("\n");
		}
		printf("\n");
	}
	

	// destroy cufft context
	cufftDestroy(plan);

	// clean up memory
	free(host_signal);
	cudaFree(device_signal);

	cudaDeviceReset();

	return time_interval;
}

