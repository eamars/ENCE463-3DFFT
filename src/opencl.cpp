#include <clFFT.h>
#include <CL/cl.h>
#include <iostream>
#include <string>

#include "utils.hpp"

const char CPU_PLATFORM_NAME[] = "Intel(R) OpenCL";
const char GPU_PLATFORM_NAME[] = "NVIDIA CUDA";


float run_opencl_cpu(int N)
{
	// initialize OCL variables
	cl_int err;

	// performance measure
	cl_ulong time_start, time_end, time_interval;

	// event performance counter
	cl_event event = NULL;

	cl_uint platform_count;
	cl_platform_id *platforms;

	// get platform counts
	clGetPlatformIDs(5, NULL, &platform_count);
	// printf("Found %d platforms\n", platform_count);

	// get all platforms
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
	clGetPlatformIDs(platform_count, platforms, NULL);

	// cpu and gpu platforms
	cl_platform_id platform_cpu;

	// gpu and cpu devices
	cl_device_id device_cpu;

	cl_context_properties properties_cpu[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	cl_context context_cpu;
	cl_command_queue queue_cpu;

	// print platfrom names
	for (cl_uint i = 0; i < platform_count; i++)
	{
		char platform_names[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_names), platform_names, NULL);

		// get cpu platform
		if (strcmp(platform_names, CPU_PLATFORM_NAME) == 0)
		{
			platform_cpu = platforms[i];
		}
	}

	free(platforms);

	// retrieve devices from platform
	err = clGetDeviceIDs(platform_cpu, CL_DEVICE_TYPE_CPU, 1, &device_cpu, NULL);

	// list select devices
	char platform_name[128];
	char device_name[128];

	/*
	clGetPlatformInfo(platform_cpu, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
	printf("[CPU Platform] %s\n", platform_name);

	clGetDeviceInfo(device_cpu, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	printf("\t%s\n", device_name);
	*/

	// setup contexts
	properties_cpu[1] = (cl_context_properties)platform_cpu;

	context_cpu = clCreateContext(properties_cpu, 1, &device_cpu, NULL, NULL, &err);

	queue_cpu = clCreateCommandQueue(context_cpu, device_cpu, CL_QUEUE_PROFILING_ENABLE, &err);

	// calculation parameters
	const size_t N0 = N, N1 = N, N2 = N;
	double *X;

	size_t cl_lengths[3] = { N0, N1, N2 };

	// allocate data
	size_t buffer_size = N0 * N1 * N2 * 2 * sizeof(*X);
	X = (double *)malloc(buffer_size);

	// create data
	matrix_generator(X, cl_lengths);

	// opencl_fft(context_gpu, queue_gpu, X, cl_lengths);
	cl_context context = context_cpu;
	cl_command_queue queue = queue_cpu;

	// setup clFFT library
	clfftSetupData fft_setup;
	err = clfftInitSetupData(&fft_setup);
	err = clfftSetup(&fft_setup);

	// declear FFT library related variables
	clfftPlanHandle plan_handle;
	clfftDim dim = CLFFT_3D;

	// prepare opencl memory object for data transfer
	cl_mem cl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);

	// enqueue the buffer
	err = clEnqueueWriteBuffer(queue, cl_buffer, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL);

	// create a default plan for a complex FFT
	err = clfftCreateDefaultPlan(&plan_handle, context, dim, cl_lengths);

	// set plan parameters
	err = clfftSetPlanPrecision(plan_handle, CLFFT_DOUBLE);
	err = clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(plan_handle, CLFFT_INPLACE);

	// bake the plan
	err = clfftBakePlan(plan_handle, 1, &queue, NULL, NULL);

	// execute the plan
	err = clfftEnqueueTransform(plan_handle, CLFFT_FORWARD, 1, &queue, 0, NULL, &event, &cl_buffer, NULL, NULL);

	// wait for calculation to be finished
	err = clFinish(queue);

	// print performance
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	time_interval = time_end - time_start;
	// printf("Total time taken for performing FFT on CPU: %0.6f\n", (time_interval / 1000000.0f));

	// fetch the result
	err = clEnqueueReadBuffer(queue, cl_buffer, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL);

	/*
	// print the output array
	printf("OpenCL CPU result\n");
	for (size_t i = 0; i<N0; ++i) {
		for (size_t j = 0; j<N1; ++j) {
			for (size_t k = 0; k<N2; ++k) {
				size_t idx = 2 * (k + j*N2 + i*N1*N2);
				printf("(%f, %f) ", X[idx], X[idx + 1]);
			}
			printf("\n");
		}
		printf("\n");
	}
	*/

	// release the opencl memory object
	clReleaseMemObject(cl_buffer);
	free(X);

	// release the plan
	clfftTeardown();	

	// release opencl working objects
	clReleaseCommandQueue(queue_cpu);
	clReleaseContext(context_cpu);
	clReleaseEvent(event);

	return (time_interval / 1000000.0f);
}


float run_opencl_gpu(int N)
{
	// initialize OCL variables
	cl_int err;

	// performance measure
	cl_ulong time_start, time_end, time_interval;

	// event performance counter
	cl_event event = NULL;

	cl_uint platform_count;
	cl_platform_id *platforms;

	// get platform counts
	clGetPlatformIDs(5, NULL, &platform_count);
	// printf("Found %d platforms\n", platform_count);
	

	// get all platforms
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
	clGetPlatformIDs(platform_count, platforms, NULL);

	// cpu and gpu platforms
	cl_platform_id platform_gpu;

	// gpu and cpu devices
	cl_device_id device_gpu;

	cl_context_properties properties_gpu[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	cl_context context_gpu;
	cl_command_queue queue_gpu;

	// print platfrom names
	for (cl_uint i = 0; i < platform_count; i++)
	{
		char platform_names[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_names), platform_names, NULL);

		// select gpu platform
		if (strcmp(platform_names, GPU_PLATFORM_NAME) == 0)
		{
			platform_gpu = platforms[i];
		}
	}

	free(platforms);

	// retrieve devices from platform
	err = clGetDeviceIDs(platform_gpu, CL_DEVICE_TYPE_GPU, 1, &device_gpu, NULL);

	// list select devices
	char platform_name[128];
	char device_name[128];

	/*
	clGetPlatformInfo(platform_gpu, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
	printf("[GPU Platform] %s\n", platform_name);

	clGetDeviceInfo(device_gpu, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	printf("\t%s\n", device_name);
	*/

	// setup contexts
	properties_gpu[1] = (cl_context_properties)platform_gpu;

	context_gpu = clCreateContext(properties_gpu, 1, &device_gpu, NULL, NULL, &err);

	queue_gpu = clCreateCommandQueue(context_gpu, device_gpu, CL_QUEUE_PROFILING_ENABLE, &err);

	// calculation parameters
	const size_t N0 = N, N1 = N, N2 = N;
	double *X;

	size_t cl_lengths[3] = { N0, N1, N2 };

	// allocate data
	size_t buffer_size = N0 * N1 * N2 * 2 * sizeof(*X);
	X = (double *)malloc(buffer_size);

	// create data
	matrix_generator(X, cl_lengths);

	// opencl_fft(context_gpu, queue_gpu, X, cl_lengths);
	cl_context context = context_gpu;
	cl_command_queue queue = queue_gpu;

	// setup clFFT library
	clfftSetupData fft_setup;
	err = clfftInitSetupData(&fft_setup);
	err = clfftSetup(&fft_setup);

	// declear FFT library related variables
	clfftPlanHandle plan_handle;
	clfftDim dim = CLFFT_3D;

	// prepare opencl memory object for data transfer
	cl_mem cl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);

	// enqueue the buffer
	err = clEnqueueWriteBuffer(queue, cl_buffer, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL);

	// create a default plan for a complex FFT
	err = clfftCreateDefaultPlan(&plan_handle, context, dim, cl_lengths);

	// set plan parameters
	err = clfftSetPlanPrecision(plan_handle, CLFFT_DOUBLE);
	err = clfftSetLayout(plan_handle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	err = clfftSetResultLocation(plan_handle, CLFFT_INPLACE);

	// bake the plan
	err = clfftBakePlan(plan_handle, 1, &queue, NULL, NULL);

	// execute the plan
	err = clfftEnqueueTransform(plan_handle, CLFFT_FORWARD, 1, &queue, 0, NULL, &event, &cl_buffer, NULL, NULL);

	// wait for calculation to be finished
	err = clFinish(queue);

	// print performance
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	time_interval = time_end - time_start;
	// printf("Total time taken for performing FFT on GPU: %0.6f\n", (time_interval / 1000000.0f));

	// fetch the result
	err = clEnqueueReadBuffer(queue, cl_buffer, CL_TRUE, 0, buffer_size, X, 0, NULL, NULL);

	// print the output array
	/*
	printf("OpenCL GPU Result\n");
	for (size_t i = 0; i<N0; ++i) {
	for (size_t j = 0; j<N1; ++j) {
	for (size_t k = 0; k<N2; ++k) {
	size_t idx = 2 * (k + j*N2 + i*N1*N2);
	printf("(%f, %f) ", X[idx], X[idx + 1]);
	}
	printf("\n");
	}
	printf("\n");
	}
	*/

	// release the opencl memory object
	clReleaseMemObject(cl_buffer);
	free(X);

	// release the plan
	clfftTeardown();

	

	// release opencl working objects
	clReleaseCommandQueue(queue_gpu);
	clReleaseContext(context_gpu);
	clReleaseEvent(event);

	return (time_interval / 1000000.0f);
}