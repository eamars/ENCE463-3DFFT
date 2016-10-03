#include <iostream>

#include "opencl.hpp"
#include "cuda.hpp"
#include "opencl.hpp"

#include <iostream>
#include <stdlib.h>

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Invalid argument\n");
		return -1;
	}

	int d = atoi(argv[1]);

	if (d == 0)
	{
		printf("Invalid dimension size\n");
		return -2;
	}

	float ocl_c, ocl_g, cuda_g;

	ocl_c = run_opencl_cpu(d);
	ocl_g = run_opencl_gpu(d);
	cuda_g = run_cuda(d);

	printf("%d, %0.6f, %0.6f, %0.6f\n", d, ocl_c, ocl_g, cuda_g);
	
	return 0;
}




