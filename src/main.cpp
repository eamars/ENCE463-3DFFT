#include <iostream>

#include "opencl.hpp"
#include "cuda.hpp"

using std::cout;
using std::endl;
using std::string;


int main(void)
{
	run_opencl_cpu(512);
	run_opencl_gpu(512);
	run_cuda(512);

	return 0;

}




