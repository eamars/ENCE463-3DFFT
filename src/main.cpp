#include <iostream>

#include "opencl.hpp"
#include "cuda.hpp"

using std::cout;
using std::endl;
using std::string;


int main(void)
{
	run_opencl_cpu(4);
	run_opencl_gpu(4);
	run_cuda(4);

	return 0;

}




