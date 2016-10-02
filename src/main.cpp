#include <iostream>

#include "opencl.hpp"
#include "cuda.hpp"

using std::cout;
using std::endl;
using std::string;


int main(void)
{
	run_opencl(2);
	run_cuda(2);

	return 0;

}




