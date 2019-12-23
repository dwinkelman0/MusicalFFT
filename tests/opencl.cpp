#include "opencl_fixture.h"

#include <gtest/gtest.h>

#include <stdexcept>


TEST_F(OpenCLTest, BasicContext)
{
	cl_kernel kernel = ctx->createKernel("vector_add", "../kernels/vector_add.cl");
}