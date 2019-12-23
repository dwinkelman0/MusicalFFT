#include "opencl_fixture.h"

#include <gtest/gtest.h>

#include <stdexcept>


TEST_F(OpenCLTest, BasicContext)
{
	cl_kernel kernel = ctx->createKernel("vector_add", "../kernels/vector_add.cl");
}


TEST_F(OpenCLTest, DeviceInfo)
{
	std::vector<OpenCLDevice*> devices = ctx->getDevices();
	std::cout << "Local memory size: " << devices[0]->getLocalMemorySize() << std::endl;
	std::cout << "Max workgroup size: " << devices[0]->getMaxWorkGroupSize() << std::endl;
	std::cout << "Max compute units: " << devices[0]->getMaxComputeUnits() << std::endl;
}


TEST_F(OpenCLTest, FFT)
{
	cl_kernel kernel = ctx->createKernel("musical_fft", "../kernels/musical_fft.cl");
}