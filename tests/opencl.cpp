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


TEST_F(OpenCLTest, MemoryReadWrite)
{
	std::vector<OpenCLDevice*> devices = ctx->getDevices();
	OpenCLMemory* gpu_mem = devices[0]->newMemory(16 * sizeof(uint32_t));

	uint32_t test_data[32];
	for (int i = 0; i < 32; ++i)
	{
		test_data[i] = i;
	}

	gpu_mem->write(reinterpret_cast<uint8_t*>(test_data), 16 * sizeof(uint32_t));

	uint32_t output[16];
	size_t n_written = 0;
	gpu_mem->read(16 * sizeof(uint32_t), reinterpret_cast<uint8_t*>(output), &n_written);

	EXPECT_EQ(n_written, 16 * sizeof(uint32_t));
	for (int i = 0; i < 16; ++i)
	{
		EXPECT_EQ(test_data[i], output[i]);
	}
}


TEST_F(OpenCLTest, FFT)
{
	cl_kernel kernel = ctx->createKernel("musical_fft", "../kernels/musical_fft.cl");
}