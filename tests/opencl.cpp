#include "opencl_fixture.h"

#include <ffthw.h>

#include <gtest/gtest.h>

#include <math.h>
#include <stdexcept>


TEST_F(OpenCLTest, BasicContext)
{
	EXPECT_NO_THROW(ctx->createKernel("vector_add", "../kernels/vector_add.cl"));
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


TEST_F(OpenCLTest, MusicalFFT)
{
	const float data_freq = 44100;
	const uint32_t n_data = 44101;
	const float base_note_freq = 110; // A2

	// Generate a signal for a C#3 (and some overtones)
	float* data = new float[n_data];
	float note_freq = 138.59;
	for (int i = 0; i < n_data; ++i)
	{
		float sum = 0;
		for (int j = 1; j <= 16; ++j)
		{
			sum += sin(i / data_freq * 2*M_PI * note_freq * j) / j;
		}
		data[i] = sum;
	}

	size_t n_output = 0;
	float* output = musical_fft_hw(ctx, data_freq, n_data, data, base_note_freq, 100, &n_output);
}