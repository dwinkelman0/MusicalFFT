#include "opencl_fixture.h"

#include <ffthw.h>
#include <opencl_mem.h>

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
	// Set up a buffer
	const size_t mem_size = 16 * sizeof(uint32_t);
	std::vector<OpenCLDevice*> devices = ctx->getDevices();
	OpenCLReadWriteMemory* mem = new OpenCLReadWriteMemory(devices[0], mem_size, 0);

	// Generate test data
	uint32_t test_data[32];
	for (int i = 0; i < 32; ++i)
	{
		test_data[i] = i;
	}

	// Write a portion of the data to buffer
	uint8_t* input_buffer = mem->getWriteableBuffer();
	memcpy(input_buffer, test_data, mem_size);
	mem->write();
	memset(input_buffer, 0, mem_size);

	// Read the buffer
	const uint32_t* output_buffer = reinterpret_cast<const uint32_t*>(mem->read());

	// Check that the new values match the original
	for (int i = 0; i < 16; ++i)
	{
		EXPECT_EQ(test_data[i], output_buffer[i]);
	}

	delete mem;
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

	OpenCLReadOnlyMemory* mem = musical_fft_hw(ctx, data_freq, n_data, data, 220, base_note_freq);
	const float* output = reinterpret_cast<const float*>(mem->read());
}