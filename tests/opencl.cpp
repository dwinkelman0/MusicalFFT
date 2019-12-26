#include "opencl_fixture.h"

#include <ffthw.h>
#include <opencl_mem.h>
#include <wav.h>

#include <gtest/gtest.h>

#include <math.h>
#include <stdexcept>
#include <stdio.h>


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
	mem->write(nullptr);
	memset(input_buffer, 0, mem_size);

	// Read the buffer
	const uint32_t* output_buffer = reinterpret_cast<const uint32_t*>(mem->read(nullptr));

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
	const uint32_t n_data = 44101 * 60;
	const float base_note_freq = 110; // A2

	// Generate a signal for a C#3 (and some overtones)
	float* data = new float[n_data];
	float note_freq = 4 * 138.59;
	for (int i = 0; i < n_data; ++i)
	{
		float sum = 0;
		for (int j = 1; j <= 1; ++j)
		{
			sum += sin(i / data_freq * 2*M_PI * note_freq * j) / j;
		}
		data[i] = sum;
	}

	MusicalFFT mfft(ctx);
	mfft.run(data_freq, n_data, data, 220, base_note_freq);

	size_t n_chunks, n_overtones_per_note;
	const float* complete_output = mfft.readComplete(&n_chunks, &n_overtones_per_note);

	std::cout << "Computed " << n_chunks << " chunks" << std::endl;

	for (int i = 0; i < 32; ++i)
	{
		printf("%2d: ", i);
		for (int j = 0; j < 12; ++j)
		{
			printf("%.2e | ", complete_output[FFT_SIZE / 2 * j + i]);
		}
		std::cout << std::endl;
	}

	size_t n_notes;
	const float* notes_output = mfft.readNotes(&n_chunks, &n_notes);

	for (int row = 0; row < n_notes / 12; ++row)
	{
		printf("%2d: ", row);
		for (int col = 0; col < 12; ++col)
		{
			printf("%.2e | ", notes_output[row * 12 + col]);
		}
		std::cout << std::endl;
	}
}


TEST_F(OpenCLTest, MusicalFFTRecording)
{
	WavFile file("../data/english_suite_4.wav");

	const size_t n_samples = (size_t)(file.getSampleRate() * 2);
	float* channel_left = new float[n_samples];
	float* channel_right = new float[n_samples];
	std::vector<float*> channels { channel_left, channel_right };

	// Read file
	EXPECT_EQ(n_samples, file.readSamples(n_samples, channels));

	MusicalFFT mfft(ctx);
	mfft.run(file.getSampleRate(), n_samples, channel_left, 200, 65.4064);

	size_t n_chunks, n_overtones_per_note;
	const float* complete_output = mfft.readComplete(&n_chunks, &n_overtones_per_note);

	std::cout << "Computed " << n_chunks << " chunks" << std::endl;

	for (int i = 0; i < 32; ++i)
	{
		printf("%2d: ", i);
		for (int j = 0; j < 12; ++j)
		{
			printf("%.2e | ", complete_output[6 * FFT_SIZE * 200 + FFT_SIZE / 2 * j + i]);
		}
		std::cout << std::endl;
	}

	size_t n_notes;
	const float* notes_output = mfft.readNotes(&n_chunks, &n_notes);

	for (int chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
	{
		const float* chunk = notes_output + chunk_id * n_notes;

		float loudest_mag = -1e10;
		int loudest_index = 0;
		for (int i = 0; i < n_notes; ++i)
		{
			if (chunk[i] > loudest_mag)
			{
				loudest_mag = chunk[i];
				loudest_index = i;
			}
		}
		std::cout << loudest_index << " --> " << loudest_mag << std::endl;
	}

	delete[] channel_left;
	channel_left = nullptr;
	delete[] channel_right;
	channel_right = nullptr;
}