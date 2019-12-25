#ifndef _FFTSW_H_
#define _FFTSW_H_

#include "opencl_context.h"
#include "opencl_mem.h"

#include <math.h>
#include <iostream>
#include <string.h>
#include <vector>

#define FFT_SIZE 1024


OpenCLReadOnlyMemory* musical_fft_hw(OpenCLContext* ctx, const float data_rate, const size_t n_signal, const float* signal, const float chunk_rate, const float base_note_freq)
{
	// Calculate number of samples per chunk
	float samples_per_chunk = data_rate / chunk_rate;

	// Calculate number of samples for the longest frequency
	// NOTE: not good practice to have base_note_freq > chunk_rate
	float samples_per_base_note = data_rate / base_note_freq;
	float max_base_note_freq = base_note_freq * pow(2, 11.0f / 12);
	if (max_base_note_freq > chunk_rate)
	{
		throw std::runtime_error("The highest base note frequency is higher than the chunk rate");
	}

	// Calculate number of chunks that can be done with amount of data supplied
	float max_chunks = ((n_signal - 3) - samples_per_base_note) / samples_per_chunk + 1;
	size_t n_chunks = (size_t)floor(max_chunks);
	if (n_chunks == 0)
	{
		throw std::runtime_error("Cannot have 0 chunks");
	}

	// Compile the kernel
	cl_kernel kernel = ctx->createKernel("musical_fft", "../kernels/musical_fft.cl");

	// Create buffers for the input and output
	std::vector<OpenCLDevice*> devices = ctx->getDevices();
	OpenCLWriteOnlyMemory* input_mem = new OpenCLWriteOnlyMemory(devices[0], n_signal * sizeof(float), CL_MEM_READ_ONLY);
	OpenCLReadOnlyMemory* output_mem = new OpenCLReadOnlyMemory(devices[0], n_chunks * FFT_SIZE * 6 * sizeof(float), CL_MEM_WRITE_ONLY);

	// Write signal to device memory
	uint8_t* signal_buffer = input_mem->getWriteableBuffer();
	memcpy(signal_buffer, signal, input_mem->getSize());
	input_mem->write();

	// Set up arguments
	cl_int err = 0;
	input_mem->setAsKernelArgument(kernel, 0);
	err = clSetKernelArg(kernel, 1, sizeof(float), (void*)&samples_per_chunk);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 2, sizeof(float), (void*)&samples_per_base_note);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 3, (size_t)floor(samples_per_base_note + 2) * sizeof(cl_float), nullptr);
	checkError(err, "clSetKernelArg");
	output_mem->setAsKernelArgument(kernel, 4);

	// Kernel execution configuration
	cl_uint work_dim = 1;
	size_t global_work_offset[] = { 0 };
	size_t global_work_size[] = { (FFT_SIZE / 2) * n_chunks };
	size_t local_work_size[] = { FFT_SIZE / 2 };

	// Execute kernel
	cl_event kernel_exec;
	err = clEnqueueNDRangeKernel(devices[0]->getCommandQueue(), kernel, work_dim, global_work_offset, global_work_size, local_work_size, 0, nullptr, &kernel_exec);
	checkError(err, "clEnqueueNDRangeKernel");
	clWaitForEvents(1, &kernel_exec);

	// Clean up
	delete input_mem;
	input_mem = nullptr;

	return output_mem;
}


#endif