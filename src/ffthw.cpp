#include "ffthw.h"

#include <iostream>
#include <vector>


#define FFT_SIZE 1024


float* musical_fft_hw(OpenCLContext* ctx, const float data_freq, const size_t n_data, const float* data, const float base_note_freq, const uint32_t n_chunks, size_t* n_output)
{
	cl_event event;

	// Compile the kernel
	cl_kernel kernel = ctx->createKernel("musical_fft", "../kernels/musical_fft.cl");
	std::vector<OpenCLDevice*> devices = ctx->getDevices();

	// Allocate memory for the input and load signal
	OpenCLMemory* input_buffer = devices[0]->newMemory(n_data * sizeof(float));
	cl_mem input_handle = input_buffer->getHandle();
	input_buffer->write(reinterpret_cast<const uint8_t*>(data), n_data * sizeof(float));

	// Allocate memory for the output
	size_t output_size = n_chunks * FFT_SIZE * 6;
	*n_output = output_size;
	float* output = new float[output_size];
	OpenCLMemory* output_buffer = devices[0]->newMemory(output_size * sizeof(float));
	cl_mem output_handle = output_buffer->getHandle();

	// Set arguments
	cl_uint err = 0;
	err = clSetKernelArg(kernel, 0, sizeof(float), (void*)&data_freq);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 1, sizeof(unsigned int), (void*)&n_data);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&input_handle);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 3, sizeof(float), (void*)&base_note_freq);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&output_handle);
	checkError(err, "clSetKernelArg");

	// Kernel configuration
	cl_uint work_dim = 1;
	size_t global_work_offset[] = { 0 };
	size_t global_work_size[] = { 512 * n_chunks };
	size_t local_work_size[] = { 512 };
	
	// Execute the kernel
	err = clEnqueueNDRangeKernel(devices[0]->getCommandQueue(), kernel, work_dim, global_work_offset, global_work_size, local_work_size, 0, nullptr, &event);
	checkError(err, "clEnqueueNDRangeKernel");
	clWaitForEvents(1, &event);

	// Read output
	std::cout << output_size * sizeof(float) << std::endl;
	input_buffer->read(output_size * sizeof(float), reinterpret_cast<uint8_t*>(output), n_output);

	// Clean up
	delete input_buffer;
	input_buffer = nullptr;
	delete output_buffer;
	output_buffer = nullptr;

	return output;
}