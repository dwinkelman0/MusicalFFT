#include "ffthw.h"

#include <iostream>


MusicalFFT::MusicalFFT(OpenCLContext* ctx) :
	ctx(ctx),
	fft_kernel(nullptr),
	fft_kernel_done(nullptr),
	fft_input_mem(nullptr),
	fft_output_mem(nullptr),
	n_chunks(0)
{}


MusicalFFT::~MusicalFFT()
{
	if (fft_kernel)
	{
		cl_int err = clReleaseKernel(fft_kernel);
		checkError(err, "clReleaseKernel");
		fft_kernel = nullptr;
	}
	if (fft_kernel_done)
	{
		cl_int err = clReleaseEvent(fft_kernel_done);
		checkError(err, "clReleaseEvent");
		fft_kernel_done = nullptr;
	}
	if (fft_input_mem)
	{
		delete fft_input_mem;
		fft_input_mem = nullptr;
	}
	if (fft_output_mem)
	{
		delete fft_output_mem;
		fft_output_mem = nullptr;
	}
}


void MusicalFFT::run(const float data_rate, const size_t n_signal, const float* signal, const float chunk_rate, const float base_note_freq)
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
	n_chunks = (size_t)floor(max_chunks);
	if (n_chunks == 0)
	{
		throw std::runtime_error("Cannot have 0 chunks");
	}

	// Compile the kernel
	if (!fft_kernel)
	{
		std::cout << "Compile kernel" << std::endl;
		fft_kernel = ctx->createKernel("musical_fft", "../kernels/musical_fft.cl");
	}

	// Create buffers for the input and output
	// If the buffer is the wrong size, delete and resize
	std::vector<OpenCLDevice*> devices = ctx->getDevices();

	size_t fft_input_mem_size = n_signal * sizeof(float);
	if (fft_input_mem && fft_input_mem->getSize() != fft_input_mem_size)
	{
		delete fft_input_mem;
		fft_input_mem = nullptr;
	}
	if (!fft_input_mem)
	{
		fft_input_mem = new OpenCLWriteOnlyMemory(devices[0], fft_input_mem_size, CL_MEM_READ_ONLY);
	}

	size_t fft_output_mem_size = n_chunks * FFT_SIZE * 6 * sizeof(float);
	if (fft_output_mem && fft_output_mem->getSize() != fft_output_mem_size)
	{
		delete fft_output_mem;
		fft_output_mem = nullptr;
	}
	if (!fft_output_mem)
	{
		fft_output_mem = new OpenCLReadOnlyMemory(devices[0], fft_output_mem_size, CL_MEM_WRITE_ONLY);
	}

	// Write signal to device memory
	uint8_t* signal_buffer = fft_input_mem->getWriteableBuffer();
	memcpy(signal_buffer, signal, fft_input_mem->getSize());
	fft_input_mem->write();

	// Set up arguments
	cl_int err = 0;
	fft_input_mem->setAsKernelArgument(fft_kernel, 0);
	err = clSetKernelArg(fft_kernel, 1, sizeof(float), (void*)&samples_per_chunk);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(fft_kernel, 2, sizeof(float), (void*)&samples_per_base_note);
	checkError(err, "clSetKernelArg");
	err = clSetKernelArg(fft_kernel, 3, (size_t)floor(samples_per_base_note + 2) * sizeof(cl_float), nullptr);
	checkError(err, "clSetKernelArg");
	fft_output_mem->setAsKernelArgument(fft_kernel, 4);

	// Kernel execution configuration
	cl_uint work_dim = 1;
	size_t global_work_offset[] = { 0 };
	size_t global_work_size[] = { (FFT_SIZE / 2) * n_chunks };
	size_t local_work_size[] = { FFT_SIZE / 2 };

	// Execute kernel
	err = clEnqueueNDRangeKernel(devices[0]->getCommandQueue(), fft_kernel, work_dim, global_work_offset, global_work_size, local_work_size, 0, nullptr, &fft_kernel_done);
	checkError(err, "clEnqueueNDRangeKernel");
}


const float* MusicalFFT::readComplete(size_t* n_chunks, size_t* n_overtones_per_note)
{
	// Make sure the computation executed and completed
	if (!fft_output_mem) return nullptr;
	if (fft_kernel_done)
	{
		cl_int err = 0;
		err = clWaitForEvents(1, &fft_kernel_done);
		checkError(err, "clWaitForEvents");
		err = clReleaseEvent(fft_kernel_done);
		checkError(err, "clReleaseEvent");
		fft_kernel_done = nullptr;
	}

	// Retrieve output from the buffer
	*n_chunks = this->n_chunks;
	*n_overtones_per_note = FFT_SIZE / 2;
	return reinterpret_cast<const float*>(fft_output_mem->read());
}