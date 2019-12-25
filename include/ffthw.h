#ifndef _FFTSW_H_
#define _FFTSW_H_

#include "opencl_context.h"
#include "opencl_mem.h"

#include <math.h>
#include <iostream>
#include <string.h>
#include <vector>

#define FFT_SIZE 1024
#define N_STAGES 10


class MusicalFFT
{
public:
	MusicalFFT(OpenCLContext* ctx);

	~MusicalFFT();

	void run(const float data_rate, const size_t n_signal, const float* signal, const float chunk_rate, const float base_note_freq);

	const float* readComplete(size_t* n_chunks, size_t* n_overtones_per_note);

	const float* readNotes(size_t* n_chunks, size_t* n_notes);

protected:
	static void waitForEvent(cl_event* event);

protected:
	OpenCLContext* ctx;

	cl_kernel fft_kernel;
	cl_event fft_kernel_done;
	OpenCLWriteOnlyMemory* fft_input_mem;
	OpenCLReadOnlyMemory* fft_output_mem;

	cl_kernel notes_kernel;
	OpenCLReadOnlyMemory* notes_output_mem;

	size_t n_chunks;
};


#endif