#ifndef _FFTSW_H_
#define _FFTSW_H_

#include "opencl_context.h"


float* musical_fft_hw(OpenCLContext* ctx, const float data_freq, const size_t n_data, const float* data, const float base_note_freq, const uint32_t n_chunks, size_t* n_output);


#endif