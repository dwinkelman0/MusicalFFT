#ifndef _FFTSW_H_
#define _FFTSW_H_

#include <complex>


/*! Complex numbers */
typedef std::complex<float> complex_t;


/*! Compute the Fourier Transform of a signal using the FFT algorithm
 *    @param data: raw data in the time domain
 *    @param n_samples: a power of 2; the number of elements in data
 *    @param output: output buffer which has half as many elements as data
 */
void fft_sw(const float* data, const uint32_t n_samples, complex_t* output);


/*! Compute the Fourier Transform of a signal using a naive DFT implementation
 *    @param data: raw data in the time domain
 *    @param n_samples: a power of 2; the number of elements in data
 *    @param output: output buffer which has half as many elements as data
 */
void dft_sw(const float* data, const uint32_t n_samples, complex_t* output);


#endif