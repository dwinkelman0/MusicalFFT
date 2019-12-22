#include "fft_fixture.h"

#include <fftsw.h>

#include <complex>


TEST_F(FFTTest, SoftwareFFTMatchesDFT)
{
	complex_t fft_buffer[n_samples / 2];
	fft_sw(data, n_samples, fft_buffer);

	complex_t dft_buffer[n_samples / 2];
	dft_sw(data, n_samples, dft_buffer);

	for (int k = 0; k < n_samples / 2; ++k)
	{
		EXPECT_PRED3(CheckCoefficients, fft_buffer[k], dft_buffer[k], k);
	}
}