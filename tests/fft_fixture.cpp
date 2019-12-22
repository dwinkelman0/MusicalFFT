#include "fft_fixture.h"

#include <math.h>


void FFTTest::SetUp()
{
	for (int i = 0; i < n_samples; ++i)
	{
		data[i] = std::sin((float)i / 16 * 2*M_PI) + 0.5 * std::sin((float)i / 14 * 2*M_PI);
	}
}


bool FFTTest::CheckCoefficients(const complex_t c1, const complex_t c2, const uint32_t index)
{
	return abs(std::abs(c1) - std::abs(c2)) < (std::abs(c1) / 50);
}