#ifndef _TEST_FFT_FIXTURE_H_
#define _TEST_FFT_FIXTURE_H_

#include <fftsw.h>

#include <gtest/gtest.h>

#include <complex>
#include <math.h>
#include <stdint.h>


/*! Test fixture for everything FFT */
class FFTTest : public ::testing::Test
{
protected:
	void SetUp() override;

	/*! Check that two Fourier coefficients are close */
	static bool CheckCoefficients(const complex_t c1, const complex_t c2, const uint32_t index);

protected:
	const static uint32_t n_samples = 1024;
	float data[n_samples];
};


#endif