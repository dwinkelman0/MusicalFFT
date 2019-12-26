#include <wav.h>

#include <gtest/gtest.h>

#include <vector>


TEST(WavFile, BasicRead)
{
	WavFile file("../data/english_suite_4.wav");

	const size_t n_samples = 1000000;
	float* buffer_left = new float[n_samples];
	float* buffer_right = new float[n_samples];
	std::vector<float*> buffers {buffer_left, buffer_right};

	EXPECT_EQ(n_samples, file.readSamples(n_samples, buffers));

	delete[] buffer_left;
	buffer_left = nullptr;
	delete[] buffer_right;
	buffer_right = nullptr;
}