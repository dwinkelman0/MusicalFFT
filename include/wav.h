#ifndef _WAV_H_
#define _WAV_H_

#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>


class WavFile
{
public:
	WavFile(const std::string& fname);

	~WavFile();

	float getSampleRate() const
	{
		return (float)sample_rate;
	}

	float getNumChannels() const
	{
		return n_channels;
	}

	size_t readSeconds(const float seconds, const std::vector<float*> outputs);

	size_t readSamples(const size_t n_samples, const std::vector<float*> outputs);

	size_t skipSeconds(const float seconds);

	size_t skipSamples(const size_t samples);

protected:
	uint16_t read16();
	uint32_t read32();
	uint32_t read32be();

protected:
	uint32_t sample_rate;
	uint32_t n_channels;
	uint32_t block_align;
	uint32_t sample_size;

	std::ifstream ist;
	int64_t data_bytes_remaining;
};




#endif