#include "wav.h"

#include <endian.h>
#include <iostream>
#include <math.h>


WavFile::WavFile(const std::string& fname) :
	sample_rate(0),
	n_channels(0),
	block_align(0),
	sample_size(0),
	ist(std::ifstream(fname)),
	data_bytes_remaining(0)
{
	// http://soundfile.sapp.org/doc/WaveFormat/
	// http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
	char buffer[4];

	// Validate first 12 bytes
	if (read32be() != 0x52494646)
	{
		throw std::runtime_error("Invalid file format (RIFF)");
	}
	data_bytes_remaining = read32();
	if (read32be()!= 0x57415645)
	{
		throw std::runtime_error("Invalid file format (WAVE)");
	}

	// Iterate through each subchunk until data is found
	bool seen_fmt = false;
	bool seen_data = false;
	while (data_bytes_remaining > 0)
	{
		uint32_t subchunk_type = read32be();
		size_t subchunk_size = read32();

		if (subchunk_type == 0x666d7420)
		{
			// Format subchunk
			seen_fmt = true;

			// Check format
			uint16_t format_type = read16();
			if (format_type != 1)
			{
				throw std::runtime_error("Non-PCM is not supported");
			}

			n_channels = read16();

			sample_rate = read32();
			uint32_t byte_rate = read32();

			block_align = read16();
			sample_size = read16();
			if (sample_size != 16)
			{
				throw std::runtime_error("Samples are not a proper size");
			}
			sample_size /= 8;

			// Jump ahead (since there could be extra data)
			ist.seekg(subchunk_size - 16, std::ios_base::cur);
			data_bytes_remaining -= subchunk_size - 16;
		}
		else if (subchunk_type == 0x64617461)
		{
			// Data subchunk
			seen_data = true;

			if (!seen_fmt)
			{
				throw std::runtime_error("DATA section precedes FMT section");
			}
			data_bytes_remaining = subchunk_size;
			break;
		}
		else
		{
			// Skip the rest of the subchunk
			ist.seekg(subchunk_size, std::ios_base::cur);
			data_bytes_remaining -= subchunk_size;
		}
	}

	if (!seen_fmt)
	{
		throw std::runtime_error("Did not find a FMT subchunk");
	}
	else if (!seen_data)
	{
		throw std::runtime_error("Did not find a DATA subchunk");
	}
}


WavFile::~WavFile()
{
	ist.close();
}


size_t WavFile::readSeconds(const float seconds, const std::vector<float*> outputs)
{
	return readSamples((size_t)floor(seconds * sample_rate), outputs);
}


size_t WavFile::readSamples(const size_t samples, const std::vector<float*> outputs)
{
	if (outputs.size() != n_channels)
	{
		throw std::runtime_error("There must be as many output buffers as there are channels");
	}

	const float factor = 1 / (float)(1 << (sample_size * 8 - 1));

	// Set up the buffer
	const size_t buffer_size = 65536;
	char buffer[buffer_size];
	const size_t buffer_capacity = buffer_size / block_align;

	// Determine how many samples to read
	const size_t samples_in_file = data_bytes_remaining / block_align;
	const size_t total_samples = samples < samples_in_file ? samples : samples_in_file;

	size_t output_offset = 0;
	size_t samples_left = total_samples;

	while (samples_left > 0)
	{
		size_t samples_to_read = buffer_capacity < samples_left ? buffer_capacity : samples_left;
		ist.read(buffer, samples_to_read * block_align);
		samples_left -= samples_to_read;

		for (int channel = 0; channel < outputs.size(); ++channel)
		{
			float* output = outputs[channel];
			for (int j = 0; j < samples_to_read; ++j)
			{
				output[output_offset + j] = le16toh(reinterpret_cast<int16_t*>(buffer)[j * n_channels + channel]) * factor;
			}
		}

		output_offset += samples_to_read;
	}

	return total_samples;
}


uint16_t WavFile::read16()
{
	uint16_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 2);
	data_bytes_remaining -= 2;
	return le16toh(output);
}


uint32_t WavFile::read32()
{
	uint32_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 4);
	data_bytes_remaining -= 4;
	return le32toh(output);
}


uint32_t WavFile::read32be()
{
	uint32_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 4);
	data_bytes_remaining -= 4;
	return be32toh(output);
}