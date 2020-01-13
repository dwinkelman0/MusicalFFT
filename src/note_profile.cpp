#include "note_profile.h"

#include "ffthw.h"
#include "wav.h"

#include <iostream>


NoteProfile::NoteProfile(const int32_t base_note_id) :
	timestamps(nullptr),
	n_samples_per_second(0),
	notes(nullptr),
	n_chunks(0),
	n_notes_per_chunk(12 * (N_STAGES - 1)),
	base_note_id(base_note_id),
	n_samples_per_chunk(0)
{}


NoteProfile::~NoteProfile()
{
	if (timestamps)
	{
		delete[] timestamps;
		timestamps = nullptr;
	}
	if (notes)
	{
		delete[] notes;
		notes = nullptr;
	}
}


void NoteProfile::fromWav(const std::string& fname, const float a4_freq, const size_t n_samples_per_chunk)
{
	WavFile file(fname);
	MusicalFFT mfft(OpenCLContext::getInstance());

	// Determine parametrizations of note frequency
	const float base_note_freq = a4_freq * pow(2, (float)(base_note_id - 69) / 12.0);
	const float samples_per_base_note = file.getSampleRate() / base_note_freq;

	// Determine how much memory to allocate for notes
	this->n_samples_per_chunk = n_samples_per_chunk;
	const size_t n_total_samples = file.getNumSamplesRemaining();
	n_chunks = (n_total_samples - 3 - (size_t)ceil(samples_per_base_note)) / n_samples_per_chunk + 1;

	// Allocate memory for output
	timestamps = new uint64_t[n_chunks];
	notes = new float[n_chunks * n_notes_per_chunk];

	// The number of chunks processed at a time is dependent on the rate at
	// which the audio file is read
	const size_t buffer_size = file.getSampleRate() * 5;
	std::vector<float*> buffers(file.getNumChannels());
	for (size_t i = 0; i < file.getNumChannels(); ++i)
	{
		buffers[i] = new float[buffer_size];
	}

	// The FFT will not consume all samples, so there will usually be an offset
	size_t n_unused_samples = 0;
	std::vector<float*> buffers_with_offset(buffers);

	// The FFT will return a certain number of chunks that will not increase
	// over time; to aggregate over multiple channels, have another buffer
	float* aggregation_buffer = nullptr;

	// Keep track of how many notes have been processed
	size_t chunk_index = 0;

	while (1)
	{
		// Move remaining samples to the beginning of the buffers and define
		// buffer pointers with an offset
		for (size_t i = 0; i < file.getNumChannels(); ++i)
		{
			memcpy(buffers[i], buffers[i] + buffer_size - n_unused_samples, n_unused_samples * sizeof(float));
			buffers_with_offset[i] = buffers[i] + n_unused_samples;
		}
		const size_t n_samples_to_read = buffer_size - n_unused_samples;

		// Read as many samples as possible into the buffers
		size_t n_samples_read = file.readSamples(n_samples_to_read, buffers_with_offset);
		if (n_samples_read == 0) break;

		// Perform the FFT and aggregate the data
		size_t n_samples_to_process = n_samples_read + n_unused_samples;
		size_t n_samples_processed = 0;
		size_t n_new_chunks = 0;
		size_t n_new_notes = 0;

		for (size_t channel_index = 0; channel_index < file.getNumChannels(); ++channel_index)
		{
			std::cout << n_samples_to_process << std::endl;
			// Perform the FFT
			n_new_chunks = mfft.runFFT(file.getSampleRate(), n_samples_to_process, buffers[channel_index], n_samples_per_chunk, base_note_freq);
			n_samples_processed = n_new_chunks * n_samples_per_chunk;
			n_unused_samples = n_samples_to_process - n_samples_processed;
			n_new_notes = n_new_chunks * n_notes_per_chunk;

			// Create a buffer for aggregating the results from all channels
			if (!aggregation_buffer)
			{
				aggregation_buffer = new float[n_new_notes];
			}
			if (channel_index == 0)
			{
				for (size_t i = 0; i < n_new_notes; ++i)
				{
					aggregation_buffer[i] = 0;
				}
			}

			// Add the results to the buffer
			const float* notes_output = mfft.readNotes(nullptr, nullptr);
			for (size_t i = 0; i < n_new_notes; ++i)
			{
				aggregation_buffer[i] += notes_output[i];
			}
		}

		// Average the signal and copy into the output
		for (size_t i = 0; i < n_new_notes; ++i)
		{
			aggregation_buffer[i] /= file.getNumChannels();
		}
		memcpy(notes + chunk_index * n_notes_per_chunk, aggregation_buffer, n_new_notes * sizeof(float));
		chunk_index += n_new_chunks;
	}

	// There might be one or two extra slots for FFT output
	n_chunks = chunk_index;

	// Fill in timestamps
	n_samples_per_second = file.getSampleRate();
	const size_t center_offset = samples_per_base_note / 2;
	for (size_t i = 0; i < n_chunks; ++i)
	{
		timestamps[i] = i * n_samples_per_chunk + center_offset;
	}

	// Clean up
	for (size_t i = 0; i < file.getNumChannels(); ++i)
	{
		delete[] buffers[i];
		buffers[i] = nullptr;
	}
	delete[] aggregation_buffer;
	aggregation_buffer = nullptr;
}