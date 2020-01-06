#include "note_profile.h"

#include "ffthw.h"
#include "wav.h"

#include <iostream>


NoteProfile::NoteProfile(const size_t n_samples_per_chunk, const int32_t base_note_id, const float a4_freq) :
	timestamps(nullptr),
	notes(nullptr),
	n_chunks(0),
	n_notes_per_chunk(12 * N_STAGES),
	n_samples_per_chunk(n_samples_per_chunk),
	base_note_id(base_note_id),
	a4_freq(a4_freq)
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


void NoteProfile::fromWav(const std::string& fname)
{
	WavFile file(fname);
	MusicalFFT mfft(OpenCLContext::getInstance());

	// Determine parametrizations of note frequency
	const float base_note_freq = a4_freq * pow(2, (float)(base_note_id - 69) / 12.0);
	const float samples_per_base_note = file.getSampleRate() / base_note_freq;

	// Determine how much memory to allocate for notes
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

	// Keep track of how many chunks have been processed
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
		std::cout << "Samples remaining " << file.getNumSamplesRemaining() << std::endl;
		size_t n_samples_read = file.readSamples(n_samples_to_read, buffers_with_offset);
		if (n_samples_read == 0) break;

		// Perform the FFT and aggregate the data
		size_t n_new_chunks = 0;
		for (size_t i = 0; i < file.getNumChannels(); ++i)
		{
			n_new_chunks = mfft.runFFT(file.getSampleRate(), n_unused_samples + n_samples_read, buffers[i], n_samples_per_chunk, base_note_freq);
			if (!aggregation_buffer)
			{
				aggregation_buffer = new float[n_new_chunks * n_notes_per_chunk];
			}
			const float* notes_output = mfft.readNotes(nullptr, nullptr);
			if (i == 0)
			{
				memcpy(aggregation_buffer, notes_output, n_new_chunks * n_notes_per_chunk * sizeof(float));
			}
			else
			{
				for (size_t index = 0; index < n_new_chunks * n_notes_per_chunk; ++index)
				{
					aggregation_buffer[index] += notes_output[index];
				}
			}
		}
		if (file.getNumChannels() > 1)
		{
			for (size_t index = 0; index < n_new_chunks * n_notes_per_chunk; ++index)
			{
				aggregation_buffer[index] /= 2;
			}
		}

		std::cout << "New Chunks " << n_new_chunks << std::endl;

		// Copy the data into output
		memcpy(notes + chunk_index * n_notes_per_chunk, aggregation_buffer, n_new_chunks * n_notes_per_chunk * sizeof(float));
		chunk_index += n_new_chunks;
	}

	// Clean up
	for (size_t i = 0; i < file.getNumChannels(); ++i)
	{
		delete[] buffers[i];
		buffers[i] = nullptr;
	}
}