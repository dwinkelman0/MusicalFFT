#ifndef _NOTE_PROFILE_H_
#define _NOTE_PROFILE_H_

#include <stdint.h>
#include <string>


class NoteProfile
{
public:
	NoteProfile(const int32_t base_note_id);

	~NoteProfile();

	void fromWav(const std::string& fname, const float a4_freq, const size_t n_samples_per_chunk);

	uint64_t getSamplesPerSecond() const
	{
		return n_samples_per_second;
	}

	size_t getSamplesPerChunk() const
	{
		return n_samples_per_chunk;
	}

	size_t getNotesPerChunk() const
	{
		return n_notes_per_chunk;
	}

	uint64_t getTimestampByIndex(const size_t index) const
	{
		if (!timestamps || index > n_chunks) return 0;
		else return timestamps[index];
	}

	const float* getNotesByIndex(const size_t index) const
	{
		if (!notes || index > n_chunks) return nullptr;
		else return notes + index * n_notes_per_chunk;
	}


protected:
	uint64_t* timestamps;
	uint64_t n_samples_per_second;
	float* notes;
	size_t n_notes_per_chunk;
	size_t n_chunks;
	size_t n_samples_per_chunk;
	int32_t base_note_id;
};





#endif