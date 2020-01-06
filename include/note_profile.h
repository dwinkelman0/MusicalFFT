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


public:
	uint64_t* timestamps;
	uint64_t timestamp_freq;
	float* notes;
	size_t n_notes_per_chunk;
	size_t n_chunks;
	int32_t base_note_id;
};





#endif