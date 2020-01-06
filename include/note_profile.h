#ifndef _NOTE_PROFILE_H_
#define _NOTE_PROFILE_H_

#include <stdint.h>
#include <string>


class NoteProfile
{
public:
	NoteProfile(const size_t n_samples_per_chunk, const int32_t base_note_id, const float a4_freq);

	~NoteProfile();

	void fromWav(const std::string& fname);


protected:
	uint64_t* timestamps;
	float* notes;
	size_t n_samples_per_chunk;
	size_t n_notes_per_chunk;
	size_t n_chunks;
	int32_t base_note_id;
	float a4_freq;
};





#endif