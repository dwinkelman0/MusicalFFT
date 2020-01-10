#ifndef _MIDI_H_
#define _MIDI_H_

#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>


class MidiFile
{
public:
	MidiFile(const std::string& fname);

	~MidiFile();

protected:
	uint8_t read8();
	uint16_t read16be();
	uint32_t read32be();
	uint32_t readVarInt();

protected:
	std::ifstream ist;
	int64_t data_bytes_remaining;
};


#endif