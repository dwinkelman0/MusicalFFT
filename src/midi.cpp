#include "midi.h"

#include <assert.h>
#include <iostream>


MidiFile::MidiFile(const std::string& fname) :
	ist(fname)
{
	// Parse the header
	if (read32be() != 0x4d546864 || read32be() != 6)
	{
		throw std::runtime_error("Invalid file format (MIDI header)");
	}

	// Discard the next 6 bytes
	read16be();
	read16be();
	read16be();

	// There should be exactly one track
	if (read32be() != 0x4d54726b)
	{
		throw std::runtime_error("Invalid file format (MIDI track)");
	}

	data_bytes_remaining = read32be();

	int c = 0;
	while (data_bytes_remaining > 0)
	{
		uint32_t time = readVarInt();
		uint8_t type = read8() >> 4;
		if (type == 8 || type == 9)
		{
			uint8_t note = read8();
			uint8_t velocity = read8();
			assert(note < 128);
			assert(velocity < 128);
		}
		else
		{
			uint8_t byte = read8();
		}
	}
}


MidiFile::~MidiFile()
{
	ist.close();
}


uint8_t MidiFile::read8()
{
	uint8_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 1);
	data_bytes_remaining -= 1;
	return output;
}


uint16_t MidiFile::read16be()
{
	uint16_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 2);
	data_bytes_remaining -= 2;
	return be16toh(output);
}


uint32_t MidiFile::read32be()
{
	uint32_t output = 0;
	ist.read(reinterpret_cast<char*>(&output), 4);
	data_bytes_remaining -= 4;
	return be32toh(output);
}


uint32_t MidiFile::readVarInt()
{
	uint32_t output = 0;
	for (size_t i = 0; i < 4; ++i)
	{
		output <<= 7;
		uint8_t next_byte;
		ist.read(reinterpret_cast<char*>(&next_byte), 1);
		data_bytes_remaining -= 1;
		output += next_byte & 0x7f;
		if (!(next_byte & 0x80)) break;
	}
	return output;
}