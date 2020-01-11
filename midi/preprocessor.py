import mido
import os
import sys


def createVarInt(x):
	output = []
	for i in range(4):
		byte = x & 0x7f
		if i != 0: byte |= 128
		output.append(byte)
		x = x >> 7;
		if x == 0:
			break
	return bytes(reversed(output))


def process(fname):

	# Open file and scan tracks for tempo and time signature messages
	# Eventually use tempo and time signature to refine estimation
	mid = mido.MidiFile(fname)
	meta_msgs = [msg for track in mid.tracks for msg in track if type(msg) == mido.MetaMessage]
	tempo_msgs = [msg for msg in meta_msgs if msg.type == "set_tempo"]
	timesig_msgs = [msg for msg in meta_msgs if msg.type == "time_signature"]

	# Get absolute times
	for track in mid.tracks:
		for msg, prev in zip(track[1:], track[:-1]):
			msg.time += prev.time

	# Merge the useful signals of all tracks into a single track
	output_track = []
	for track_index, track in enumerate(mid.tracks):

		# Convert "note_on" with 0 velocity to "note_off"
		for i, msg in enumerate(track):
			if msg.type == "note_on" and msg.velocity == 0 or msg.type == "note_off":
				track[i] = mido.Message(type="note_off", velocity=0, note=msg.note, time=msg.time, channel=track_index)
			elif msg.type == "note_on":
				track[i] = mido.Message(type="note_on", velocity=msg.velocity, note=msg.note, time=msg.time, channel=track_index)


		# Isolate relevant messages and append to output
		output_track += [msg for msg in track if type(msg) == mido.Message and (msg.type == "note_on" or msg.type == "note_off")]

	# Sort output messages by absolute time
	output_track.sort(key=lambda msg: msg.time * 128 + msg.note)

	# Adjust time on each message to relative time
	for msg, prev in zip(reversed(output_track[1:]), reversed(output_track[:-1])):
		msg.time = msg.time - prev.time

	with open("{}-reduced.mid".format(fname[:-4]), "wb") as output_file:
		output_file.write(bytes([0x4d ,0x54, 0x68, 0x64]))
		output_file.write(bytes([0, 0, 0, 6, 0, 0, 0, 0, mid.ticks_per_beat >> 8, mid.ticks_per_beat & 0xff]))
		output_file.write(bytes([0x4d, 0x54, 0x72, 0x6b]))

		data_bytes = bytes()
		for msg in output_track:
			data_bytes += createVarInt(msg.time)
			data_bytes += bytes([0x90 if msg.type == "note_on" else 0x80, msg.note, msg.velocity])

		x = len(data_bytes) + 4
		output_file.write(bytes([x >> 24, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]))
		output_file.write(data_bytes)
		output_file.write(bytes([0, 0xff, 0x2f, 0]))

	return True


if __name__ == "__main__":

	# Gather files to pre-process
	fnames = ["../data/{}".format(fname) for fname in sys.argv[1:]]

	# Process each file
	for fname in fnames:
		if not os.path.exists(fname):
			print("File {} does not exist".format(fname))
			continue

		process(fname)