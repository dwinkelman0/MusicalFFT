import mido
import os
import sys


def process(fname):

	# Open file and scan tracks for tempo and time signature messages
	# Eventually use tempo and time signature to refine estimation
	mid = mido.MidiFile(fname)
	meta_msgs = [msg for track in mid.tracks for msg in track if type(msg) == mido.MetaMessage]
	tempo_msgs = [msg for msg in meta_msgs if msg.type == "set_tempo"]
	timesig_msgs = [msg for msg in meta_msgs if msg.type == "time_signature"]

	"""
	if len(tempo_msgs) == 0:
		tempo_msgs.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
		print("File {} does not have a tempo".format(fname))

	if len(timesig_msgs) == 0:
		print("File {} does not have a time signature".format(fname))
		return False
	"""

	# Get absolute times
	for track in mid.tracks:
		for msg, prev in zip(track[1:], track[:-1]):
			msg.time += prev.time

	# Merge the useful signals of all tracks into a single track
	output_track = mido.MidiTrack()
	for track in mid.tracks:

		# Convert "note_on" with 0 velocity to "note_off"
		for i, msg in enumerate(track):
			if msg.type == "note_on" and msg.velocity == 0:
				track[i] = mido.Message(type="note_off", velocity=0, note=msg.note, time=msg.time)

		# Isolate relevant messages and append to output
		output_track += [msg for msg in track if type(msg) == mido.Message and (msg.type == "note_on" or msg.type == "note_off")]

	# Sort output messages by absolute time
	output_track.sort(key=lambda msg: msg.time * 128 + msg.note)

	# Adjust time on each message to relative time
	for msg, prev in zip(reversed(output_track[1:]), reversed(output_track[:-1])):
		msg.time = msg.time - prev.time

	output_file = mido.MidiFile()
	output_file.tracks = [output_track]
	output_file.save("{}-reduced.mid".format(fname[:-4]))

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