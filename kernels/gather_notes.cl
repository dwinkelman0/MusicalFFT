#define FFT_SIZE (1 << N_STAGES)


__kernel void gather_notes(__read_only __global float* fft_output, __write_only __global float* notes_output)
{
	// Each workitem is responsible for a chunk
	unsigned int j = get_global_id(0);

	unsigned int input_offset = 6 * FFT_SIZE * j;
	unsigned int output_offset = 12 * N_STAGES * j;

	for (unsigned int note_id = 0; note_id < 12; ++note_id)
	{
		for (unsigned int octave = 0; octave < N_STAGES; ++octave)
		{
			notes_output[output_offset + note_id + 12 * octave] = fft_output[input_offset + note_id * FFT_SIZE / 2 + (1 << octave)];
		}
	}
}