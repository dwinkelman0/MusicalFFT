#define FFT_SIZE 1024
#define N_STAGES 10


/*! Compute a complex root of unity */
float2 cexp(float angle)
{
	float2 output = (float2)(cos(angle), sin(angle));
	return output;
}


/*! Multiply two complex numbers */
float2 cmult(float2 c1, float2 c2)
{
	float2 output;
	output.s0 = c1.s0 * c2.s0 - c1.s1 * c2.s1;
	output.s1 = c1.s0 * c2.s1 + c1.s1 * c2.s0;
	return output;
}


/*! Perform an FFT on a musical signal
 *
 *  The signal is a series of N samples in the time domain; the sampling
 *  frequency of this signal is provided; the signal is analyzed in periods
 *  corresponding to chunk size
 *
 *  Each workgroup in the kernel is responsible for analyzing a particular
 *  chunk at a particular frequency; there are 12 frequencies, one representing
 *  each musical note, that are spaced according to an exponential scale
 *  starting at base_note_freq
 */
__kernel void musical_fft(__global float* signal, float samples_per_chunk, float samples_per_base_note, __local float* signal_chunk, __global float* output)
{
	// Determine which portion of the signal to use
	unsigned int chunk_id = get_group_id(0);
	float begin = chunk_id * samples_per_chunk;
	unsigned int begin_index = (unsigned int)floor(begin);

	// Determine which portion of the output to use
	event_t output_copy;
	unsigned int big_output_offset = chunk_id * FFT_SIZE * 6;

	// Get index of workitem
	unsigned int j = get_local_id(0);

	// Local memory for performing the FFT
	event_t chunk_copy;
	__local float2 fft_mem[FFT_SIZE];

	// Local memory for storing the output of the FFT
	__local float fft_output[FFT_SIZE / 2];

	// Cache the relevant portion of the signal into local memory
	chunk_copy = async_work_group_copy(signal_chunk, signal + (unsigned int)floor(begin), (unsigned int)floor(samples_per_base_note + 2), 0);
	wait_group_events(1, &chunk_copy);

	for (unsigned int note_id = 0; note_id < 12; ++note_id)
	{
		for (unsigned int i = 0; i < 2; ++i)
		{
			float samples_per_fft_slot = (samples_per_base_note / pow(2, (float)note_id / 12)) / FFT_SIZE;
			float rel_pos = begin + (2 * j + i) * samples_per_fft_slot - begin_index;
			float weight_hi = rel_pos - floor(rel_pos);
			float weight_lo = 1 - weight_hi;
			fft_mem[j * 2 + i] = (float2)(weight_lo * signal_chunk[(unsigned int)floor(rel_pos)] + weight_hi * signal_chunk[(unsigned int)ceil(rel_pos)], 0);
		}

		// Synchronize before performing FFT
		work_group_barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the FFT algorithm
		for (unsigned int stage = 0; stage < N_STAGES; ++stage)
		{
			unsigned int n_universes = FFT_SIZE >> (stage + 1);
			unsigned int n_pairs = 1 << stage;
			unsigned int exp_spacing = N_STAGES - (stage + 1);

			unsigned int k = j % n_pairs;
			unsigned int u = j / n_pairs;

			float2 even = fft_mem[(u << stage) + k];
			float2 odd = cmult(fft_mem[((u + n_universes) << stage) + k], cexp(k * 2*M_PI_F / (1 << (stage + 1))));

			work_group_barrier(CLK_LOCAL_MEM_FENCE);

			fft_mem[(u << (stage + 1)) + k] = even + odd;
			fft_mem[(u << (stage + 1)) + k + n_pairs] = even - odd;

			//work_group_barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Transfer results to output buffer and synchronize before copying
		fft_output[j] = length(fft_mem[j]);
		work_group_barrier(CLK_LOCAL_MEM_FENCE);


		if (note_id != 0)
		{
			wait_group_events(1, &output_copy);
		}
		unsigned int small_output_offset = note_id * FFT_SIZE / 2;
		output_copy = async_work_group_copy(output + big_output_offset + small_output_offset, fft_output, FFT_SIZE / 2, 0);
	}
}