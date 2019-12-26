#define N_STAGES 10
#define FFT_SIZE (1 << N_STAGES)


/*! Compute a complex root of unity; the output are the rectangular coordinates
 *  of the polar coordinates with the given angle and radius of 1
 *
 *    @param angle: angle of the polar form of the exponential
 */
float2 cexp(float angle)
{
	float2 output = (float2)(cos(angle), sin(angle));
	return output;
}


/*! Multiply two complex numbers
 *
 *    @param c1: first complex number
 *    @param c2: second complex number
 */
float2 cmult(float2 c1, float2 c2)
{
	float2 output;
	output.s0 = c1.s0 * c2.s0 - c1.s1 * c2.s1;
	output.s1 = c1.s0 * c2.s1 + c1.s1 * c2.s0;
	return output;
}


/*! Perform a specialized sequence of FFT's on a musical signal
 *
 *  The signal is a series of N samples in the time domain; each workgroup
 *  analyzes chunks of data at a certain rate, so each chunk will contain a
 *  fixed number of samples
 *
 *  For each chunk, an FFT with the base frequency of each of the 12 chromatic
 *  notes is calculated; before any of these FFT's are calculated, the signal
 *  data needed to analyze the longest wavelength in the chunk is copied to
 *  local memory
 *
 *  For each note, one complete wavelength is interpolated into the local
 *  analysis buffer (each analysis buffer is the same length for any note, and
 *  must be a power of 2); the FFT is then performed
 *
 *  Once the FFT completes, each complex number is transformed into a power
 *  quantity measured in decibels in local memory; the memory is asynchronously
 *  copied to global output memory
 *
 *    @param signal: musical signal to analyze
 *    @param samples_per_chunk: number of samples per chunk
 *    @param samples_per_base_note: number of samples for the note of the
 *                                  longest wavelength
 *    @param signal_chunk: local memory of variable length for buffering chunks
 *    @param output: memory for the final result organized as a 3D array with
 *                   the following axes: (chunk, note, overtone)
 */
__kernel void musical_fft(
		__read_only __global float* signal, float samples_per_chunk, float samples_per_base_note,
		__local float* signal_chunk, __write_only __global float* fft_output, __write_only __global float* cepstrum_output)
{
	// Determine which portion of the signal to use
	unsigned int chunk_id = get_group_id(0);
	float begin = chunk_id * samples_per_chunk;
	unsigned int begin_index = (unsigned int)floor(begin);

	// Determine which portion of the output to use
	unsigned int big_output_offset = chunk_id * FFT_SIZE * 6;

	// Get index of workitem
	unsigned int j = get_local_id(0);

	// Local memory for performing the FFT
	event_t chunk_copy;
	__local float2 fft_mem[FFT_SIZE];

	// Local memory for storing the output of the FFT
	event_t output_copy;
	__local float local_output[FFT_SIZE];

	// Cache the relevant portion of the signal into local memory
	chunk_copy = async_work_group_copy(signal_chunk, signal + (unsigned int)floor(begin), (unsigned int)floor(samples_per_base_note + 2), 0);
	wait_group_events(1, &chunk_copy);

	for (unsigned int note_id = 0; note_id < 12; ++note_id)
	{
		float samples_per_fft_slot = (samples_per_base_note / pow(2, (float)note_id / 12)) / FFT_SIZE;
		for (unsigned int i = 0; i < 2; ++i)
		{
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

			work_group_barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Transfer results to output buffer and synchronize before copying
		// Output is in decibels
		local_output[j] = 20 * log10(length(fft_mem[j]) / FFT_SIZE);
		work_group_barrier(CLK_LOCAL_MEM_FENCE);

		if (note_id != 0)
		{
			wait_group_events(1, &output_copy);
		}
		unsigned int small_output_offset = note_id * FFT_SIZE / 2;
		output_copy = async_work_group_copy(fft_output + big_output_offset + small_output_offset, local_output, FFT_SIZE / 2, 0);

		// Transform complex FFT output into power
		// Changes magnitude but preserves phase shift
		for (unsigned int i = 0; i < 2; ++i)
		{
			float power = 2 * log10(length(fft_mem[2 * j + i]) / FFT_SIZE);
			fft_mem[2 * j + i] = (float2)(power, 0);
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

			work_group_barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Get the real value of the result and place in output buffer
		float swap = fft_mem[j].s0;
		local_output[j] = fft_mem[FFT_SIZE - j - 1].s0;
		local_output[FFT_SIZE - j - 1] = swap;
		work_group_barrier(CLK_LOCAL_MEM_FENCE);

		wait_group_events(1, &output_copy);
		output_copy = async_work_group_copy(cepstrum_output + 2 * (big_output_offset + small_output_offset), local_output, FFT_SIZE, 0);
	}
}