#define FFT_SIZE 1024


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
__kernel void musical_fft(float data_freq, unsigned int n_data, __global float* data, float base_note_freq, __global float* output)
{
	// Each workgroup is responsible for a given chunk
	unsigned int chunk_id = get_group_id(0);
	unsigned int chunk_size = n_data / get_num_groups(0);
	unsigned int data_offset = chunk_size * chunk_id;
	unsigned int output_offset = chunk_id * FFT_SIZE * 6;

	// Each workitem is responsible for one pair in each stage
	unsigned int j = get_local_id(0);

	// Allocate local memory for each workgroup and copy chunk
	__local float sig_mem[FFT_SIZE + 1];
	event_t initial_copy = async_work_group_copy(sig_mem, data + data_offset, chunk_size + 1, 0);
	wait_group_events(1, &initial_copy);

	// Allocate local memory for computing an FFT
	__local float2 fft_mem[FFT_SIZE];

	// Allocate local memory for buffering the results
	__local float temp_output_mem[FFT_SIZE / 2];
	event_t temp_output_mem_copy;

	// Iterate through each note
	for (unsigned int note_id = 0; note_id < 12; ++note_id)
	{
		// Determine how many samples are in a note's cycle
		float note_freq = base_note_freq * pow(2, note_id / 12.0f);
		float samples_per_cycle = data_freq / note_freq;
		float samples_per_fft_slot = samples_per_cycle / FFT_SIZE;

		// Interpolate from the original signal into the FFT memory
		for (unsigned int i = 0; i < 2; ++i)
		{
			float data_index = (j * 2 + i + 0.5) * samples_per_fft_slot;
			float weight_lo = ceil(data_index) - data_index;
			float weight_hi = 1 - weight_lo;
			fft_mem[j * 2 + i] = (float2)(weight_lo * sig_mem[(unsigned int)floor(data_index)] + weight_hi * sig_mem[(unsigned int)ceil(data_index)], 0);
		}

		work_group_barrier(CLK_LOCAL_MEM_FENCE);

		// Compute stages
		for (unsigned int stage = 0; stage < 10; ++stage)
		{
			unsigned int n_universes = FFT_SIZE >> (stage + 1);
	        unsigned int n_pairs = 1 << stage;
	        unsigned int exp_spacing = 10 - (stage + 1);

	        unsigned int k = j % n_pairs;
	        unsigned int u = j - k;

            float2 even = fft_mem[(u << stage) + k];
            float2 odd = cmult(fft_mem[((u + n_universes) << stage) + k], cexp((k << exp_spacing) * 2 * M_PI_F / FFT_SIZE));

            // After all values gathered, synchronize and write
            work_group_barrier(CLK_LOCAL_MEM_FENCE);

            fft_mem[(u << (stage + 1)) + k] = even + odd;
            fft_mem[(u << (stage + 1)) + k + n_pairs] = even - odd;

            work_group_barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Make sure output buffer is clear
		if (note_id > 0)
		{
			wait_group_events(1, &temp_output_mem_copy);
		}

		// Get magnitude of each complex number and store
		temp_output_mem[j] = length(fft_mem[j]);

		// Copy to global memory
		temp_output_mem_copy = async_work_group_copy(output + output_offset + note_id * FFT_SIZE, temp_output_mem, FFT_SIZE / 2, 0);
	}
}