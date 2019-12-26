#define N_STAGES 10
#define FFT_SIZE (1 << N_STAGES)


__kernel void musical_cepstrum(__read_only __global float* fft_output, unsigned int n_octaves, __write_only __global float* cepstrum_output)
{
	// Each workitem is responsible for one note
	unsigned int j = get_global_id(0);
}