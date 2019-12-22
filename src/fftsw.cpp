#include <fftsw.h>

#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <string.h>


void fft_sw(const float* data, const uint32_t n_samples, complex_t* output)
{
	const uint32_t max_stage = __builtin_ctz(n_samples);

	// Create a lookup table for complex exponentials
    complex_t exp[n_samples / 2];
    for (int i = 0; i < n_samples / 2; ++i)
    {
        complex_t rotation = 1i * i * 2*M_PI / n_samples;
        exp[i] = std::exp(rotation);
    }

    // Create a pair of swapped input/output buffers
    complex_t* inputs = new complex_t[n_samples];
    complex_t* outputs = new complex_t[n_samples];
    for (int i = 0; i < n_samples; ++i)
    {
        inputs[i] = data[i];
    }

    // Within each stage, the input and output arrays can be thought of as
    // matrices; each row corresponds to a universe, and each column to a
    // coefficient for the data contained in that universe
    for (int stage = 0; stage < max_stage; ++stage)
    {
        const uint32_t n_universes = n_samples >> (stage + 1);
        const uint32_t n_pairs = 1 << stage;
        const uint32_t exp_spacing = max_stage - (stage + 1);

        for (int u = 0; u < n_universes; ++u)
        {
            complex_t* evens = inputs + (u << stage);
            complex_t* odds = inputs + ((u + n_universes) << stage);
            complex_t* results = outputs + (u << (stage + 1));

            for (int k = 0; k < n_pairs; ++k)
            {
                complex_t even = evens[k];
                complex_t odd = odds[k] * exp[k << exp_spacing];
                results[k] = even + odd;
                results[k + n_pairs] = even - odd;
            }
        }

        std::swap(inputs, outputs);
    }

    // The output (due to the swap at the end of the loop) is in "inputs"
    memcpy(output, inputs, n_samples / 2 * sizeof(complex_t));

    // Clean up buffers
    delete[] inputs;
    inputs = nullptr;
    delete[] outputs;
    outputs = nullptr;
}


void dft_sw(const float* data, const uint32_t n_samples, complex_t* output)
{
	for (int i = 0; i < n_samples/2; ++i)
    {
        complex_t sum = 0;
        for (int j = 0; j < n_samples; ++j)
        {
            complex_t rotation = 1i * j * i * (2*M_PI/n_samples);
            sum += data[j] * std::exp(rotation);
        }
        output[i] = sum;
    }
}