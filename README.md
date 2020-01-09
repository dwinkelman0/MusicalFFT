# MusicalFFT

## Environment

 * OpenCL 2.0 (`apt install ocl-icd-opencl-dev`)
 * Google Test (`apt install libgtest-dev`, `cd /usr/src/gtest`, `cmake CMakeLists.txt`, `make`, `cp *.a /usr/lib`)
 * Boost (`apt install libboost-all-dev`)
 * Mido (`pip3 install mido`)

## Features

 * GPU-accelerated Fast Fourier Transform specifically for musical frequencies
 * Extract note profiles from musical FFT's
 * Perform musical FFT on complete WAV files