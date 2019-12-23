/*! Dummy kernel to test the kernel compiling process */
__kernel void vector_add(__global float* A, __global float* B, __global float* C)
{
	int index = get_global_id(0);
	C[index] = A[index] + B[index];
}