#ifndef _OPENCL_MEM_H_
#define _OPENCL_MEM_H_

#include "opencl_context.h"


class OpenCLKernelMemory
{
public:
	OpenCLKernelMemory(OpenCLDevice* device, const size_t size, const cl_mem_flags flags) :
		device(device),
		device_buffer(nullptr),
		size(size),
		flags(flags)
	{}

	~OpenCLKernelMemory()
	{
		if (device_buffer)
		{
			clReleaseMemObject(device_buffer);
			device_buffer = nullptr;
		}
	}

	bool allocateDeviceMemory()
	{
		if (!device_buffer)
		{
			cl_int err = 0;
    		device_buffer = clCreateBuffer(device->getContext(), flags, size, nullptr, &err);
    		checkError(err, "clCreateBuffer");
    		return true;
		}
		return false;
	}

	void setAsKernelArgument(cl_kernel kernel, uint32_t arg_index)
	{
		// Check whether the device buffer has been initialized
		allocateDeviceMemory();

		cl_int err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), (void*)&device_buffer);
		checkError(err, "clSetKernelArg");
	}

	size_t getSize() const
	{
		return size;
	}

protected:
	OpenCLDevice* device;
	cl_mem device_buffer;
	size_t size;
	cl_mem_flags flags;
};


class OpenCLKernelHostMemory : public OpenCLKernelMemory
{
public:
	OpenCLKernelHostMemory(OpenCLDevice* device, const size_t size, const cl_mem_flags flags) :
		OpenCLKernelMemory(device, size, flags),
		host_buffer(nullptr)
	{}

	~OpenCLKernelHostMemory()
	{
		if (host_buffer)
		{
			delete[] host_buffer;
			host_buffer = nullptr;
		}
	}

	bool allocateHostMemory()
	{
		if (!host_buffer)
		{
			host_buffer = new uint8_t[size];
			return true;
		}
		return false;
	}

protected:
	uint8_t* host_buffer;
};


class OpenCLReadOnlyMemory : virtual public OpenCLKernelHostMemory
{
public:
	OpenCLReadOnlyMemory(OpenCLDevice* device, const size_t size, const cl_mem_flags flags) :
		OpenCLKernelHostMemory(device, size, flags | CL_MEM_HOST_READ_ONLY)
	{}

	bool readTo(uint8_t* dst, const size_t n_dst, size_t* n_read)
	{
		// If the device buffer is not yet created, do nothing
		if (!device_buffer) return false;

		// Copy memory from device to host and return the internal buffer
		size_t read_size = n_dst < size ? n_dst : size;
		if (n_read) *n_read = read_size;
		cl_int err = clEnqueueReadBuffer(device->getCommandQueue(), device_buffer, CL_TRUE, 0, read_size, dst, 0, nullptr, nullptr);
		checkError(err, "clEnqueueReadBuffer");
		return true;
	}

	const uint8_t* read(size_t* n_read)
	{
		// Check that host memory is allocated
		allocateHostMemory();
		
		return readTo(host_buffer, size, n_read) ? host_buffer : nullptr;
	}
};


class OpenCLWriteOnlyMemory : virtual public OpenCLKernelHostMemory
{
public:
	OpenCLWriteOnlyMemory(OpenCLDevice* device, const size_t size, const cl_mem_flags flags) :
		OpenCLKernelHostMemory(device, size, flags | CL_MEM_HOST_WRITE_ONLY)
	{}

	uint8_t* getWriteableBuffer()
	{
		// Check that host memory is allocated
		allocateHostMemory();
		return host_buffer;
	}

	bool writeFrom(const uint8_t* src, const size_t n_src, size_t* n_write)
	{
		// If the host buffer is not yet created, do nothing
		if (!src) return false;

		// Copy memory from host to device
		size_t write_size = n_src > size ? n_src : size;
		if (n_write) *n_write = write_size;
		cl_int err = clEnqueueWriteBuffer(device->getCommandQueue(), device_buffer, CL_TRUE, 0, write_size, src, 0, nullptr, nullptr);
		checkError(err, "clEnqueueWriteBuffer");
		return true;
	}

	bool write(size_t* n_write)
	{
		// Check that device memory is allocated
		allocateDeviceMemory();

		return writeFrom(host_buffer, size, n_write);
	}
};


class OpenCLReadWriteMemory : public OpenCLWriteOnlyMemory, public OpenCLReadOnlyMemory
{
public:
	OpenCLReadWriteMemory(OpenCLDevice* device, const size_t size, const cl_mem_flags flags) :
		OpenCLReadOnlyMemory(device, size, flags),
		OpenCLWriteOnlyMemory(device, size, flags),
		OpenCLKernelHostMemory(device, size, flags)
	{}
};


#endif