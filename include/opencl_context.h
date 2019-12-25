#ifndef _OPENCL_H_
#define _OPENCL_H_

#include <CL/cl.h>

#include <string>
#include <vector>


class OpenCLMemory;
class OpenCLDevice;
class OpenCLContext;


/*! Validate OpenCL errors after each operation; throws if fatal */
void checkError(const cl_int err, const char* message);


class OpenCLMemory
{
public:
	OpenCLMemory(const size_t size, OpenCLDevice* device);

	~OpenCLMemory();

	void write(const uint8_t* data, const size_t n_data);

	void read(const size_t n_buffer, uint8_t* buffer, size_t* n_read);

	cl_mem getHandle() const
	{
		return mem_handle;
	}

protected:
	cl_mem mem_handle;
	size_t size;
	OpenCLDevice* device;
};


class OpenCLDevice
{
	friend class OpenCLMemory;

public:
	OpenCLDevice(const cl_device_id device_id, cl_context ctx);

	~OpenCLDevice();

	OpenCLMemory* newMemory(const size_t size);

	uint32_t getLocalMemorySize();
	uint32_t getMaxWorkGroupSize();
	uint32_t getMaxComputeUnits();

	cl_command_queue getCommandQueue() const
	{
		return cmdq;
	}

protected:
	cl_context ctx;
	cl_device_id device;
	cl_command_queue cmdq;
};


class OpenCLContext
{
	friend class OpenCLDevice;

public:
	/*! Create an OpenCL context with a GPU */
	OpenCLContext();

	/*! Destructor */
	~OpenCLContext();

	cl_kernel createKernel(const std::string& kernel_name, const std::string& file_name);

	std::vector<OpenCLDevice*> getDevices() const
	{
		return devices;
	}


protected:
	cl_kernel compileKernelFromSource(const std::string& kernel_name, const std::string& file_path);

	cl_kernel loadKernelFromBinary(const std::string& kernel_name, const std::string& file_path);

protected:
	cl_context ctx;
	std::vector<OpenCLDevice*> devices;
	cl_uint n_devices;
	cl_device_id* device_ids;
};


#endif