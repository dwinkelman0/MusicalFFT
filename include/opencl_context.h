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


class OpenCLDevice
{
	friend class OpenCLMemory;

public:
	OpenCLDevice(const cl_device_id device_id, cl_context ctx);

	~OpenCLDevice();

	uint32_t getLocalMemorySize();
	uint32_t getMaxWorkGroupSize();
	uint32_t getMaxComputeUnits();

	cl_command_queue getCommandQueue() const
	{
		return cmdq;
	}

	cl_context getContext() const
	{
		return ctx;
	}

protected:
	cl_context ctx;
	cl_device_id device;
	cl_command_queue cmdq;
};


class OpenCLContext
{
	friend class OpenCLDevice;

protected:
	/*! Create an OpenCL context with a GPU */
	OpenCLContext();

	/*! Destructor */
	~OpenCLContext();

public:
	static OpenCLContext* getInstance()
	{
		static OpenCLContext instance;
		return &instance;
	}

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