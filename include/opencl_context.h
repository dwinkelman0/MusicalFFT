#ifndef _OPENCL_H_
#define _OPENCL_H_

#include <CL/cl.h>

#include <string>
#include <vector>


class OpenCLContext;


/*! Validate OpenCL errors after each operation; throws if fatal */
static void checkError(const cl_int err, const char* message);


class OpenCLDevice
{
public:
	OpenCLDevice(const cl_device_id device_id, cl_context ctx);

	~OpenCLDevice();

protected:
	cl_context ctx;
	cl_device_id device;
	cl_command_queue cmdq;
};


class OpenCLContext
{
public:
	/*! Create an OpenCL context with a GPU */
	OpenCLContext();

	/*! Destructor */
	~OpenCLContext();

	cl_kernel createKernel(const std::string& kernel_name, const std::string& file_name);


protected:
	cl_kernel compileKernelFromSource(const std::string& kernel_name, const std::string& file_path);

	cl_kernel loadKernelFromBinary(const std::string& kernel_name, const std::string& file_path);

protected:
	cl_context ctx;
	std::vector<OpenCLDevice> queues;
	cl_uint n_devices;
	cl_device_id* device_ids;
};


#endif