#ifndef _OPENCL_H_
#define _OPENCL_H_

#include <CL/cl.h>

#include <vector>


class OpenCLContext;


/*! Validate OpenCL errors after each operation; throws if fatal */
static void checkError(const cl_int err, const char* message);


class OpenCLQueue
{
public:
	OpenCLQueue(const cl_device_id device_id, cl_context ctx);

	~OpenCLQueue();

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

protected:
	cl_context ctx;
	std::vector<OpenCLQueue> queues;
};


#endif