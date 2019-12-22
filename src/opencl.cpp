#include "opencl.h"

#include <iostream>
#include <stdexcept>
#include <string.h>


void checkError(cl_int err, const char * message) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error " << err << ": " << message << std::endl;
        throw std::runtime_error(message);
    }
}


OpenCLQueue::OpenCLQueue(const cl_device_id device_id, cl_context ctx) :
    device(device_id),
    ctx(ctx),
    cmdq(nullptr)
{
    cl_int err = 0;
    cmdq = clCreateCommandQueueWithProperties(ctx, device_id, nullptr, &err);
    checkError(err, "clCreateCommandQueueWithProperties");
}


OpenCLQueue::~OpenCLQueue()
{
    // Todo: address release
}


OpenCLContext::OpenCLContext() :
    ctx(nullptr),
    queues()
{
    cl_int err = 0;


    // Get a list of OpenCL platforms and choose one
    cl_uint n_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &n_platforms);
    checkError(err, "clGetPlatformIDs");
    if (n_platforms == 0)
    {
        throw std::runtime_error("There are no OpenCL platforms");
    }

    cl_platform_id* platform_ids = new cl_platform_id[n_platforms];
    memset(platform_ids, 0, n_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(n_platforms, platform_ids, nullptr);
    checkError(err, "clGetPlatformIDs");

    cl_platform_id platform = platform_ids[0];
    delete[] platform_ids;
    platform_ids = nullptr;


    // Get a list of GPU devices
    cl_uint n_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices);
    checkError(err, "clGetDeviceIDs");
    if (n_devices == 0)
    {
        throw std::runtime_error("There are no OpenCL GPU devices");
    }

    cl_device_id* device_ids = new cl_device_id[n_devices];
    memset(device_ids, 0, n_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n_devices, device_ids, nullptr);
    checkError(err, "clGetDeviceIDs");


    // Create an OpenCL context
    const cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0, 0
    };
    ctx = clCreateContext(ctx_props, n_devices, device_ids, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");


    // Create a queue for each device
    for (cl_uint i = 0; i < n_devices; ++i)
    {
        queues.push_back(OpenCLQueue(device_ids[i], ctx));
    }


    // Clean up
    delete[] device_ids;
    device_ids = nullptr;
}


OpenCLContext::~OpenCLContext()
{
    // TODO: address release
}