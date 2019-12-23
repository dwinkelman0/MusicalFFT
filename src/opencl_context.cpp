#include "opencl_context.h"

#include <boost/filesystem.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string.h>


void checkError(cl_int err, const char * message) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error " << err << ": " << message << std::endl;
        throw std::runtime_error(message);
    }
}


OpenCLDevice::OpenCLDevice(const cl_device_id device_id, cl_context ctx) :
    device(device_id),
    ctx(ctx),
    cmdq(nullptr)
{
    cl_int err = 0;
    cmdq = clCreateCommandQueueWithProperties(ctx, device_id, nullptr, &err);
    checkError(err, "clCreateCommandQueueWithProperties");
}


OpenCLDevice::~OpenCLDevice()
{
    // TODO: address release
}


OpenCLContext::OpenCLContext() :
    ctx(nullptr),
    queues(),
    n_devices(0),
    device_ids(nullptr)
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
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices);
    checkError(err, "clGetDeviceIDs");
    if (n_devices == 0)
    {
        throw std::runtime_error("There are no OpenCL GPU devices");
    }

    device_ids = new cl_device_id[n_devices];
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
        queues.push_back(OpenCLDevice(device_ids[i], ctx));
    }
}


OpenCLContext::~OpenCLContext()
{
    // TODO: address release
}


cl_kernel OpenCLContext::createKernel(const std::string& kernel_name, const std::string& file_path)
{
    boost::filesystem::path src_path(file_path);
    bool src_exists = boost::filesystem::exists(src_path);

    boost::filesystem::path cached_path(".kernel_cache/" + kernel_name + ".bin");
    bool cached_exists = boost::filesystem::exists(cached_path);

    if (src_exists)
    {
        if (cached_exists)
        {
            // Choose the most recent
            std::time_t src_time = boost::filesystem::last_write_time(src_path);
            std::time_t cached_time = boost::filesystem::last_write_time(cached_path);

            if (difftime(src_time, cached_time) > 0)
            {
                // The source is more recent, recompile
                return compileKernelFromSource(kernel_name, file_path);
            }
            else
            {
                // The cache is most recent, safe to use
                return loadKernelFromBinary(kernel_name, cached_path.string());
            }
        }
        else
        {
            return compileKernelFromSource(kernel_name, file_path);
        }
    }
    else
    {
        if (cached_exists)
        {
            // Go with the cached version
            return loadKernelFromBinary(kernel_name, cached_path.string());
        }
        else
        {
            // Neither option is available
            throw std::runtime_error("Neither the original source file '" + file_path + "' nor the cached binary for kernel '" + kernel_name + "' exist");
        }
    }
}


cl_kernel OpenCLContext::compileKernelFromSource(const std::string& kernel_name, const std::string& file_path)
{
    cl_int err = 0;
    
    // Read file
    std::ifstream ist(file_path);
    std::string src((std::istreambuf_iterator<char>(ist)), std::istreambuf_iterator<char>());
    
    // Create program
    const char* srcs[1] = { src.c_str() };
    const size_t lengths[1] = { src.length() };
    cl_program program = clCreateProgramWithSource(ctx, 1, srcs, lengths, &err);
    checkError(err, "clCreateProgramWithSource");
    
    // Compile program
    const char* options = "";
    err = clBuildProgram(program, n_devices, device_ids, options, NULL, NULL);
    checkError(err, "clBuildProgram");
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    checkError(err, "clCreateKernel");

    return kernel;
}


cl_kernel OpenCLContext::loadKernelFromBinary(const std::string& kernel_name, const std::string& file_path)
{
    cl_int err = 0;

    // Read file
    std::ifstream ist(file_path);
    std::string binary((std::istreambuf_iterator<char>(ist)), std::istreambuf_iterator<char>());

    // Create the program
    const unsigned char* binaries[1] = { reinterpret_cast<const unsigned char*>(binary.c_str()) };
    const size_t lengths[1] = { binary.length() };
    cl_program program = clCreateProgramWithBinary(ctx, n_devices, device_ids, lengths, binaries, nullptr, &err);
    checkError(err, "clCreateProgramWithBinary");

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    checkError(err, "clCreateKernel");

    return kernel;
}