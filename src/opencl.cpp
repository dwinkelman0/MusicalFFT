#include "opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void clCheckError(cl_int err, const char * message) {
    if (err != CL_SUCCESS) {
        printf("[OpenCL Error %d]: %s\n", err, message);
        exit(-1);
    }
}


int createContext() {

    cl_int err = CL_SUCCESS;
    cl_uint n_platform_ids = 0;
    cl_platform_id * platform_ids = NULL;
    cl_uint n_device_ids = 0;
    cl_device_id * device_ids;
    cl_context_properties ctx_props;
    cl_context ctx;
    cl_command_queue cmdq;
    cl_kernel kernel;

    err = clGetPlatformIDs(0, NULL, &n_platform_ids);
    clCheckError(err, "clGetPlatformIDs");
    printf("%d platforms found\n", n_platform_ids);
    if (n_platform_ids == 0) {
        return 0;
    }
    platform_ids = new cl_platform_id[n_platform_ids];
    memset(platform_ids, 0, n_platform_ids * sizeof(cl_platform_id));
    err = clGetPlatformIDs(n_platform_ids, platform_ids, NULL);
    clCheckError(err, "clGetPlatformIDs");
    
    return 0;
}