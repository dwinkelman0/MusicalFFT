#ifndef _OPENCL_H_
#define _OPENCL_H_

#include <CL/cl.h>


void clCheckError(cl_int err, const char * message);


int createContext();


#endif