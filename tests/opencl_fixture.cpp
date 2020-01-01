#include "opencl_fixture.h"


void OpenCLTest::SetUp()
{
	ctx = OpenCLContext::getInstance();
}