#include "opencl_fixture.h"


void OpenCLTest::SetUp()
{
	ctx = new OpenCLContext();
}


void OpenCLTest::TearDown()
{
	delete ctx;
	ctx = nullptr;
}