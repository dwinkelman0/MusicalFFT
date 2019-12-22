#ifndef _TEST_OPENCL_FIXTURE_H_
#define _TEST_OPENCL_FIXTURE_H_

#include <opencl.h>

#include <gtest/gtest.h>


/*! Test fixture for everything FFT */
class OpenCLTest : public ::testing::Test
{
protected:
	void SetUp() override;
	void TearDown() override;

protected:
	OpenCLContext* ctx;
};


#endif