// header files

#include <stdio.h>
#include <stdlib.h>

// OpenCl Header
#include "opencl.h"

#define CL_TARGET_OPENCL_VERSION 300

// global variables
const int iNumberOfArrayElements = 5;

cl_platform_id 		oclPlatformID;
cl_device_id 		oclDeviceID;

cl_context 			oclContext;
cl_command_queue 	oclCommandQueue;

cl_program 			oclProgram;
cl_kernel 			oclKernel;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

// OpenCl Kernel
const char *oclSourceCode = 
"__kernel void vecAddGPU(__global float *in1, __global float *in2, __global float *out, int len)" \

"{" \

	"int i = get_global_id(0);" \
	
	"if(i < len)" \
	
	"{" \

		"out[i] = in1[i] + in2[i];" \
	
	"}" \

"}";

// entry point function
int main(void)
{
	// function declarations
	void cleanup(void);
	
	// variable declarations
	int size = iNumberOfArrayElements *sizeof(float);
	cl_int result;
	
	// code
	// host memory allocation
	
	hostInput1 = (float*)malloc(size);
	if(hostInput1 == NULL)
	{
		printf("Host Memory Allocation is failed for HostInput1 array \n");
		
		cleanup();
	
		exit(EXIT_FAILURE);	// Generic exit to suppport all platforms and devices
	}
	
	hostInput2 = (float*)malloc(size);
	if(hostInput2 == NULL)
	{
		printf("Host Memory Allocation is failed for HostInput2 array \n");
		
		cleanup();
	
		exit(EXIT_FAILURE);	// Generic exit to suppport all platforms and devices
	}
	
	
	hostOutput = (float*)malloc(size);
	if(hostOutput == NULL)
	{
		printf("Host Memory Allocation is failed for HostOutput array \n");
		
		cleanup();
	
		exit(EXIT_FAILURE);	// Generic exit to suppport all platforms and devices
	}

	// Filling values into host array
	hostInput1[0] = 101.0;
	hostInput1[1] = 102.0;
	hostInput1[2] = 103.0;
	hostInput1[3] = 104.0;
	hostInput1[4] = 105.0;
	
	hostInput2[0] = 201.0;
	hostInput2[1] = 202.0;
	hostInput2[2] = 203.0;
	hostInput2[3] = 204.0;
	hostInput2[4] = 205.0;
	
	
	// get OpenCl Supporting Platform Ids
	result = clGetPlatformIDs(1,&oclPlatformID, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clGetPlatfromIDs() Failed 				: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	// get OpenCl Supporting CPU Device IDs
	result = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclDeviceID, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clGetDeviceIDs() Failed 				: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	// Create OpenCl Compute Context
	oclContext = clCreateContext(NULL, 1, &oclDeviceID, NULL, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateContext() Failed 				: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	// Create Command Queue
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceID, 0, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateCommandQueue() Failed 			: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	
	// Create OpenCl Program from .cl
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&oclSourceCode, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateProgramWithSource() Failed 		: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	// build OpenCl Program
	// Interview Question - Why 0 and NULL , 2nd and 3rd Parameter
	result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		
		result = clGetProgramBuildInfo(oclProgram, oclDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		sizeof((buffer), buffer, &len);
		
		printf("Program Build Log 		 		: %s \n", buffer);
		printf("clBuildProgram() Failed 		: %d \n", result);
		
		cleanup();
		
		exit(EXIT_FAILURE);
	}
	
	
	// create OpenCL Kernel by passing kernel function name that we used in .cl file
	oclKernel = clCreateKernel(oclProgram, "vecAddGPU", &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateKernel() Failed			:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	
	// device Memory Allocation
	deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For 1st Input Array 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For 2nd Input Array 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	deviceOutput = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For Output Array	 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 0th Argument i.e. deviceInput1
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceInput1);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 1st Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 1st Argument i.e. deviceInput2
	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 2nd Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 2nd Argument i.e. deviceOutput
	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 3rd Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	// set 0 based 3rd Argument i.e. len
	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&iNumberOfArrayElements);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 4th Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	// write above input device buffer to device memory
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput1, CL_FALSE, 0, size, hostInput1, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() Failed for 1st Input Device Buffer 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);	
	
	}
	
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() Failed for 2nd Input Device Buffer 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);	
	
	}
	
	// Kernel Configuration
	size_t global_size = 5;
	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueNDRangeKernel() Failed 		: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	// Finish OpenCl command Queue
	clFinish(oclCommandQueue);
	
	// read back result from device (deviceOutput) into cpu variable (hostOutput)
	result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueuReadBuffer() Failed 			:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	// display Result
	int i;
	for(i = 0; i < iNumberOfArrayElements; i++)
	{
		printf("\n");
		printf("\x1b[36m %f + %f = %f \x1b[0m \n",hostInput1[i], hostInput2[i], hostOutput[i]);
		printf("\n");
	}
	
	printf("\x1b[0m ************************************************************ \n");
		
	printf("\x1b[6m \x1b[32m Developed by - Yashraj Gaikwad 		\n");
	printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
	printf("\x1b[0m \n");
	
	
	// cleanup
	cleanup();
	
	
	
	return(0);
	
	
}

void cleanup(void)
{
	// code
	if(deviceOutput)
	{
		clReleaseMemObject(deviceOutput);
		deviceOutput = NULL;
		
	}
	
	if(deviceInput2)
	{
		clReleaseMemObject(deviceInput2);
		deviceInput2 = NULL;
		
	}
	
	if(deviceInput1)
	{
		clReleaseMemObject(deviceInput1);
		deviceInput1 = NULL;
		
	}
	
	// Release Kernel
	if(oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}
	
	if(oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}
	
	if(oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}
	
	if(oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
	}
	
	if(hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
		
	}
	
	if(hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
		
	}
	
	if(hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
		
	}
	
	
	
	
	
}















