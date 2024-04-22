#define CL_TARGET_OPENCL_VERSION 300
// header files

#include <stdio.h>
#include <stdlib.h>
#include <math.h>	// for fabs()

// OpenCl Header
#include "opencl.h"

#include "helper_timer.h"

// macros
#define BLOCK_WIDTH 512

// global variables
//const int iNumberOfArrayElements = 5;

//const int iNumberOfArrayElements = 11444777;

cl_platform_id 		oclPlatformID;
cl_device_id 		oclDeviceID;

cl_context 			oclContext;
cl_command_queue 	oclCommandQueue;

cl_program 			oclProgram;
cl_kernel			oclKernel;

int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold  = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;


// OpenCl Kernel
const char *oclSourceCode = 
"__kernel void MatMulGPU(__global int *A, __global int *B, __global int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)" \

"{" \

	"int row = get_global_id(0);" \
	"int column = get_global_id(1);" \
	
	"if((row < numARows) && (column < numBColumns))" \
	"{" \
		
		"int value = 0;" \
		
		"for(int k = 0; k < numAColumns; k++)" \
		"{" \
			
			"int a = A[row * numAColumns + k];" \
			"int b = B[k * numBColumns + column];" \
			"value += a*b;" \
		
		"}" \
		
		"C[row * numCColumns + column] = value;" \
	
	"}" \

"}";

// entry point function
int main(void)
{
	// function declarations
	void cleanup(void);
	
	void InitA(int *data, int, int);
	void InitB(int *data, int, int);
	
	void matMulCPU(int*, int*, int*, int, int, int, int);
	
	// variable declarations
	int numARows 	= BLOCK_WIDTH;
	int numAColumns = BLOCK_WIDTH;
	int numBRows 	= BLOCK_WIDTH;
	int numBColumns = BLOCK_WIDTH;
	
	int numCRows 	= numARows;
	int numCColumns = numBColumns;
	
	int numGoldRows 	= numARows;
	int numGoldColumns  = numBColumns;
	
	int sizeA 	 = numARows 	* numAColumns 	 * sizeof(int);
	int sizeB 	 = numBRows 	* numBColumns 	 * sizeof(int);
	int sizeC 	 = numCRows 	* numCColumns 	 * sizeof(int);
	int sizeGold = numGoldRows 	* numGoldColumns * sizeof(int);
	
	
	cl_int result;
	
	// code
	// host memory allocation
	
	hostA = (int *)malloc(sizeA);	// type cast
	
	if(hostA == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostB = (int *)malloc(sizeB);	// type cast
	
	if(hostB == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	
	hostC = (int *)malloc(sizeC);	// type cast
	
	if(hostC == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (int *)malloc(sizeGold);	// type cast
	
	if(gold == NULL)
	{
		printf("Host Memory Allocation is failed for gold Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	// printing matrix dimensions and sizes
	printf("\x1b[33m OpenCL Program To Do Matrix Multiplication \x1b[0m\n");
	
	printf("The Dimensions of Matrix 'hostA' are : %d x %d \n", numARows, numAColumns);
	printf("The Dimensions of Matrix 'hostB' are : %d x %d \n", numBRows, numBColumns);
	printf("The Dimensions of Matrix 'hostC' are : %d x %d \n", numCRows, numCColumns);
	printf("The Dimensions of Matrix 'gold' are  : %d x %d \n", numGoldRows, numGoldColumns);
	
	printf("Size of Matrix hostA = %d\n", sizeA);
	printf("Size of Matrix hostB = %d\n", sizeB);
	printf("Size of Matrix hostC = %d\n", sizeC);
	printf("Size of Matrix Gold  = %d\n", sizeGold);
	
	// fill source matrices
	InitA(hostA, numARows, numAColumns);
	InitB(hostB, numBRows, numBColumns);
	
	
	// get OpenCl Supporting Platform Ids
	result = clGetPlatformIDs(1, &oclPlatformID, NULL);
	
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
	oclKernel = clCreateKernel(oclProgram, "MatMulGPU", &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateKernel() Failed			:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	
	// device Memory Allocation
	deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeA, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For 1st Input Array 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeB, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For 2nd Input Array 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	deviceC = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeC, NULL, &result);
	
	if(result != CL_SUCCESS)
	{
		printf("clCreateBuffer() Failed For Output Array	 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 0th Argument i.e. deviceA
	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceA);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 1st Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 1st Argument i.e. deviceB
	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceB);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 2nd Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 2nd Argument i.e. deviceC
	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceC);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 3rd Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	// set 0 based 3rd Argument i.e. numARows
	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&numARows);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 4th Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 4th Argument i.e. numAColumns
	result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&numAColumns);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 4th Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// set 0 based 5th Argument i.e. numBColumns
	result = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void*)&numBColumns);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 4th Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	// set 0 based 6th Argument i.e. numCColumns
	result = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void*)&numCColumns);
	
	if(result != CL_SUCCESS)
	{
		printf("clKernelArg() Failed For 4th Argument		 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	
	
	// write above input device buffer to device memory
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, sizeA, hostA, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() Failed for 1st Input Device Buffer 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);	
	
	}
	
	result = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, sizeB, hostB, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer() Failed for 2nd Input Device Buffer 	:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);	
	
	}
	
	// Kernel Configuration
	// size_t localWorkSize = 5;
	//size_t localWorkSize = 256;
	
	size_t globalWorkSize[2];
	globalWorkSize[0] = BLOCK_WIDTH;
	globalWorkSize[1] = BLOCK_WIDTH;
	
	
	// start timer
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	
	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueueNDRangeKernel() Failed 		: %d\n", result);
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	// Finish OpenCl command Queue
	clFinish(oclCommandQueue);
	
	// stop timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	
	
	// read back result from device (deviceC) into cpu variable (hostC)
	result = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, sizeC, hostC, 0, NULL, NULL);
	
	if(result != CL_SUCCESS)
	{
		printf("clEnqueuReadBuffer() Failed 			:%d \n", result);
		cleanup();
		exit(EXIT_FAILURE);
		
	}
	
	// Matrix Multiplication on host
	matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns);
	
	// comparison
	//const float epsilon = 0.000001f;
	int breakValue = -1;
	bool bAccurarcy = true;
	
	
	// display Result
	int i;
	for(i = 0; i < numCRows * numCColumns; i++)
	{
		float val1 = gold[i];
		float val2 = hostC[i];
		
		if(val1 != val2)
		{
			bAccurarcy = false;
			breakValue = i;
			
			break;
			
		}
		
		
	}
	
	char str[128];
	if(bAccurarcy == false)
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is Not within accruracy of 0.000001 at array index %d", breakValue);
	else
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is within accruracy of 0.000001");
	
	
	// output
	// 3 beautiful printf()s

	printf("\x1b[36m Time Taken For Matrix Multiplication On CPU = %.6f \n", timeOnCPU);
	
	printf("\x1b[32m Time Taken For Matrix Multiplication On GPU = %.6f \n", timeOnGPU);
	
	printf("\x1b[0m %s \n", str);

	// done
	printf("\x1b[0m ************************************************************ \n");
		
	printf("\x1b[6m \x1b[32m Developed by - Yashraj Gaikwad 		\n");
	printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
	printf("\x1b[0m \n");
	
	
	
	// cleanup
	cleanup();
	
	
	return(0);
	
	
}

void InitA(int *data, int row, int col)
{
	int num = 1;
	
	// code
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num++;
			
		}
		
	}
	
}

void InitB(int *data, int row, int col)
{
	int num = BLOCK_WIDTH;
	
	// code
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num--;
			
		}
		
	}
	
}


void matMulCPU(int* A, int* B, int* C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{
	// code
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	// Romantic Lines
	for(int i = 0; i < numARows; ++i)
	{
		for(int j = 0; j < numBColumns; ++j)
		{
			int value = 0.0f;
			for(int k = 0; k < numARows; ++k)
			{
				int a = A[i * numAColumns + k];
				int b = B[i * numBColumns + j];
				value += a * b;
			}
			
			C[i * numCColumns + j] = value;
		}
		
	}
	
	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	
	timer = NULL;
}



void cleanup(void)
{
	// code
	if(deviceC)
	{
		clReleaseMemObject(deviceC);
		deviceC = NULL;
		
	}
	
	if(deviceB)
	{
		clReleaseMemObject(deviceB);
		deviceB = NULL;
		
	}
	
	if(deviceA)
	{
		clReleaseMemObject(deviceA);
		deviceA = NULL;
		
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
	
	if(gold)
	{
		free(gold);
		gold = NULL;
		
	}
	
	
	if(hostC)
	{
		free(hostC);
		hostC = NULL;
		
	}
	
	if(hostB)
	{
		free(hostB);
		hostB = NULL;
		
	}
	
	if(hostA)
	{
		free(hostA);
		hostA = NULL;
		
	}
	

	
}




















