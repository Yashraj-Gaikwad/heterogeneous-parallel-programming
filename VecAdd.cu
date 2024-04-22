/*
C like Program
Vector Addition
*/

// header files

// std header
#include <stdio.h>

// cuda header
#include <cuda.h>	// contains stdlib.h and math.h
#include "helper_timer.h"

// global variables
//const int iNumberOfArrayElements = 5;

const int iNumberOfArrayElements = 11444777;	
// Eleven million four hundred forty four thousand seven hundered seventy seven 


// host == CPU
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold 	  = NULL;	// NVIDIA Sample

// device == GPU
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;


// CUDA Kernel
// no return type
// kernel replaces loop
// __global__ = Call on CPU, run on GPU

__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{

	// code
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// only use the threads created
	if(i < len)
	{
		out[i] = in1[i] + in2[i];
	
	}

}

// entry point function
int main(void)
{
	// function declarations
	void cleanup(void);
	void fillFloatArrayWithRandomNumbers(float*, int);	// fills array with random nos.
	void vecAddCPU(const float*, const float*, float*, int);	// runs on cpus

	// variable declarations
	int 		size 	= iNumberOfArrayElements * sizeof(float);
	cudaError_t result 	= cudaSuccess;
	
	cudaError_t ret_cuda_rt;
	int dev_count;
	
	ret_cuda_rt = cudaGetDeviceCount(&dev_count);

	printf("\x1b[33m CUDA Program To Do Vector Addition \x1b[0m\n");

	if(ret_cuda_rt != cudaSuccess)
	{
		printf("\x1b[31m CUDA RunTime API Error - cudaGetDeviceCount() Failed Due to %s. \n ", cudaGetErrorString(ret_cuda_rt));
	}

	else if(dev_count == 0)
	{
		printf("\x1b[31m There is no CUDA Supported device on this system. \n");
		
		return;
	}
	
	else
	{
		cudaDeviceProp dev_prop;
		
		ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, 0);
		
		printf("\x1b[32m GPU Device Name 			: %s \x1b[0m \n", dev_prop.name);
	
	}
	
	
	// code
	// host memory allocation
	hostInput1 = (float *)malloc(size);	// type cast
	
	if(hostInput1 == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float *)malloc(size);	// type cast
	
	if(hostInput2 == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	
	hostOutput = (float *)malloc(size);	// type cast
	
	if(hostOutput == NULL)
	{
		printf("Host Memory Allocation is failed for hostInput Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(size);	// type cast
	
	if(gold == NULL)
	{
		printf("Host Memory Allocation is failed for gold Array  \n");	
		cleanup();
		exit(EXIT_FAILURE);
	}

	
	
	// Filling values into host array
	fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	// device memory allocation
	result = cudaMalloc((void**)&deviceInput1, size);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Input 1 array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}

	
	result = cudaMalloc((void**)&deviceInput2, size);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Input 2 array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}

	result = cudaMalloc((void**)&deviceOutput, size);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Output array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}


	// copy data from host array into device array
	result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);

	if(result != cudaSuccess)
	{
		printf("Host to Device Data Copy Failed for device Input 1 Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

	if(result != cudaSuccess)
	{
		printf("Host to Device Data Copy Failed for device Input 2 Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	// Kernel Configuration
	dim3 dimGrid = dim3((int) ceil ((float)iNumberOfArrayElements / 256.0f), 1, 1);	// Atleast 256 threads in 1 block
	dim3 dimBlock = dim3(256, 1, 1);		// no. of threads in x,y,z axis

	// CUDA Kernel for Vector Addition
	// Kernel Execution Configuration
	
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);		// allocate memory
	sdkStartTimer(&timer);		// start timer

	// call gpu
	vecAddGPU <<< dimGrid, dimBlock >>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	// copy data from device array into host array
	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

	if(result != cudaSuccess)
	{
		printf("Device to Host Data Copy Failed for HostOutput Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	// vector addition on host
	vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

	//comparison
	const float epsilon = 0.000001f;

	int breakValue = -1;
	bool bAccuracy = true;
	
	for(int i = 0; i < iNumberOfArrayElements; i++)
	{
		float val1 = gold[i];
		float val2 = hostOutput[i];
		
		if(fabs(val1 - val2) > epsilon)
		{
			bAccuracy = false;
			breakValue = i;
			
			break;
		}
		
	}

	char str[128];
	if(bAccuracy == false)
		sprintf(str, "Comparison of CPU and GPU Vector Addition is not within the accuracy of 0.000001 at array index %d", breakValue);

	else
		sprintf(str, "Comparison of CPU and GPU Vector Addition is within the accuracy of 0.000001");

	
	// output
	printf("Array 1 begins from 0th index %.6f to %dth index %.6f \n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);

	printf("Array 2 begins from 0th index %.6f to %dth index %.6f \n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
	
	printf("CUDA Kernel Grid Dimensions = %d, %d, %d and Block Dimensions = %d,%d,%d \n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	
	printf("Output Array begins from 0th index %.6f to %dth index %.6f \n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
	
	printf("\x1b[36m Time Taken For Vector Addition On CPU = %.6f \n", timeOnCPU);
	
	printf("\x1b[32m Time Taken For Vector Addition On GPU = %.6f \n", timeOnGPU);
	
	printf("\x1b[0m %s \n", str);

	printf("\x1b[0m ************************************************************ \n");
		
	printf("\x1b[6m \x1b[36m Developed by - Yashraj Gaikwad 		\n");
	printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
	printf("\x1b[0m \n");
		

	// cleanup
	cleanup();

	return(0);
}


void fillFloatArrayWithRandomNumbers(float* arr, int len)
{
	// code
	const float fscale = 1.0f / (float) RAND_MAX;
	
	for(int i = 0; i < len; i++)
	{
		arr[i] = fscale * rand();	// Random value between 0 and 1
		
	}
	
}


void vecAddCPU(const float* arr1, const float* arr2, float *out, int len)
{
	// code
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	for(int i = 0; i < len; i++)
	{
		out[i] = arr1[i] + arr2[i];
	}
	
	sdkStopTimer(&timer);
	
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
}





void cleanup(void)
{
	// code
	// Free in reverse order of initialization
	
	if(deviceOutput)
	{
		cudaFree(deviceOutput);
		deviceOutput = NULL;
	}
	
	if(deviceInput2)
	{
		cudaFree(deviceInput2);
		deviceInput2 = NULL;
	}
	
	if(deviceInput1)
	{
		cudaFree(deviceInput1);
		deviceInput1 = NULL;
	}
	
	if(gold)
	{
		
		free(gold);
		gold = NULL;
		
	}
	
	// host
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






