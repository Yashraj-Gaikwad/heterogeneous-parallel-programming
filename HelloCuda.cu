/*
C like Program
Hello CUDA
*/

// header files

// std header
#include <stdio.h>

// cuda header
#include <cuda.h>	// contains stdlib.h and math.h

// global variables
const int iNumberOfArrayElements = 5;

// host == CPU
float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;


// device == GPU
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;


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

	// variable declarations
	int 		size 	= iNumberOfArrayElements * sizeof(float);
	cudaError_t result 	= cudaSuccess;
	
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
	dim3 dimGrid = dim3(iNumberOfArrayElements, 1, 1);
	dim3 dimBlock = dim3(1, 1, 1);		// no. of threads in x,y,z axis

	// CUDA Kernel for Vector Addition
	// Kernel Execution Configuration
	
	vecAddGPU <<< dimGrid, dimBlock >>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

	// copy data from device array into host array
	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

	if(result != cudaSuccess)
	{
		printf("Device to Host Data Copy Failed for HostOutput Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	// vector addition on host
	for(int i = 0; i < iNumberOfArrayElements; i++)
	{
		printf("\x1b[32m\n");  // Green
		printf("%f + %f = %f \n ", hostInput1[i], hostInput2[i], hostOutput[i]);
		printf("\x1b[0m\n");   // Reset to default color
	}

	printf("\x1b[0m ************************************************************ \n");
		
	printf("\x1b[6m \x1b[36m Developed by - Yashraj Gaikwad 		\n");
	printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
	printf("\x1b[0m \n");
		

	// cleanup
	cleanup();

	return(0);
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






