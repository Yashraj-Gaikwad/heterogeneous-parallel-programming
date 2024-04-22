/*
C like Program
Matrix Multiplication
*/

// header files

// std header
#include <stdio.h>

// cuda header
#include <cuda.h>	// contains stdlib.h and math.h
#include "helper_timer.h"

// macros
#define BLOCK_WIDTH 1024

// global variables
int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold = NULL;

int *deviceA = NULL;
int *deviceB = NULL;
int *deviceC = NULL;


float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;


// CUDA Kernel
// no return type
// kernel replaces loop
// __global__ = Call on CPU, run on GPU

__global__ void matMulGPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{

	// code
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	
	if((row < numARows) && (column < numBColumns))
	{
		int value = 1;
		
		for(int k = 0; k < numAColumns; k++)
		{
			int a = A[row * numAColumns + k];
			int b = B[k * numBColumns + column];
			value += a*b;
		}
		
		C[row * numCColumns + column] = value;
	}
	
}

// entry point function
int main(void)
{
	// function declarations
	void cleanup(void);
	void matMulCPU(int*, int*, int*, int, int, int, int);	// runs on cpus
	void InitA(int *data, int, int);
	void InitB(int *data, int, int);

	// variable declarations
	cudaError_t result 	= cudaSuccess;
	cudaError_t ret_cuda_rt;
	int dev_count;
	
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
	
	ret_cuda_rt = cudaGetDeviceCount(&dev_count);

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
	printf("\x1b[33m CUDA Program To Do Matrix Multiplication \x1b[0m\n");
	
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
	
	
	// device memory allocation
	result = cudaMalloc((void**)&deviceA, sizeA);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Input 1 array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}

	
	result = cudaMalloc((void**)&deviceB, sizeB);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Input 2 array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}

	result = cudaMalloc((void**)&deviceC, sizeC);

	if(result != cudaSuccess)
	{
		printf("Device Memory Allocation is Failed for Device Output array \n");
		cleanup();
		exit(EXIT_FAILURE);
		
	}


	// copy data from host array into device array
	result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);

	if(result != cudaSuccess)
	{
		printf("Host to Device Data Copy Failed for device Input 1 Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

	if(result != cudaSuccess)
	{
		printf("Host to Device Data Copy Failed for device Input 2 Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	// Kernel Configuration
	// C matrix has numARows and numBColumns
	dim3 dimGrid = dim3((int) ceil ((int)numBColumns / (int) BLOCK_WIDTH), (int) ceil ((int)numARows / (int) BLOCK_WIDTH), 1);	// Atleast 256 threads in 1 block
	
	dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);		// no. of threads in x,y,z axis

	// CUDA Kernel for Matrix Multiplication
	// Kernel Execution Configuration
	
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);		// allocate memory
	sdkStartTimer(&timer);		// start timer

	// call gpu
	matMulGPU <<< dimGrid, dimBlock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns, numCColumns);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	// copy data from device array into host array
	result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

	if(result != cudaSuccess)
	{
		printf("Device to Host Data Copy Failed for HostOutput Array \n");
		cleanup();
		exit(EXIT_FAILURE);
	}


	// matrix multiplication on host
	matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns);

	//no epsilon
	//const float epsilon = 0.000001f;

	int breakValue = -1;
	bool bAccuracy = true;
	
	for(int i = 0; i < numCRows * numCColumns; i++)
	{
		float val1 = gold[i];
		float val2 = hostC[i];
		
		if(val1 != val2)	// for int no fabs
		{
			bAccuracy = false;
			breakValue = i;
			
			break;
		}
		
	}

	char str[128];
	if(bAccuracy == false)
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is not within the accuracy of 0.000001 at array index %d", breakValue);

	else
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is within the accuracy of 0.000001");

	
	// output 
	
	printf("\x1b[36m Time Taken For Matrix Multiplication On CPU = %.6f \n", timeOnCPU);
	
	printf("\x1b[32m Time Taken For Matrix Multiplication On GPU = %.6f \n", timeOnGPU);
	
	printf("\x1b[0m %s \n", str);

	printf("\x1b[0m ************************************************************ \n");
		
	printf("\x1b[6m \x1b[36m Developed by - Yashraj Gaikwad 		\n");
	printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
	printf("\x1b[0m \n");
		

	// cleanup
	cleanup();

	return(0);
}


void InitA(int *data, int row, int col)
{
	int num = 1;
	
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
	
	// ROMANTIC LINES
	for(int i = 0; i < numARows; ++i)
	{
		for(int j = 0; j < numBColumns; ++j)
		{
			int value = 1;
			
			for(int k = 0; k < numAColumns; ++k)
			{
				int a = A[i * numAColumns + k];
				int b = B[k * numBColumns + j];
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
	// Free in reverse order of initialization
	
	if(deviceC)
	{
		cudaFree(deviceC);
		deviceC = NULL;
	}
	
	if(deviceB)
	{
		cudaFree(deviceB);
		deviceB = NULL;
	}
	
	if(deviceA)
	{
		cudaFree(deviceA);
		deviceA = NULL;
	}
	
	if(gold)
	{
		
		free(gold);
		gold = NULL;
		
	}
	
	// host
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






