#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>

// OpenCl Header
#include <opencl.h>
//#include <CL/cl_platform.h>

int main(void)
{
	// function declaration
	void printOpenCLDeviceProperties(void);
	
	// code
	printOpenCLDeviceProperties();
	
}

void printOpenCLDeviceProperties(void)
{
	// Code
	printf("\x1b[36m OpenCL Information : \n");
	printf("\x1b[0m ======================================================== \n");
	
	// data Type	variable name
	cl_int 			result;
	cl_platform_id 	ocl_platform_id; 	// ocl - OpenCl
	cl_uint 		dev_count;
	cl_device_id 	*ocl_device_ids;
	char 			oclPlatformInfo[512];
	
	// get first platform ID
	result = clGetPlatformIDs(1, &ocl_platform_id, NULL);

	if(result != CL_SUCCESS)
	{
		printf("clGetPlatformIDs() Failed \n");
		
		exit(EXIT_FAILURE);
	}

	// get GPU Device Count
	result = clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);

	if(result != CL_SUCCESS)
	{
		printf("clGetDeviceIDs() Failed \n");
		
		exit(EXIT_FAILURE);
	}

	else if(dev_count == 0)
	{
		printf("There is No OpenCl Supported Device on this System \n");
		
		exit(EXIT_FAILURE);	
	}

	// SUCCESS
	else
	{
		// get platform name
		clGetPlatformInfo(ocl_platform_id, CL_PLATFORM_NAME, 500, &oclPlatformInfo, NULL);
		
		printf("OpenCL Supporting GPU Platform Name : %s  \n", oclPlatformInfo);
	
		// get platform version
		clGetPlatformInfo(ocl_platform_id, CL_PLATFORM_VERSION, 500, &oclPlatformInfo, NULL);
	
		printf("OpenCL Supporting GPU Platform Version : %s \n", oclPlatformInfo);
	
		// print supporting device no.
		printf("Total Number of OpenCL Supporting GPU Device/Devices On This System : %d \n", dev_count);
	
		// allocate memory to hold those device ids
		ocl_device_ids = (cl_device_id *)malloc(sizeof(cl_device_id) * dev_count);
	
		// get ids into allocated buffer
		clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_GPU, dev_count, ocl_device_ids, NULL);
	
		char ocl_dev_prop[1024];
	
		int i;
		
		for(i = 0; i < (int)dev_count; i++)
		{
			printf("\n");
			printf("\x1b[36m ******** GPU DEVICE GENERAL INFORMATION ********\n");
			printf("\x1b[0m ===================================================================================\n");
			
			printf("GPU Device Number 												: %d \n", i);
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_NAME, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			
			printf("GPU Device Name													: %s \n", ocl_dev_prop);
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VENDOR, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			
			printf("GPU Device Vendor												: %s \n", ocl_dev_prop);
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DRIVER_VERSION, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			
			printf("GPU Device Driver Version											: %s \n", ocl_dev_prop);
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VERSION, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			
			printf("GPU Device OpenCL Version											: %s \n", ocl_dev_prop);
			
			cl_uint clock_frequency;
			
			clGetDeviceInfo(ocl_device_ids [i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
			
			printf("GPU Device Clock Rate												: %u \n", clock_frequency);
			
			
			
			printf("\n");
			printf("\x1b[36m ******** GPU DEVICE MEMORY INFORMATION ********\n");
			printf("\x1b[0m ===========================================================================\n");
			
			cl_ulong mem_size;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
			
			printf("GPU Device Global Memory										: %llu Bytes\n", (unsigned long long)mem_size);
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
			
			printf("GPU Device Constant Buffer Size										: %llu Bytes \n", (unsigned long long)mem_size);
			
			
			
			
			printf("\n");
			printf("\x1b[36m ******** GPU DEVICE COMPUTE INFORMATION ********\n");
			printf("\x1b[0m ==============================================================================\n");
			
			cl_uint compute_units;
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
			
			printf("GPU Device Number of Parallel Processors Cores							: %u \n", compute_units);
			
			size_t workgroup_size;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
			
			printf("GPU Device Number Work Group size								: %u \n", (unsigned int)workgroup_size);
			
			size_t workitem_dims;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
			
			printf("GPU Device Number Work Item Dimension								: %u \n", (unsigned int)workitem_dims);
			
			size_t workitem_size[3];
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
			
			printf("GPU Device Number Work Item Sizes								: %u / %u / %u  \n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
			
			printf("\n \x1b[0m ************************************************************ \n");
		
			printf("\x1b[6m \x1b[32m Developed by - Yashraj Gaikwad 		\n");
			printf("\x1b[6m \x1b[31m Inspired by - Dr. Vijay Gokhale \n");
			printf("\x1b[0m \n");
		
			
			
		}	
		
		free(ocl_device_ids);
	
	}


	
	
}







