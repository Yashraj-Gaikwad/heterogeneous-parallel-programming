# Heterogeneous Parallel Programming using CUDA and OpenCL in C++

Welcome to the repository for Heterogeneous Parallel Programming using CUDA and OpenCL in C++. This repository contains various projects and examples demonstrating the use of CUDA and OpenCL for parallel programming on GPUs.

## Projects

### 1. Device GPU Properties Program

**Description:**  
This program retrieves and displays the properties of the available GPU devices. It provides detailed information such as device name, total memory, compute capability, and other relevant properties.

**Files:**
- `DeviceProperties_CUDA.cu`
- `DeviceProperties_OpenCL.cpp`

### 2. Hello World

**Description:**  
A simple "Hello World" program demonstrating the basics of writing and running CUDA and OpenCL kernels.

**Files:**
- `HelloWorld_CUDA.cu`
- `HelloWorld_OpenCL.cpp`

### 3. Vector Addition

**Description:**  
This project implements vector addition using both CUDA and OpenCL. It showcases how to set up data on the host, transfer it to the GPU, perform the computation, and retrieve the results.

**Files:**
- `VectorAddition_CUDA.cu`
- `VectorAddition_OpenCL.cpp`

### 4. Matrix Multiplication

**Description:**  
A more complex example of matrix multiplication using CUDA and OpenCL. This project illustrates the efficient handling of two-dimensional data and highlights the performance benefits of parallel computing.

**Files:**
- `MatrixMultiplication_CUDA.cu`
- `MatrixMultiplication_OpenCL.cpp`

## Getting Started

### Prerequisites

- CUDA Toolkit: [Download here](https://developer.nvidia.com/cuda-downloads)
- OpenCL SDK: Available from your GPU vendor (e.g., NVIDIA, AMD, Intel)
- A compatible C++ compiler (e.g., GCC, MSVC)
- CMake (optional for build management)

### Building and Running the Projects

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/heterogeneous-parallel-programming.git
   cd heterogeneous-parallel-programming
   ```

2. **Navigate to the project directory and compile the code:**

   For CUDA:
   ```bash
   nvcc -o device_properties_cuda DeviceProperties_CUDA.cu
   ./device_properties_cuda
   ```

   For OpenCL:
   ```bash
   g++ -o device_properties_opencl DeviceProperties_OpenCL.cpp -lOpenCL
   ./device_properties_opencl
   ```

3. **Follow similar steps for other projects (Hello World, Vector Addition, Matrix Multiplication) by navigating to their respective directories and compiling the source files.**

### Usage

Each project directory contains detailed instructions on how to compile and run the programs. Refer to the comments within the source files for further information on the implementation details.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with your enhancements, bug fixes, or new examples.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Dr.Vijay Gokhale Sir Guruji Teacher at Astromedicomp for their HPP Seminar
- Special thanks to the developers of CUDA and OpenCL for providing the tools and libraries that make parallel programming accessible.
- Thanks to the open-source community for their contributions and support.

