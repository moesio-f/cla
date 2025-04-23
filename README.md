<h1 align="center">
  <div style="display: flex; justify-content: space-between;">
  <a><img src="https://upload.wikimedia.org/wikipedia/commons/1/19/C_Logo.png" alt="C logo" height="80"></a>
  <a><img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png" alt="Python logo" height="80"></a>
  <a><img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg" alt="CUDA logo" height="80"></a>
  </div>
  C Linear Algebra (CLA) Library
  <br>
</h1>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#build">Build</a> •
  <a href="#architecture">Architecture</a>
</p>

CLA is a simple toy library for basic vector/matrix operations in C. This project main goal is to learn the foundations of [CUDA](https://docs.nvidia.com/cuda/), and Python bindings, using [`ctypes`](https://docs.python.org/3/library/ctypes.html) as a wrapper, through simple Linear Algebra operations (additions, subtraction, multiplication, broadcasting, etc). 


# Features

- C17 support, Python 3.13, CUDA 12.8;
- Linux support;
- Vector-vector operations;
- Matrix-matrix operations;
- Vector and matrix norms;
- GPU device selection to run operations;
- Get CUDA information from the system (i.e., support, number of devices, etc);
- Management of memory (CPU memory vs GPU memory), allowing copies between devices;

# Quick Start

> [!IMPORTANT]  
> In order to use the library the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) must be installed and available in the system. Even if you don't intend to use a GPU, the library uses the CUDA Runtime to query for CUDA-capable devices and will fail if the `cudart` is unavailable.

## Installation

For the C-only API, obtain the latest binaries and headers from the [releases](https://github.com/moesio-f/cla/releases) tab in GitHub. For the Python API, use your favorite package manager (i.e., `pip`, `uv`) and install `pycla` from PyPi (e.g., `pip install pycla`).

## C API

The C API provides structs (see [`cla/include/entities.h`](cla/include/entities.h)) and functions (see [`cla/include/vector_operations.h`](cla/include/vector_operations.h), [`cla/include/matrix_operations.h`](cla/include/matrix_operations.h)) that operate over those structs. The two main entities are `Vector` and `Matrix`. A vector or matrix can reside in either the CPU memory (host memory, from CUDA's terminology) or GPU memory (device memory). Those structs always keep metadata on the CPU (i.e., shape, current device), which allows the CPU to coordinate most of the workflow. In order for an operation to be run on the GPU the entities must first be copied to the GPU's memory.

For a quickstart, compile the [samples/c_api.c](samples/c_api.c) with: (i) `gcc -l cla <filename>.c`, if you installed the library system-wide (i.e., copied the headers to `/usr/include/` and shared library to `/usr/lib/`); or (ii) `gcc -I <path-to-include> -L <path-to-root-wih-libcla> -l cla <filename>.c`. 

To run, make the `libcla.so` findable by the executable (i.e., either update `LD_LIBRARY_PATH` environment variable or include it on `/usr/lib`) and run in the shell of your preference (i.e., `./a.out`).

## Python API

The Python API provides an object-oriented approach for using the low-level C API. All features of the C API are exposed by the [`Vector`](pycla/core/vector.py) and [`Matrix`](pycla/core/matrix.py) classes. Some samples are available at [`samples`](samples) using Jupyter Notebooks. The code below showcases the basic features of the API:

```python
# Core entities
from pycla import Vector, Matrix

# Contexts for intensive computation
from pycla.core import ShareDestionationVector, ShareDestionationMatrix

# Vector and Matrices can be instantiated directly from Python lists/sequences
vector = Vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# e.g, for Matrices
# matrix = Matrix([[1, 2, 3, 4], [1, 2, 3, 4]])

# Vector and Matrices can be moved forth and back to the GPU with the `.to(...)` and `.cpu()` methods
# Once an object is on the GPU, we cannot directly read its data from the CPU,
#    however we can still retrieve its metadata (i.e., shape, device name, etc)
vector.to(0)
print(vector)

# We can bring an object back to the CPU with either
#   .to(None) or .cpu() calls
vector.cpu()
print(vector)

# The Vector class overrides the built-in operators
#  of Python. Most of the time, the result of an operation
#  return a new Vector instead of updating the current one
#  in place.
result = vector ** 2
print(result)

# We can also directly release the memory allocated
#  for a vector with
vector.release()
del vector

# Whenever we apply an operation on a Vector/Matrix,
#   a new object is allocated in memory to store the result.
# The only exception are the 'i' operations (i.e., *=, +=, -=, etc),
#   which edit the object in place.
# However, for some extensive computation, it is desirable to
#   waste as little memory and time as possible. Thus, the
#   ShareDestination{Vector,Matrix} contexts allow for using
#   a single shared object for most operation with vectors and matrices.
a = Vector([1.0] * 10)
b = Vector([2.0] * 10)
with ShareDestionationVector(a, b) as result:
    op1 = a + b
    op2 = result * 2
    op3 = result / 2

# All op1, op2 and op3 vectors represent the
#  same vector.
print(result)
print(Vector.has_shared_data(op1, result))
print(Vector.has_shared_data(op2, result))
print(Vector.has_shared_data(op3, result))
```

# Build

The whole library can be built using the `make` targets defined on the [Makefile](Makefile). All you have to do is make the required libraries available on the system (i.e., install CUDA 12.8, Python 3.13, gcc/g++ 17, CMake 4.0.0) and install the Python libraries for development (i.e., [py-dev-requirements](py-dev-requirements.txt)). The table below describes the main targets that can be run with `make <target>`.

| Target | Description |
| --- | --- |
| `all` | Prepare and compile the `CLA` library and install the library (`.so`) in [`pycla.bin`](pycla/bin) |
| `test` | Run all unit tests for `cla` and `pycla`. |
| `release` | Run tests for `cla` and `pycla` and create release files (i.e., Python wheel and C zip file). |
| `clean` | Utility target that removes the CMake build directory. |
| `test-cla-memory-leak` | Runs Valgrind and CUDA compute sanitizer for memory leaks in the C API. |
| `test-pycla` | Run tests for the Python API only. | 



# Architecture

The library is organized as simply as possible. The goal is to make a slight distinction between the C and Python APIs, while allowing the core code with CUDA to be flexible.

The C API provides a shared library named `cla` to be used by other programs/libraries during the linking stage or runtime. This C library is static linked to the CUDA kernel/functions during build.

The Python API provides a wrapper to the `cla` library by a Python package named `pycla`, which dynamics load the `cla` library during runtime. It is necessaary to have the CUDA runtime available to use CUDA-related functionanilty.

The aforementioned relationship is depicted in the diagram below:

```mermaid
flowchart LR
  cla("`cla`")
  pycla("`pycla`")
  cuda["CUDA code"]

  cla-.->|Static links| cuda
  pycla==>|Dynamic loads| cla
```

## Directory structure

The source code is organized as follows:

- [`cla`](cla): source code for the C API;
  - [`include`](cla/include): header files (i.e., `.h`, `.cuh`), has subdirectories for each module (e.g., `cuda`, `vector`, `matrix`);
  - [`matrix`](cla/matrix): matrix module;
  - [`vector`](cla/vector): vector module;
  - [`cuda`](cla/cuda): CUDA management code;
- [`pycla`](pycla): source code for the Python API;
  - [`bin`](pycla/bin): wrapper for the `cla` shared library;
  - [`core`](pycla/core): core entities;

## `cla` library

The following diagram shows the module/package organization.

```mermaid
flowchart TD
  vector("<strong>Vector Module</strong><br>Vector operations, norms, conversions.")
  matrix("<strong>Matrix Module</strong><br>Matrix operations, norms, conversions, vector-matrix operations.")
  cuda("<strong>CUDA Module</strong><br> alternative operations for Matrix and Vectors with CUDA kernels.")

  subgraph cla
  matrix -->|Uses for Matrix-Vector operations| vector
  matrix -->|Uses for parallel operations| cuda
  vector -->|Uses for parallel operations| cuda
  end
```

## `pycla` library

The following diagram shows the module/package organization.

```mermaid
flowchart TD
  core("<strong>Core Module</strong><br>Core entities.")
  wrapper("<strong>CLA Module</strong><br>CLA wrapper with ctypes.")

  subgraph pycla
  core -->|Uses| wrapper
  end
```

