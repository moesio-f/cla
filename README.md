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

CLA is a simple toy library for basic vector/matrix operations in C. This project main goal is to learn the foundations of [CUDA](https://docs.nvidia.com/cuda/), and Python bindings, using [`ctypes`](https://docs.python.org/3/library/ctypes.html) as a wrapper, through simple Linear Algebra operations (additions, subtraction, multiplication, broadcasating, transformations, etc). 


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

## Installation

For the C-only API, obtain the latest binaries and headers from the [releases](https://github.com/moesio-f/cla/releases) tab in GitHub. For the Python API, use your favorite package manager (i.e., `pip`, `uv`) and install `pycla` from PyPi (e.g., `pip install pycla`).

## C API

The C API provides structs (see [`cla/include/entities.h`](cla/include/entities.h)) and functions (see [`cla/include/vector_operations.h`](cla/include/vector_operations.h), [`cla/include/matrix_operations.h`](cla/include/matrix_operations.h)) that operate over those structs. The two main entities are `Vector` and `Matrix`. A vector or matrix can reside in either the CPU memory (host memory, from CUDA's terminology) or GPU memory (device memory). Those structs always keep metadata on the CPU (i.e., shape, current device), which allows the CPU to coordinate most of the workflow. In order for an operation to be run on the GPU the entities must first be copied to the GPU's memory.

A sample example is as follows:

```c
#include "cla/include/entities.h"
#include "cla/include/matrix_operations.h"
#include "cla/include/matrix_utils.h"
#include "cla/include/vector_operations.h"
#include "cla/include/vector_utils.h"
#include <stdio.h>
#include <stdlib.h>

void print_vector_result(char *title, Vector *a, Vector *b, Vector *dst,
                         char operation) {
  printf("%s\n", title);
  print_vector(a, " ");
  printf("%c ", operation);
  print_vector(b, " ");
  printf("= ");
  print_vector(dst, "\n\n");
}

void print_matrix_result(char *title, Matrix *a, Matrix *b, Matrix *dst,
                         char operation) {
  printf("%s", title);
  print_matrix(a, "\n");
  printf("%c", operation);
  print_matrix(b, "\n");
  printf("=");
  print_matrix(dst, "\n\n");
}

int main() {
  /** Some vector operations. */
  // We can instantiate vectors directly
  Vector vec_a = {(double[2]){1.0, 2.0}, 2, NULL, NULL};

  // ...or using constructors
  Vector *vec_b = create_vector(2, NULL, -1.0, 0.5);
  Vector *vec_dst = const_vector(2, 0.0, NULL);

  // Addition
  vector_add(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Addition", &vec_a, vec_b, vec_dst, '+');

  // Subtraction
  vector_sub(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Subtraction", &vec_a, vec_b, vec_dst, '-');

  // Element-wise product
  vector_element_wise_prod(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Element wise product", &vec_a, vec_b, vec_dst,
                      '*');

  // For vectors created by constructors, we can use the
  //    utility functions to clean-up their memory.
  destroy_vector(vec_b);
  destroy_vector(vec_dst);

  /** Some matrix operations. */
  // Matrix construction is a little more cumbersome
  //    without constructors. The test suite contains
  //    some examples.
  Matrix *mat_a = const_matrix(2, 2, 1.0, NULL);
  Matrix *mat_b = const_matrix(2, 2, 0.5, NULL);
  Matrix *mat_dst = const_matrix(2, 2, 0.0, NULL);

  // We can access the underlying data store
  //    and make changes
  mat_a->arr[0][0] = 2.0;
  mat_b->arr[1][1] = -3.0;

  // Addition
  matrix_add(mat_a, mat_b, mat_dst);
  print_matrix_result("Matrix Addition", mat_a, mat_b, mat_dst, '+');

  // Matrix multiplication
  matrix_mult(mat_a, mat_b, mat_dst);
  print_matrix_result("Matrix Multiplication", mat_a, mat_b, mat_dst, '*');

  // Clean-up
  destroy_matrix(mat_a);
  destroy_matrix(mat_b);
  destroy_matrix(mat_dst);

  return 0;
}
```

Compile with: (i) `gcc -l cla <filename>.c`, if you installed the library system-wide (i.e., copied the headers to `/usr/include/` and shared library to `/usr/lib/`); or (ii) `gcc -I <path-to-include> -L <path-to-root-wih-libcla> -l cla <filename>.c`. To run, make the `.so` findable by the executable (i.e., either update `LD_LIBRARY_PATH` environment variable or include it on `/usr/lib`/`/lib`).

## Python API

TODO

# Build

TODO

## Requirements

TODO

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
  - [`bin`](pycla/bin): binary directory for the `cla` shared library;

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
  cuda("<strong>CUDA Module</strong><br>Utilities for CUDA operations.")

  subgraph pycla
  core -->|Uses| cuda
  end
```

