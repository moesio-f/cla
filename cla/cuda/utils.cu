extern "C" {
#include "../include/constants.h"
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_utils.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
}

#define RED "\x1B[31m"
#define RESET "\x1B[0m"
#include "cuda_runtime_api.h"

typedef struct {
  dim3 n_threads;
  dim3 n_blocks;
} KernelLaunchParams;

typedef enum { PtC_X, PtC_Y, PtC_Z } ParamsToCheck;

int _find_n(int dims, int n, int coef) {
  // Simple algorithm to find an appropriate
  //  number of blocks/threads for the problem.
  // See https://github.com/moesio-f/cla/issues/11 for
  //    a discussion on this method.
  // Find k
  int k = 1 + (int)ceil(pow(dims, 1 / n) / coef);

  // Find n
  return coef * k;
}

bool _are_params_set(CUDAKernelLaunchParameters params, ParamsToCheck check) {
  switch (check) {
  case PtC_X:
    return params.n_threads_x > 0 && params.n_blocks_x > 0;
  case PtC_Y:
    return params.n_threads_y > 0 && params.n_blocks_y > 0;
  case PtC_Z:
    return params.n_threads_z > 0 && params.n_blocks_z > 0;
  }

  return false;
}

bool _are_params_compatible(CUDADevice *device, int dims, ParamsToCheck check) {
  // Obtain n_threads, n_blocks, max_threads, max_blocks
  //    for the selected dimension
  int nt = 0, nb = 0, mt = 0, mb = 0;
  switch (check) {
  case PtC_X:
    nt = device->params.n_threads_x;
    nb = device->params.n_blocks_x;
    mt = device->max_block_size_x;
    mb = device->max_grid_size_x;
    break;
  case PtC_Y:
    nt = device->params.n_threads_y;
    nb = device->params.n_blocks_y;
    mt = device->max_block_size_y;
    mb = device->max_grid_size_y;
    break;
  case PtC_Z:
    nt = device->params.n_threads_z;
    nb = device->params.n_blocks_z;
    mt = device->max_block_size_z;
    mb = device->max_grid_size_z;
    break;
  }

  // Conditions to fit problem in GPU
  return (nt * nb >= dims) && (nt <= mt) && (nb <= mb);
}

KernelLaunchParams get_vector_launch_parametes(CUDADevice *device,
                                               int vec_dims) {
  int n_threads = 0, n_blocks = 0;

  // Check if device has compatible parameters
  if (_are_params_set(device->params, PtC_X)) {
    if (_are_params_compatible(device, vec_dims, PtC_X)) {
      n_threads = device->params.n_threads_x;
      n_blocks = device->params.n_blocks_x;
    } else {
      // Warn user about fallback
      printf(RED
             "[CLA] Device (id=%d, name='%s') has parameters set, but they "
             "are not compatible with either GPU/Problem. Using fallback." RESET
             "\n",
             device->id, device->name);
    }
  }

  // Fallback algorithm
  if (n_threads <= 0) {
    n_threads = _find_n(vec_dims, 2, KERNEL_LAUNCH_COEF);
    n_blocks = n_threads;
  }

  // Assert we have enough threads/blocks to compute
  //    and that those values fit the device capabilities
  int max_threads = device->max_block_size_x;
  int max_grid = device->max_grid_size_x;
  assert(n_threads * n_blocks >= vec_dims);
  assert(n_threads <= max_threads);
  assert(n_blocks <= max_grid);

  // Return parameters
  return {dim3(n_threads), dim3(n_blocks)};
}

KernelLaunchParams get_matrix_launch_parametes(CUDADevice *device, int mat_rows,
                                               int mat_columns) {
  int n_threads_x = 0, n_blocks_x = 0, n_threads_y = 0, n_blocks_y = 0;

  // Check if device has compatible parameters
  if (_are_params_set(device->params, PtC_X) &&
      _are_params_set(device->params, PtC_Y)) {
    if (_are_params_compatible(device, mat_rows, PtC_X) &&
        _are_params_compatible(device, mat_columns, PtC_Y)) {
      n_threads_x = device->params.n_threads_x;
      n_blocks_x = device->params.n_blocks_x;
      n_threads_y = device->params.n_threads_y;
      n_blocks_y = device->params.n_blocks_y;
    } else {
      // Warn user about fallback
      printf(RED
             "[CLA] Device (id=%d, name='%s') has parameters set, but they "
             "are not compatible with either GPU/Problem. Using fallback." RESET
             "\n",
             device->id, device->name);
    }
  }

  // Fallback algorithm
  if (n_threads_x <= 0) {
    n_threads_x = _find_n(mat_rows, 2, KERNEL_LAUNCH_COEF);
    n_threads_y = _find_n(mat_columns, 2, KERNEL_LAUNCH_COEF);
    n_blocks_x = n_threads_x;
    n_blocks_y = n_threads_y;
  }

  // Assert we have enough threads/blocks to compute
  //    and that those values fit the device capabilities
  int max_threads_x = device->max_block_size_x;
  int max_threads_y = device->max_grid_size_y;
  int max_threads = device->max_threads_per_block;
  int max_grid_x = device->max_grid_size_x;
  int max_grid_y = device->max_grid_size_y;
  assert(n_threads_x * n_blocks_x >= mat_rows);
  assert(n_threads_y * n_blocks_y >= mat_columns);
  assert(n_threads_x * n_threads_y <= max_threads);
  assert(n_threads_x <= max_threads_x);
  assert(n_threads_y <= max_threads_y);
  assert(n_blocks_x <= max_grid_x);
  assert(n_blocks_y <= max_grid_y);

  return {dim3(n_threads_x, n_threads_y), dim3(n_blocks_x, n_blocks_y)};
}

extern "C" Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *),
    bool (*validate)(Vector *, Vector *, Vector *), Vector *a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device) {
  // Allocate destination Vector if needed
  dst = maybe_alloc_vector(dst, alloc_dims, alloc_device);

  // Assert pre-conditions
  assert(validate(a, b, dst));

  // Apply operation
  if (a->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(a, b, dst);
  } else {
    KernelLaunchParams params =
        get_vector_launch_parametes(dst->device, dst->dims);

    // Launch the kernel with the cu_vectors
    gpu_op<<<params.n_blocks, params.n_threads>>>(a->cu_vector, b->cu_vector,
                                                  dst->cu_vector);

    // Synchronize CUDA devices
    cudaDeviceSynchronize();
  }

  // Return dst
  return dst;
}

Vector *cpu_gpu_conditional_apply_scalar_vector_operator(
    void (*cpu_op)(double *, Vector *, Vector *),
    void (*gpu_op)(double *, Vector *, Vector *), double a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device) {
  // Allocate destination Matrix if needed
  dst = maybe_alloc_vector(dst, alloc_dims, alloc_device);

  // Apply operation
  if (b->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(&a, b, dst);
  } else {
    CUDADevice *device = dst->device;
    KernelLaunchParams params = get_vector_launch_parametes(device, dst->dims);

    // Allocate temporary memory for double
    double *cu_a = NULL;
    cudaMalloc(&cu_a, sizeof(double));
    cudaMemcpy(cu_a, &a, sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel with the cu_vectors
    gpu_op<<<params.n_blocks, params.n_threads>>>(cu_a, b->cu_vector,
                                                  dst->cu_vector);

    // Synchronize CUDA devices
    cudaDeviceSynchronize();

    // Deallocate memory
    cudaFree(cu_a);
  }

  // Return dst
  return dst;
}

extern "C" Matrix *cpu_gpu_conditional_apply_matrix_operator(
    void (*cpu_op)(Matrix *, Matrix *, Matrix *),
    void (*gpu_op)(Matrix *, Matrix *, Matrix *),
    bool (*validate)(Matrix *, Matrix *, Matrix *), Matrix *a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device) {
  // Allocate destination Matrix if needed
  dst = maybe_alloc_matrix(dst, alloc_rows, alloc_columns, alloc_device);

  // Assert pre-conditions
  assert(validate(a, b, dst));

  // Apply operation
  if (a->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(a, b, dst);
  } else {
    KernelLaunchParams params =
        get_matrix_launch_parametes(dst->device, dst->rows, dst->columns);

    // Launch the kernel with the cu_matrices
    gpu_op<<<params.n_blocks, params.n_threads>>>(a->cu_matrix, b->cu_matrix,
                                                  dst->cu_matrix);

    // Synchronize CUDA devices
    cudaDeviceSynchronize();
  }

  // Return dst
  return dst;
}

extern "C" Matrix *cpu_gpu_conditional_apply_scalar_matrix_operator(
    void (*cpu_op)(double *, Matrix *, Matrix *),
    void (*gpu_op)(double *, Matrix *, Matrix *), double a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device) {
  // Allocate destination Matrix if needed
  dst = maybe_alloc_matrix(dst, alloc_rows, alloc_columns, alloc_device);

  // Apply operation
  if (b->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(&a, b, dst);
  } else {
    CUDADevice *device = dst->device;
    KernelLaunchParams params =
        get_matrix_launch_parametes(device, dst->rows, dst->columns);

    // Allocate temporary memory for double
    double *cu_a = NULL;
    cudaMalloc(&cu_a, sizeof(double));
    cudaMemcpy(cu_a, &a, sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel with the cu_matrices
    gpu_op<<<params.n_blocks, params.n_threads>>>(cu_a, b->cu_matrix,
                                                  dst->cu_matrix);

    // Synchronize CUDA devices
    cudaDeviceSynchronize();

    // Deallocate memory
    cudaFree(cu_a);
  }

  // Return dst
  return dst;
}
