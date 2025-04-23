extern "C" {
#include "../include/constants.h"
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_utils.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
}

#include "cuda_runtime_api.h"

typedef struct {
  dim3 n_threads;
  dim3 n_blocks;
} KernelLaunchParams;

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

KernelLaunchParams get_vector_launch_parametes(CUDADevice *device,
                                               int vec_dims) {
  int max_threads = device->max_threads_per_block;
  int max_grid = device->max_grid_size_x;
  int n = _find_n(vec_dims, 2, KERNEL_LAUNCH_COEF);

  // Assert we have enough threads/blocks to compute
  //    and that those values fit the device capabilities
  assert(n * n >= vec_dims);
  assert(n <= max_threads);
  assert(n <= max_grid);

  return {dim3(n), dim3(n)};
}

KernelLaunchParams get_matrix_launch_parametes(CUDADevice *device, int mat_rows,
                                               int mat_columns) {
  int max_threads = device->max_threads_per_block;
  int max_grid_x = device->max_grid_size_x;
  int max_grid_y = device->max_grid_size_y;
  int max_threads_dim = (int)floor(sqrt(max_threads));
  int n_x = _find_n(mat_rows, 2, KERNEL_LAUNCH_COEF);
  int n_y = _find_n(mat_columns, 2, KERNEL_LAUNCH_COEF);

  // Assert we have enough threads/blocks to compute
  //    and that those values fit the device capabilities
  assert(n_x * n_x >= mat_rows);
  assert(n_y * n_y >= mat_columns);
  assert(n_x <= max_threads_dim);
  assert(n_y <= max_threads_dim);
  assert(n_x <= max_grid_x);
  assert(n_y <= max_grid_y);

  return {dim3(n_x, n_y), dim3(n_x, n_y)};
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

    // Deallocate memory
    cudaFree(cu_a);
  }

  // Return dst
  return dst;
}
