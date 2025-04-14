extern "C" {
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

KernelLaunchParams get_vector_launch_parametes(CUDADevice *device,
                                               int vec_dims) {
  // Simple algorithm to find an appropriate
  //  number of blocks/threads based on device.
  int max_threads = device->max_threads_per_block;
  int n_threads = max_threads > vec_dims ? vec_dims : max_threads;
  int n_blocks = 1 + (int)ceil((vec_dims - n_threads) / n_threads);

  return {dim3(n_threads), dim3(n_blocks)};
}

KernelLaunchParams get_maitrx_launch_parametes(CUDADevice *device, int mat_rows,
                                               int mat_columns) {
  // Simple algorithm to find an appropriate
  //  number of blocks/threads based on device.
  int max_threads = device->max_threads_per_block;
  int max_threads_dim = (int)floor(sqrt(max_threads));
  dim3 n_threads(max_threads_dim > mat_rows ? mat_rows : max_threads_dim,
                 max_threads_dim > mat_columns ? mat_columns : max_threads_dim);
  dim3 n_blocks(1 + (int)ceil((mat_rows - n_threads.x) / n_threads.x),
                1 + (int)ceil((mat_columns - n_threads.y) / n_threads.y));
  return {n_threads, n_blocks};
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
        get_maitrx_launch_parametes(dst->device, dst->rows, dst->columns);

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
        get_maitrx_launch_parametes(device, dst->rows, dst->columns);

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
