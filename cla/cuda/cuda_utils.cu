extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_utils.h"
#include "../include/vector_utils.h"
#include <assert.h>
}

extern "C" Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *), bool (*validate)(int, ...),
    Vector *a, Vector *b, Vector *dst, int alloc_dims,
    CUDADevice *alloc_device) {
  // Allocate destination Vector if needed
  dst = maybe_alloc_vector(dst, alloc_dims, alloc_device);

  // Assert pre-conditions
  assert(validate(3, a, b, dst));

  // Apply operation
  if (a->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(a, b, dst);
  } else {
    // If it's GPU, add memory management
    // and use <<<...,...>>> syntax;
    gpu_op<<<1, 1>>>(a, b, dst);
  }

  // Return dst
  return dst;
}

extern "C" Matrix *cpu_gpu_conditional_apply_matrix_operator(
    void (*cpu_op)(Matrix *, Matrix *, Matrix *),
    void (*gpu_op)(Matrix *, Matrix *, Matrix *), bool (*validate)(int, ...),
    Matrix *a, Matrix *b, Matrix *dst, int alloc_rows, int alloc_columns,
    CUDADevice *alloc_device) {
  // Allocate destination Matrix if needed
  dst = maybe_alloc_matrix(dst, alloc_rows, alloc_columns, alloc_device);

  // Assert pre-conditions
  assert(validate(3, a, b, dst));

  // Apply operation
  if (a->device == NULL) {
    // If it's CPU, just call it directly
    cpu_op(a, b, dst);
  } else {
    // If it's GPU, add memory management
    // and use <<<...,...>>> syntax;
    gpu_op<<<1, 1>>>(a, b, dst);
  }

  // Return dst
  return dst;
}
