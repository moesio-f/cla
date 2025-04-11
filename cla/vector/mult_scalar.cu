extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
}
__host__ __device__ void _vector_mult_scalar(double *a, Vector *b,
                                             Vector *dst) {
#if defined(__CUDA__ARCH__)
  return;
#else
  for (int i = 0; i < b->dims; i++) {
    dst->arr[i] = (*a) * b->arr[i];
  }
#endif
}

__global__ void _cu_vector_mult_scalar(double *a, Vector *b, Vector *dst) {
  _vector_mult_scalar(a, b, dst);
}

extern "C" Vector *vector_mult_scalar(double a, Vector *b, Vector *dst) {
  return cpu_gpu_conditional_apply_scalar_vector_operator(
      &_vector_mult_scalar, NULL, a, b, dst, b->dims, b->device);
}
