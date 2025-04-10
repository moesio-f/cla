extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
}
__host__ __device__ void _vector_add(Vector *a, Vector *b, Vector *dst) {
#if defined(__CUDA__ARCH__)
  int i = threadIdx.x;
  dst->arr[i] = a->arr[i] + b->arr[i];
#else
  for (int i = 0; i < a->dims; i++) {
    dst->arr[i] = a->arr[i] + b->arr[i];
  }
#endif
}

__global__ void _cu_vector_add(Vector *a, Vector *b, Vector *dst) {
  _vector_add(a, b, dst);
}

extern "C" Vector *vector_add(Vector *a, Vector *b, Vector *dst) {
  return cpu_gpu_conditional_apply_vector_operator(
      &_vector_add, NULL, &vec_same_dims_same_devices, a, b, dst, a->dims,
      a->device);
}
