extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
}
__host__ __device__ void _vector_element_wise_prod(Vector *a, Vector *b,
                                                   Vector *dst) {
#if defined(__CUDA__ARCH__)
  // Flatten index
  int i = blockIdx.x + threadIdx.x;
  if (i < a->dims) {
    // Only access available addresses
    dst->arr[i] = a->arr[i] * b->arr[i];
  }
#else
  for (int i = 0; i < a->dims; i++) {
    dst->arr[i] = a->arr[i] * b->arr[i];
  }
#endif
}

__global__ void _cu_vector_element_wise_prod(Vector *a, Vector *b,
                                             Vector *dst) {
  _vector_element_wise_prod(a, b, dst);
}

extern "C" Vector *vector_element_wise_prod(Vector *a, Vector *b, Vector *dst) {
  return cpu_gpu_conditional_apply_vector_operator(
      &_vector_element_wise_prod, &_cu_vector_element_wise_prod,
      &vector_has_same_dims_same_devices, a, b, dst, a->dims, a->device);
}
