extern "C" {
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
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
  // Vectors should be on same device
  assert(a->device == b->device);

  // Vectors should have same size
  assert(a->dims == b->dims);

  // Allocate dst if needed
  dst = maybe_alloc_vector(dst, a->dims, a->device);

  // Select implementation to run
  if (a->device == NULL) {
    _vector_add(a, b, dst);
  } else {
    // TODO
  }

  // Return dst
  return dst;
}
