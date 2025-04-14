#include "../include/constants.h"
#include "../include/cuda_utils.h"
#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>

bool vector_equals(Vector *a, Vector *b) {
  assert(vector_has_same_dims_same_devices(a, b, a));

  // Guarantee a and b are on CPU
  CUDADevice *device = a->device;
  vector_to_cpu(a);
  vector_to_cpu(b);

  // Operation
  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - b->arr[i]) > D_TOL) {
      if (device != a->device) {
        // Maybe send them back to GPU
        vector_to_cu(a, device);
        vector_to_cu(b, device);
      }
      return false;
    }
  }

  if (device != a->device) {
    // Maybe send them back to GPU
    vector_to_cu(a, device);
    vector_to_cu(b, device);
  }

  return true;
}

bool vector_orthogonal(Vector *a, Vector *b) {
  return fabs(vector_dot_product(a, b)) <= D_TOL;
}

bool vector_orthonormal(Vector *a, Vector *b) {
  return vector_orthogonal(a, b) && (fabs(vector_l2_norm(a) - 1.0) <= D_TOL) &&
         (fabs(vector_l2_norm(b) - 1.0) <= D_TOL);
}
