#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>

bool vector_equals(Vector *a, Vector *b) {
  assert(vec_same_dims_same_devices(2, a, b));

  // Guarantee a and b are on CPU
  // ...
  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - b->arr[i]) > 0.000001) {
      return false;
    }
  }

  return true;
}

bool vector_orthogonal(Vector *a, Vector *b) {
  return fabs(vector_dot_product(a, b)) <= 0.000001;
}

bool vector_orthonormal(Vector *a, Vector *b) {
  return vector_orthogonal(a, b) &&
         (fabs(vector_l2_norm(a) - 1.0) <= 0.00001) &&
         (fabs(vector_l2_norm(b) - 1.0) <= 0.00001);
}
