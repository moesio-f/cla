#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <float.h>
#include <math.h>

double vector_lp_norm(Vector *a, double p) {
  assert(p >= 1.0);
  double norm = 0.0;

  // Guarantee a is on CPU
  // ...
  for (int i = 0; i < a->dims; i++) {
    norm += pow(fabs(a->arr[i]), p);
  }
  norm = pow(norm, 1.0 / p);
  return norm;
}

double vector_l2_norm(Vector *a) { return vector_lp_norm(a, 2.0); }

double vector_max_norm(Vector *a) {
  double max = -DBL_MAX, v;
  // Guarantee a is on CPU
  // ...
  for (int i = 0; i < a->dims; i++) {
    v = a->arr[i];
    if (v > max) {
      max = v;
    }
  }
  return max;
}
