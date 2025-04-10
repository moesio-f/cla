#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>

double vector_angle_between_rad(Vector *a, Vector *b) {
  assert(vec_same_dims_same_devices(2, a, b));

  // Guarantee a and b are on CPU
  // ...
  double dot_product = vector_dot_product(a, b);
  double l2_a = vector_l2_norm(a);
  double l2_b = vector_l2_norm(b);

  return acos(dot_product / (l2_a * l2_b));
}
