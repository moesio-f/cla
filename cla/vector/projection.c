#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"

Vector *vector_projection(Vector *a, Vector *b, Vector *dst) {
  double scalar = vector_dot_product(a, b) / vector_dot_product(b, b);
  return vector_mult_scalar(scalar, b, dst);
}
