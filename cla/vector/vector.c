#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

Vector *vector_projection(Vector *a, Vector *b, Vector *dst) {
  double scalar = vector_dot_product(a, b) / vector_dot_product(b, b);
  return vector_mult_scalar(scalar, b, dst);
}

double vector_dot_product(Vector *a, Vector *b) {
  assert(a->dims == b->dims);
  double sum = 0.0;

  for (int i = 0; i < a->dims; i++) {
    sum += a->arr[i] * b->arr[i];
  }

  return sum;
}

double vector_l2_norm(Vector *a) { return sqrt(vector_dot_product(a, a)); }

bool vector_equals(Vector *a, Vector *b) {
  assert(a->dims == b->dims);

  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - b->arr[i]) > 0.001) {
      return false;
    }
  }

  return true;
}
