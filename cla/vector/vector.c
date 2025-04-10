#include "../include/vector.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

Vector *maybe_alloc_vector(Vector *ptr, int dims, VectorType type) {
  if (ptr == NULL) {
    ptr = const_vector(dims, type, 0.0);
  }

  return ptr;
}

Vector *const_vector(int dims, VectorType type, double value) {
  Vector *vector = (Vector *)malloc(sizeof(Vector));
  assert(vector != NULL);

  vector->dims = dims;
  vector->type = type;
  vector->arr = (double *)malloc(dims * sizeof(double));
  vector->x = NULL;
  vector->y = NULL;
  vector->z = NULL;

  for (int i = 0; i < dims; i++) {
    double *ptr = vector->arr + i;
    *ptr = value;
    switch (i) {
    case 0:
      vector->x = ptr;
      break;
    case 1:
      vector->y = ptr;
      break;
    case 2:
      vector->z = ptr;
      break;
    }
  }

  return vector;
}

Vector *create_vector(int dims, VectorType type, ...) {
  va_list args;
  va_start(args, type);

  Vector *vector = const_vector(dims, type, 0.0);
  for (int i = 0; i < dims; i++) {
    double value = va_arg(args, double);
    double *ptr = vector->arr + i;
    *ptr = value;
    switch (i) {
    case 0:
      vector->x = ptr;
      break;
    case 1:
      vector->y = ptr;
      break;
    case 2:
      vector->z = ptr;
      break;
    }
  }

  va_end(args);
  return vector;
}

void destroy_vector(Vector *vector) {
  free(vector->arr);
  free(vector);
}

Vector *copy_vector(Vector *a, Vector *dst) {
  if (dst == NULL) {
    dst = (Vector *)malloc(sizeof(Vector));
  }

  dst->dims = a->dims;
  dst->type = a->type;
  dst->arr = (double *)malloc(dst->dims * sizeof(double));
  for (int i = 0; i < dst->dims; i++) {
    double *ptr = dst->arr + i;
    *ptr = a->arr[i];
    switch (i) {
    case 0:
      dst->x = ptr;
      break;
    case 1:
      dst->y = ptr;
      break;
    case 2:
      dst->z = ptr;
      break;
    }
  }

  return dst;
}

void print_vector(Vector *a, char *suffix) {
  int i;
  printf("{");
  for (i = 0; i < a->dims - 1; i++) {
    printf("%f, ", a->arr[i]);
  }
  printf("%f}", a->arr[i]);

  if (suffix != NULL) {
    printf("%s", suffix);
  }
}

Vector *add_vector(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->type);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] + b->arr[i];
  }

  return dst;
}

Vector *sub_vector(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->type);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] - b->arr[i];
  }

  return dst;
}

Vector *cross_product(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims && a->dims == 3);
  dst = maybe_alloc_vector(dst, a->dims, a->type);

  // Calculate cross product
  double x = (*a->y) * (*b->z) - (*a->z) * (*b->y);
  double y = (*a->z) * (*b->x) - (*a->x) * (*b->z);
  double z = (*a->x) * (*b->y) - (*a->y) * (*b->x);

  // Assign new values
  *dst->x = x;
  *dst->y = y;
  *dst->z = z;

  return dst;
}

Vector *element_wise_prod(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->type);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] * b->arr[i];
  }

  return dst;
}

Vector *scalar_mult_vector(double a, Vector *b, Vector *dst) {
  dst = maybe_alloc_vector(dst, b->dims, b->type);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a * b->arr[i];
  }

  return dst;
}

Vector *projection_vector(Vector *a, Vector *b, Vector *dst) {
  float scalar = dot_product(a, b) / dot_product(b, b);
  return scalar_mult_vector(scalar, b, dst);
}

double dot_product(Vector *a, Vector *b) {
  assert(a->dims == b->dims);
  double sum = 0.0;

  for (int i = 0; i < a->dims; i++) {
    sum += a->arr[i] * b->arr[i];
  }

  return sum;
}

double l2_norm(Vector *a) { return sqrt(dot_product(a, a)); }

bool vector_equals(Vector *a, Vector *b) {
  assert(a->dims == b->dims);

  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - b->arr[i]) > 0.001) {
      return false;
    }
  }

  return true;
}
