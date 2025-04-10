#include "../include/entities.h"
#include "../include/vector_operations.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

Vector *maybe_alloc_vector(Vector *ptr, int dims, CUDADevice *device) {
  if (ptr == NULL) {
    ptr = const_vector(dims, 0.0, device);
  }

  return ptr;
}

Vector *const_vector(int dims, double value, CUDADevice *device) {
  Vector *vector = (Vector *)malloc(sizeof(Vector));
  assert(vector != NULL);

  vector->dims = dims;
  vector->device = device;
  vector->arr = (double *)malloc(dims * sizeof(double));

  for (int i = 0; i < dims; i++) {
    double *ptr = vector->arr + i;
    *ptr = value;
  }

  return vector;
}

Vector *create_vector(int dims, CUDADevice *device, ...) {
  va_list args;
  va_start(args, device);

  Vector *vector = const_vector(dims, 0.0, device);
  for (int i = 0; i < dims; i++) {
    double value = va_arg(args, double);
    double *ptr = vector->arr + i;
    *ptr = value;
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
  dst->device = a->device;
  dst->arr = (double *)malloc(dst->dims * sizeof(double));
  for (int i = 0; i < dst->dims; i++) {
    double *ptr = dst->arr + i;
    *ptr = a->arr[i];
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

Vector *vector_add(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->device);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] + b->arr[i];
  }

  return dst;
}

Vector *vector_sub(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->device);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] - b->arr[i];
  }

  return dst;
}

Vector *vector_element_wise_prod(Vector *a, Vector *b, Vector *dst) {
  assert(a->dims == b->dims);
  dst = maybe_alloc_vector(dst, a->dims, a->device);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i] * b->arr[i];
  }

  return dst;
}

Vector *vector_mult_scalar(double a, Vector *b, Vector *dst) {
  dst = maybe_alloc_vector(dst, b->dims, b->device);

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a * b->arr[i];
  }

  return dst;
}

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
