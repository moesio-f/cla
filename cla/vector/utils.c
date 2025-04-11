#include "../include/entities.h"
#include "../include/vector_utils.h"
#include <assert.h>
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

bool vector_has_same_dims_same_devices(Vector *a, Vector *b, Vector *dst) {
  return a->device == b->device && b->device == dst->device &&
         a->dims == b->dims && b->dims == dst->dims;
}
