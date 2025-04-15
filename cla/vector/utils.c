#include "../include/device_management.h"
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
  // Initialize on CPU
  Vector *vector = (Vector *)malloc(sizeof(Vector));
  assert(vector != NULL);

  vector->dims = dims;
  vector->device = NULL;
  vector->arr = (double *)malloc(dims * sizeof(double));
  vector->cu_vector = NULL;

  for (int i = 0; i < dims; i++) {
    double *ptr = vector->arr + i;
    *ptr = value;
  }

  // If device is set, move to GPU
  if (device != NULL) {
    vector_to_cu(vector, device);
  }

  return vector;
}

Vector *create_vector(int dims, CUDADevice *device, ...) {
  // Initialize on CPU
  va_list args;
  va_start(args, device);

  Vector *vector = const_vector(dims, 0.0, NULL);
  for (int i = 0; i < dims; i++) {
    double value = va_arg(args, double);
    double *ptr = vector->arr + i;
    *ptr = value;
  }

  va_end(args);

  // If device, move to GPU
  if (device != NULL) {
    vector_to_cu(vector, device);
  }

  return vector;
}

void destroy_vector(Vector *vector) {
  // Precondition
  assert(vector != NULL);

  if (vector->device != NULL) {
    vector_to_cpu(vector);
  }

  free(vector->arr);
  free(vector);
}

Vector *copy_vector(Vector *a, Vector *dst) {
  // Preconditions A
  assert(a != NULL);

  // Maybe allocate dst
  CUDADevice *device = a->device;
  dst = maybe_alloc_vector(dst, a->dims, device);

  // Precondition V
  assert(vector_has_same_dims_same_devices(a, dst, a));

  if (device != NULL) {
    vector_to_cpu(a);
    vector_to_cpu(dst);
  }

  for (int i = 0; i < dst->dims; i++) {
    dst->arr[i] = a->arr[i];
  }

  if (device != NULL) {
    vector_to_cu(a, device);
    vector_to_cu(dst, device);
  }

  return dst;
}

void print_vector(Vector *a, char *suffix) {
  // Precondition
  assert(a != NULL);

  CUDADevice *device = a->device;
  if (device != NULL) {
    vector_to_cpu(a);
  }

  int i;
  printf("{");
  for (i = 0; i < a->dims - 1; i++) {
    printf("%f, ", a->arr[i]);
  }
  printf("%f}", a->arr[i]);

  if (suffix != NULL) {
    printf("%s", suffix);
  }

  if (device != NULL) {
    vector_to_cu(a, device);
  }
}

bool vector_has_same_dims_same_devices(Vector *a, Vector *b, Vector *dst) {
  assert(a != NULL && b != NULL && dst != NULL);
  return a->device == b->device && b->device == dst->device &&
         a->dims == b->dims && b->dims == dst->dims;
}
