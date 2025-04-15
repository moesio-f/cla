#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/matrix_utils.h"
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

Matrix *maybe_alloc_matrix(Matrix *ptr, int rows, int columns,
                           CUDADevice *device) {
  if (ptr == NULL) {
    ptr = const_matrix(rows, columns, 0.0, device);
  }

  return ptr;
}

Matrix *const_matrix(int rows, int columns, double value, CUDADevice *device) {
  // Initialize on CPU
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->rows = rows;
  matrix->columns = columns;
  matrix->device = NULL;
  matrix->cu_matrix = NULL;

  // Initializing array
  double **vectors = (double **)malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    vectors[i] = (double *)malloc(columns * sizeof(double));
    for (int j = 0; j < columns; j++) {
      vectors[i][j] = value;
    }
  }

  // Store pointer
  matrix->arr = vectors;

  // If device is set, move to GPU
  if (device != NULL) {
    matrix_to_cu(matrix, device);
  }

  return matrix;
}

void destroy_matrix(Matrix *matrix) {
  // Precondition
  assert(matrix != NULL);

  if (matrix->device != NULL) {
    matrix_to_cpu(matrix);
  }

  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->arr[i]);
  }
  free(matrix->arr);
  free(matrix);
}

Matrix *copy_matrix(Matrix *a, Matrix *dst) {
  // Precondition A
  assert(a != NULL);

  // Maybe allocate dst
  CUDADevice *device = a->device;
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns, device);

  // Precondition B
  assert(matrix_has_same_dims_same_devices(a, dst, a));

  if (device != NULL) {
    matrix_to_cpu(a);
    matrix_to_cpu(dst);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j];
    }
  }

  if (device != NULL) {
    matrix_to_cu(a, device);
    matrix_to_cu(dst, device);
  }

  return dst;
}

void print_matrix(Matrix *a, char *suffix) {
  // Precondition
  assert(a != NULL);

  int i, j;
  CUDADevice *device = a->device;
  matrix_to_cpu(a);

  for (i = 0; i < a->rows; i++) {
    printf("\n[");
    for (j = 0; j < a->columns - 1; j++) {
      printf("%f ", a->arr[i][j]);
    }
    printf("%f]", a->arr[i][j]);
  }

  if (suffix != NULL) {
    printf("%s", suffix);
  }

  if (device != NULL) {
    matrix_to_cu(a, device);
  }
}

Matrix *matrix_from_vector(Vector *a, Vector2MatrixStrategy strategy) {
  // Precondition
  assert(a != NULL);

  int rows = 1, columns = a->dims;
  if (strategy == Vector2MatrixStrategy_COLUMN) {
    rows = a->dims;
    columns = 1;
  }

  // Make assignments on CPU
  CUDADevice *device = a->device;
  vector_to_cpu(a);
  Matrix *matrix = const_matrix(rows, columns, 0.0, NULL);
  double *target = a->arr;

  // Assign values
  if (strategy == Vector2MatrixStrategy_ROW) {
    for (int i = 0; i < a->dims; i++) {
      matrix->arr[0][i] = a->arr[i];
    }
  } else {
    for (int i = 0; i < a->dims; i++) {
      matrix->arr[i][0] = a->arr[i];
    }
  }

  // Maybe send objects back to GPU
  if (device != NULL) {
    vector_to_cu(a, device);
    matrix_to_cu(matrix, device);
  }

  return matrix;
}

bool matrix_has_same_dims_same_devices(Matrix *a, Matrix *b, Matrix *dst) {
  assert(a != NULL && b != NULL && dst != NULL);
  return a->device == b->device && b->device == dst->device &&
         a->rows == b->rows && a->columns == b->columns &&
         b->rows == dst->rows && b->columns == dst->columns;
}

bool matrix_is_mult_compat(Matrix *a, Matrix *b, Matrix *dst) {
  assert(a != NULL && b != NULL && dst != NULL);
  return a->device == b->device && b->device == dst->device &&
         a->columns == b->rows && dst->rows == a->rows &&
         dst->columns == b->columns;
}

bool matrix_is_square(Matrix *a) {
  assert(a != NULL);
  return a->rows == a->columns;
}
