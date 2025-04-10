#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/vector_operations.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void assert_same_shape(Matrix *a, Matrix *b) {
  assert(a->rows == b->rows);
  assert(a->columns == b->columns);
}

void assert_compatible_shape(Matrix *a, Matrix *b) {
  assert(a->columns == b->rows);
}

Matrix *maybe_alloc_matrix(Matrix *ptr, int rows, int columns,
                           CUDADevice *device) {
  if (ptr == NULL) {
    ptr = const_matrix(rows, columns, 0.0, device);
  }

  return ptr;
}

Matrix *const_matrix(int rows, int columns, double value, CUDADevice *device) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->rows = rows;
  matrix->columns = columns;
  matrix->device = device;

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

  return matrix;
}

void destroy_matrix(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->arr[i]);
  }
  free(matrix->arr);
  free(matrix);
}

Matrix *copy_matrix(Matrix *a, Matrix *dst) {
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns, a->device);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j];
    }
  }

  return dst;
}

void print_matrix(Matrix *a, char *suffix) {
  int i, j;

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
}

Matrix *matrix_from_vector(Vector *a, Vector2MatrixStrategy strategy) {
  Matrix *matrix = const_matrix(1, a->dims, 0.0, a->device);
  double *target = a->arr;

  // Create copy of original array
  target = (double *)malloc(a->dims * sizeof(double));
  for (int i = 0; i < a->dims; i++) {
    target[i] = a->arr[i];
  }

  matrix->arr[0] = target;
  return matrix;
}

Matrix *matrix_add(Matrix *a, Matrix *b, Matrix *dst) {
  assert_same_shape(a, b);
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns, a->device);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j] + b->arr[i][j];
    }
  }

  return dst;
}

Matrix *matrix_sub(Matrix *a, Matrix *b, Matrix *dst) {
  assert_same_shape(a, b);
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns, a->device);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j] - b->arr[i][j];
    }
  }

  return dst;
}

Matrix *matrix_mult(Matrix *a, Matrix *b) {
  assert_compatible_shape(a, b);
  Matrix *dst = const_matrix(a->rows, b->columns, 0.0, a->device);

  // Apply operation
  for (int i = 0; i < dst->rows; i++) {
    for (int j = 0; j < dst->columns; j++) {
      double sum = 0.0;

      for (int k = 0; k < a->columns; k++) {
        sum += a->arr[i][k] * b->arr[k][j];
      }

      dst->arr[i][j] = sum;
    }
  }

  return dst;
}

Vector *mult_matrix_by_vector(Matrix *a, Vector *b) {
  assert(a->columns == b->dims);
  Vector *dst = const_vector(a->rows, 0.0, a->device);

  for (int i = 0; i < dst->dims; i++) {
    double sum = 0.0;
    for (int j = 0; j < b->dims; j++) {
      sum += a->arr[i][j] * b->arr[j];
    }

    dst->arr[i] = sum;
  }

  return dst;
}

Matrix *matrix_mult_scalar(double a, Matrix *b, Matrix *dst) {
  int rows = b->rows;
  int columns = b->columns;
  dst = maybe_alloc_matrix(dst, rows, columns, b->device);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a * b->arr[i][j];
    }
  }

  return dst;
}
