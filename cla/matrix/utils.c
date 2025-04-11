#include "../include/entities.h"
#include "../include/matrix_utils.h"
#include <assert.h>
#include <stdarg.h>
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


