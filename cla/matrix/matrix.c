#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
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
