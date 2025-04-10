#include "../include/matrix.h"
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

Matrix *maybe_alloc_matrix(Matrix *ptr, int rows, int columns) {
  if (ptr == NULL) {
    ptr = const_matrix(rows, columns, 0.0);
  }

  return ptr;
}

Matrix *const_matrix(int rows, int columns, double value) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->rows = rows;
  matrix->columns = columns;

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
  dst = maybe_alloc_matrix(dst, rows, columns);

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

Matrix *matrix_from_vector(Vector *a, bool copy) {
  Matrix *matrix = const_matrix(1, a->dims, 0.0);
  double *target = a->arr;

  // Create copy of original array if needed
  if (copy) {
    target = (double *)malloc(a->dims * sizeof(double));
    for (int i = 0; i < a->dims; i++) {
      target[i] = a->arr[i];
    }
  }

  matrix->arr[0] = target;
  return matrix;
}

Matrix *add_matrix(Matrix *a, Matrix *b, Matrix *dst) {
  assert_same_shape(a, b);
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j] + b->arr[i][j];
    }
  }

  return dst;
}

Matrix *sub_matrix(Matrix *a, Matrix *b, Matrix *dst) {
  assert_same_shape(a, b);
  int rows = a->rows;
  int columns = a->columns;
  dst = maybe_alloc_matrix(dst, rows, columns);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a->arr[i][j] - b->arr[i][j];
    }
  }

  return dst;
}

Matrix *mult_matrix(Matrix *a, Matrix *b) {
  assert_compatible_shape(a, b);
  Matrix *dst = const_matrix(a->rows, b->columns, 0.0);

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
  Vector *dst = const_vector(a->rows, b->type, 0.0);

  for (int i = 0; i < dst->dims; i++) {
    double sum = 0.0;
    for (int j = 0; j < b->dims; j++) {
      sum += a->arr[i][j] * b->arr[j];
    }

    dst->arr[i] = sum;
  }

  return dst;
}

Matrix *scalar_mult_matrix(float a, Matrix *b, Matrix *dst) {
  int rows = b->rows;
  int columns = b->columns;
  dst = maybe_alloc_matrix(dst, rows, columns);

  // Apply operation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      dst->arr[i][j] = a * b->arr[i][j];
    }
  }

  return dst;
}

Matrix *inverse(Matrix *a, Matrix *dst) {
  double det = determinant(a);
  assert(fabs(det) > 1e-8);
  dst = maybe_alloc_matrix(NULL, a->rows, a->columns);

  // Obtaining matrix entries
  double a_ = a->arr[0][0];
  double b = a->arr[0][1];
  double c = a->arr[0][2];
  double d = a->arr[1][0];
  double e = a->arr[1][1];
  double f = a->arr[1][2];
  double g = a->arr[2][0];
  double h = a->arr[2][1];
  double i = a->arr[2][2];

  // Setting the values of new matrix
  dst->arr[0][0] = e * i - f * h;
  dst->arr[0][1] = -(d * i - f * g);
  dst->arr[0][2] = d * h - e * g;
  dst->arr[1][0] = -(b * i - c * h);
  dst->arr[1][1] = a_ * i - c * g;
  dst->arr[1][2] = -(a_ * h - b * g);
  dst->arr[2][0] = b * f - c * e;
  dst->arr[2][1] = -(a_ * f - c * d);
  dst->arr[2][2] = a_ * e - b * d;

  // Divide by determinant to get the inverse
  dst = scalar_mult_matrix(1.0 / det, dst, dst);
  return dst;
}

double determinant(Matrix *a) {
  assert(a->rows == a->columns && a->rows == 3);

  // Obtaining matrix entries
  double a_ = a->arr[0][0];
  double b = a->arr[0][1];
  double c = a->arr[0][2];
  double d = a->arr[1][0];
  double e = a->arr[1][1];
  double f = a->arr[1][2];
  double g = a->arr[2][0];
  double h = a->arr[2][1];
  double i = a->arr[2][2];

  // Obtaining terms
  double term1 = a_ * e * i;
  double term2 = b * f * g;
  double term3 = c * d * h;
  double term4 = c * e * g;
  double term5 = b * d * i;
  double term6 = a_ * f * h;

  // Obtain the determinant
  return term1 + term2 + term3 - term4 - term5 - term6;
}
