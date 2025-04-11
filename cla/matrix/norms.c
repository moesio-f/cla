#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include <assert.h>
#include <float.h>
#include <math.h>

double matrix_lpq_norm(Matrix *a, double p, double q) {
  assert(p >= 1.0 && q >= 1.0);
  double norm = 0.0;

  // Guarantee a is on CPU
  // ...
  for (int j = 0; j < a->columns; j++) {
    double column_norm = 0.0;

    for (int i = 0; i < a->rows; i++) {
      column_norm += pow(fabs(a->arr[i][j]), p);
    }

    norm += pow(column_norm, q / p);
  }

  norm = pow(norm, 1.0 / q);
  return norm;
}

double matrix_frobenius_norm(Matrix *a) { return matrix_lpq_norm(a, 2.0, 2.0); }
