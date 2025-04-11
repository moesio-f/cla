#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include <assert.h>

double matrix_trace(Matrix *a) {
  // Precondition
  assert(matrix_square(a));

  double trace = 0.0;

  // Guarantee a is on CPU
  // ...
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      if (i == j) {
        trace += a->arr[i][j];
      }
    }
  }

  return trace;
}
