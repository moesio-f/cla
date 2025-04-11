#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>

bool matrix_equals(Matrix *a, Matrix *b) {
  assert(matrix_same_dims_same_devices(2, a, b));

  // Guarantee a and b are on CPU
  // ...
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      if (fabs(a->arr[i][j] - b->arr[i][j]) > 0.000001) {
        return false;
      }
    }
  }

  return true;
}
