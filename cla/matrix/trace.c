#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include <assert.h>

double matrix_trace(Matrix *a) {
  // Precondition
  assert(matrix_is_square(a));

  double trace = 0.0;

  // Guarantee a and b are on CPU
  CUDADevice *device = a->device;
  matrix_to_cpu(a);

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      if (i == j) {
        trace += a->arr[i][j];
      }
    }
  }

  // Maybe send back to GPU
  if (a->device != device) {
    matrix_to_cu(a, device);
  }

  return trace;
}
