#include "../include/cuda_utils.h"
#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>

bool matrix_equals(Matrix *a, Matrix *b) {
  assert(matrix_has_same_dims_same_devices(a, b, a));

  // Guarantee a and b are on CPU
  CUDADevice *device = a->device;
  matrix_to_cpu(a);
  matrix_to_cpu(b);

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      if (fabs(a->arr[i][j] - b->arr[i][j]) > 0.000001) {
        return false;
      }
    }
  }

  // Maybe send back to GPU
  if (a->device != device) {
    matrix_to_cu(a, device);
    matrix_to_cu(b, device);
  }

  return true;
}
