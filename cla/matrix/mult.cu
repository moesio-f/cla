extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
}
__host__ __device__ void _matrix_mult(Matrix *a, Matrix *b, Matrix *dst) {
#if defined(__CUDA__ARCH__)
  return;
#else
  for (int i = 0; i < dst->rows; i++) {
    for (int j = 0; j < dst->columns; j++) {
      double sum = 0.0;

      for (int k = 0; k < a->columns; k++) {
        sum += a->arr[i][k] * b->arr[k][j];
      }

      dst->arr[i][j] = sum;
    }
  }
#endif
}

__global__ void _cu_matrix_mult(Matrix *a, Matrix *b, Matrix *dst) {
  _matrix_mult(a, b, dst);
}

extern "C" Matrix *matrix_mult(Matrix *a, Matrix *b, Matrix *dst) {
  return cpu_gpu_conditional_apply_matrix_operator(
      &_matrix_mult, NULL, &matrix_mult_compat, a, b, dst, a->rows,
      b->columns, a->device);
}
