extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
}
__host__ __device__ void _matrix_mult_scalar(double *a, Matrix *b,
                                             Matrix *dst) {
#if defined(__CUDA__ARCH__)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < b->rows && j < b->columns) {
    dst->arr[i][j] = (*a) * b->arr[i][j];
  }

#else
  for (int i = 0; i < dst->rows; i++) {
    for (int j = 0; j < dst->columns; j++) {
      dst->arr[i][j] = (*a) * b->arr[i][j];
    }
  }
#endif
}

__global__ void _cu_matrix_mult_scalar(double *a, Matrix *b, Matrix *dst) {
  _matrix_mult_scalar(a, b, dst);
}

extern "C" Matrix *matrix_mult_scalar(double a, Matrix *b, Matrix *dst) {
  return cpu_gpu_conditional_apply_scalar_matrix_operator(
      &_matrix_mult_scalar, &_cu_matrix_mult_scalar, a, b, dst, b->rows,
      b->columns, b->device);
}
