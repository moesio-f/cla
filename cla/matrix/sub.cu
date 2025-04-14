extern "C" {
#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/matrix_operations.h"
#include "../include/matrix_utils.h"
}
__host__ __device__ void _matrix_sub(Matrix *a, Matrix *b, Matrix *dst) {
#if defined(__CUDA__ARCH__)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < a->rows && j < a->columns) {
    dst->arr[i][j] = a->arr[i][j] - b->arr[i][j];
  }

#else
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      dst->arr[i][j] = a->arr[i][j] - b->arr[i][j];
    }
  }
#endif
}

__global__ void _cu_matrix_sub(Matrix *a, Matrix *b, Matrix *dst) {
  _matrix_sub(a, b, dst);
}

extern "C" Matrix *matrix_sub(Matrix *a, Matrix *b, Matrix *dst) {
  return cpu_gpu_conditional_apply_matrix_operator(
      &_matrix_sub, &_cu_matrix_sub, &matrix_has_same_dims_same_devices, a, b,
      dst, a->rows, a->columns, a->device);
}
