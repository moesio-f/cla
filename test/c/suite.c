/* C API test suite.
 *
 * This file contains multiple tests for the
 *  C API.
 *
 * */
#include "../../cla/include/device_management.h"
#include "../../cla/include/entities.h"
#include "../../cla/include/matrix_operations.h"
#include "../../cla/include/matrix_utils.h"
#include "../../cla/include/vector_operations.h"
#include "../../cla/include/vector_utils.h"
#include "colors.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#define N_TESTS 26

// Utility enumeration for custom error codes
typedef enum { SUCCESS, FAILED } RETURN_CODE;

// Global variables for test suite
int _TESTS_RUN = 0;
int _TESTS_PASSED = 0;
CUDADevice *_DEVICE = NULL;

RETURN_CODE
_test_vector_binop(Vector *(*operation)(Vector *, Vector *, Vector *), int dims,
                   double *a_value, double *b_value, double *target_value,
                   CUDADevice *device) {
  _TESTS_RUN++;
  RETURN_CODE code = SUCCESS;

  // Vector allocation
  Vector *a = const_vector(dims, 0.0, NULL);
  Vector *b = const_vector(dims, 0.0, NULL);
  Vector *target = const_vector(dims, 0.0, NULL);

  // Update values
  for (int i = 0; i < dims; i++) {
    a->arr[i] = a_value[i];
    b->arr[i] = b_value[i];
    target->arr[i] = target_value[i];
  }

  // Send to GPU if any
  if (device != NULL) {
    vector_to_cu(a, device);
    vector_to_cu(b, device);
    vector_to_cu(target, device);
  }

  // Operation
  Vector *result = operation(a, b, NULL);

  // Validation
  code = vector_equals(result, target) ? SUCCESS : FAILED;

  if (code == SUCCESS) {
    _TESTS_PASSED++;
  }

  // Clean-up
  destroy_vector(a);
  destroy_vector(b);
  destroy_vector(target);
  destroy_vector(result);

  return code;
}

RETURN_CODE
_test_matrix_binop(Matrix *(*operation)(Matrix *, Matrix *, Matrix *), int rows,
                   int columns, double **a_value, double **b_value,
                   double **target_value, CUDADevice *device) {
  _TESTS_RUN++;
  RETURN_CODE code = SUCCESS;

  // Matrix allocation
  Matrix *a = const_matrix(rows, columns, 1.0, NULL);
  Matrix *b = const_matrix(rows, columns, 0.0, NULL);
  Matrix *target = const_matrix(rows, columns, -1.0, NULL);

  // Update values
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      a->arr[i][j] = a_value[i][j];
      b->arr[i][j] = b_value[i][j];
      target->arr[i][j] = target_value[i][j];
    }
  }

  // Operation
  Matrix *result = operation(a, b, NULL);

  // Validation
  code = matrix_equals(result, target) ? SUCCESS : FAILED;

  if (code == SUCCESS) {
    _TESTS_PASSED++;
  }

  // Clean-up dynamically allocated matrices
  destroy_matrix(a);
  destroy_matrix(b);
  destroy_matrix(target);
  destroy_matrix(result);

  return code;
}

RETURN_CODE test_vector_add_cpu() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {2.0, 1.5};

  return _test_vector_binop(&vector_add, 2, a, b, target, NULL);
}

RETURN_CODE test_vector_add_cuda() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {2.0, 1.5};

  return _test_vector_binop(&vector_add, 2, a, b, target, _DEVICE);
}

RETURN_CODE test_vector_sub_cpu() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {0.0, -0.5};

  return _test_vector_binop(&vector_sub, 2, a, b, target, NULL);
}

RETURN_CODE test_vector_sub_cuda() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {0.0, -0.5};

  return _test_vector_binop(&vector_sub, 2, a, b, target, _DEVICE);
}

RETURN_CODE test_vector_element_wise_prod_cpu() {
  double a[2] = {3.0, -0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {3.0, -0.5};

  return _test_vector_binop(&vector_element_wise_prod, 2, a, b, target, NULL);
}

RETURN_CODE test_vector_element_wise_prod_cuda() {
  double a[2] = {3.0, -0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {3.0, -0.5};

  return _test_vector_binop(&vector_element_wise_prod, 2, a, b, target,
                            _DEVICE);
}

RETURN_CODE test_vector_dot_product_cpu() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector a = {(double[2]){2.0, 2.0}, 2, NULL};
  if (fabs(vector_dot_product(&a, &a) - 8.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_vector_l2_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector a = {(double[2]){2.0, 2.0}, 2, NULL};
  if (fabs(vector_l2_norm(&a) - sqrt(8.0)) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_vector_lp_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector a = {(double[2]){3.0, 3.0}, 2, NULL};

  if (fabs(vector_lp_norm(&a, 1.0) - 6.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_vector_max_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector a = {(double[4]){0.0, -1.0, 3.0, 10.0}, 4, NULL};

  if (fabs(vector_max_norm(&a) - 10.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_vector_equals() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector a = {(double[3]){1.42, -0.142, 3.321}, 3, NULL};

  if (vector_equals(&a, &a)) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_vector_copy() {
  _TESTS_RUN++;
  RETURN_CODE code = SUCCESS;

  // CPU vectors
  Vector *a = const_vector(100, -20.0, NULL);
  Vector *cpu_dst = const_vector(100, -84.0, NULL);

  // GPU vectors
  Vector *b = const_vector(100, 10.0, _DEVICE);
  Vector *gpu_dst = const_vector(100, 32.0, _DEVICE);

  // Test multiple combinations of copies
  Vector *src[2] = {a, b};
  Vector *dst[3] = {cpu_dst, gpu_dst, NULL};
  for (int src_idx = 0; src_idx < 3; src_idx++) {
    for (int dst_idx = 0; dst_idx < 3; dst_idx++) {
      Vector *loop_dst = dst[dst_idx];
      Vector *loop_src = src[src_idx];

      if (loop_dst != NULL && loop_dst->device != loop_src->device) {
        // Copying between devices is not supported
        continue;
      }

      loop_dst = copy_vector(loop_src, loop_dst);

      // Fail conditions
      bool correct_dst = (dst[dst_idx] != NULL && loop_dst == dst[dst_idx]) ||
                         (dst[dst_idx] == NULL && loop_dst != NULL);
      bool values_equals = vector_equals(loop_src, loop_dst);
      if (!correct_dst || !values_equals) {
        code = FAILED;
        break;
      }

      // Clean-up temporary dst
      if (dst[dst_idx] == NULL) {
        destroy_vector(loop_dst);
      }
    }
  }

  // Clean-up
  destroy_vector(a);
  destroy_vector(cpu_dst);
  destroy_vector(b);
  destroy_vector(gpu_dst);

  if (code == SUCCESS) {
    _TESTS_PASSED++;
  }

  return code;
}

RETURN_CODE test_matrix_add_cpu() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {4.0, 0.5}, t_2[2] = {-4.0, 9.0};

  return _test_matrix_binop(&matrix_add, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            NULL);
}

RETURN_CODE test_matrix_add_cuda() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {4.0, 0.5}, t_2[2] = {-4.0, 9.0};

  return _test_matrix_binop(&matrix_add, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            _DEVICE);
}

RETURN_CODE test_matrix_sub_cpu() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {2.0, -1.5}, t_2[2] = {-6.0, 7.0};

  return _test_matrix_binop(&matrix_sub, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            NULL);
}

RETURN_CODE test_matrix_sub_cuda() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {2.0, -1.5}, t_2[2] = {-6.0, 7.0};

  return _test_matrix_binop(&matrix_sub, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            _DEVICE);
}

RETURN_CODE test_matrix_mult_cpu() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {2.0, 2.0};
  double t_1[2] = {2.0, 2.0}, t_2[2] = {11.0, 11.0};

  return _test_matrix_binop(&matrix_mult, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            NULL);
}

RETURN_CODE test_matrix_mult_cuda() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {2.0, 2.0};
  double t_1[2] = {2.0, 2.0}, t_2[2] = {11.0, 11.0};

  return _test_matrix_binop(&matrix_mult, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            _DEVICE);
}

RETURN_CODE test_matrix_mult_scalar_cpu() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  double a = 2.0;
  double b_1[2] = {1.0, 1.0}, b_2[2] = {2.0, 2.0};
  double t_1[2] = {2.0, 2.0}, t_2[2] = {4.0, 4.0};

  Matrix b = {(double *[2]){b_1, b_2}, 2, 2, NULL};
  Matrix t = {(double *[2]){t_1, t_2}, 2, 2, NULL};
  Matrix *dst = matrix_mult_scalar(a, &b, NULL);
  if (matrix_equals(dst, &t)) {
    _TESTS_PASSED++;
    code = SUCCESS;
  }

  destroy_matrix(dst);
  return code;
}

RETURN_CODE test_matrix_trace() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;

  double a_1[2] = {1.0, 1.0}, a_2[2] = {2.0, 2.0};
  Matrix a = {(double *[2]){a_1, a_2}, 2, 2, NULL};

  if (fabs(matrix_trace(&a) - 3.0) <= 0.0001) {
    _TESTS_PASSED++;
    code = SUCCESS;
  }

  return code;
}

RETURN_CODE test_matrix_lpq_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;

  double a_1[2] = {1.0, 1.0}, a_2[2] = {2.0, 2.0};
  Matrix a = {(double *[2]){a_1, a_2}, 2, 2, NULL};

  if (fabs(matrix_lpq_norm(&a, 1.0, 1.0) - 6.0) <= 0.0001) {
    _TESTS_PASSED++;
    code = SUCCESS;
  }

  return code;
}

RETURN_CODE test_matrix_frobenius_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;

  double a_1[2] = {1.0, 1.0}, a_2[2] = {2.0, 2.0};
  Matrix a = {(double *[2]){a_1, a_2}, 2, 2, NULL};

  if (fabs(matrix_frobenius_norm(&a) - 3.1623) <= 0.0001) {
    _TESTS_PASSED++;
    code = SUCCESS;
  }

  return code;
}

RETURN_CODE test_cuda_devices() {
  _TESTS_RUN++;
  if (has_cuda()) {
    char info[512];
    printf(MAG "Found the following devices:\n" RESET);
    for (int i = 0; i < cuda_get_device_count(); i++) {
      printf(MAG "%s\n" RESET, device_to_str(get_device_by_id(i), info));
    }
  } else {
    printf(RED "No device found." RESET);
  }

  _TESTS_PASSED++;
  return SUCCESS;
}

RETURN_CODE test_vector_move_cuda_cpu() {
  _TESTS_RUN++;
  if (!has_cuda()) {
    _TESTS_PASSED++;
    return SUCCESS;
  }

  // Get first device
  CUDADevice *device = get_device_by_id(0);

  // Create vector on CPU
  Vector *a = const_vector(10, 1.0, NULL);

  // If not on CPU
  if (a->arr == NULL || a->device != NULL || a->cu_vector != NULL) {
    return FAILED;
  }

  // Move it to GPU
  vector_to_cu(a, device);

  // If not on GPU
  if (a->arr != NULL || a->cu_vector == NULL || a->device != device) {
    return FAILED;
  }

  // Move it back to CPU
  vector_to_cpu(a);

  // If not on CPU
  if (a->arr == NULL || a->device != NULL || a->cu_vector != NULL) {
    return FAILED;
  }

  // Check if values are correct
  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - 1.0) > 0.00001) {
      destroy_vector(a);
      return FAILED;
    }
  }

  destroy_vector(a);
  _TESTS_PASSED++;
  return SUCCESS;
}

RETURN_CODE test_matrix_move_cuda_cpu() {
  _TESTS_RUN++;
  if (!has_cuda()) {
    _TESTS_PASSED++;
    return SUCCESS;
  }

  double target = 232.5;

  // Get first device
  CUDADevice *device = get_device_by_id(0);

  // Create matrix on CPU
  Matrix *a = const_matrix(1000, 1000, target, NULL);

  // If not on CPU
  if (a->arr == NULL || a->device != NULL || a->cu_matrix != NULL) {
    return FAILED;
  }

  // Move it to GPU
  matrix_to_cu(a, device);

  // If not on GPU
  if (a->arr != NULL || a->cu_matrix == NULL || a->device != device) {
    return FAILED;
  }

  // Move it back to CPU
  matrix_to_cpu(a);

  // If not on CPU
  if (a->arr == NULL || a->device != NULL || a->cu_matrix != NULL) {
    return FAILED;
  }

  // Check if values are correct
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->columns; j++) {
      if (fabs(a->arr[i][j] - target) > 0.00001) {
        destroy_matrix(a);
        return FAILED;
      }
    }
  }

  destroy_matrix(a);
  _TESTS_PASSED++;
  return SUCCESS;
}

RETURN_CODE test_user_kernel_launch() {
  _TESTS_RUN++;
  if (!has_cuda()) {
    _TESTS_PASSED++;
    return SUCCESS;
  }

  RETURN_CODE code = FAILED;

  // Get first device
  CUDADevice *device = get_device_by_id(0);

  // Create vector on GPU
  Vector *a = const_vector(100, 1.0, device);
  Vector *target = const_vector(100, 4.0, device);

  // Run operation with unsupported parameters
  device->params.n_threads_x = 1;
  device->params.n_blocks_x = 1;
  vector_add(a, a, a);

  // Run operation with supported parameters
  device->params.n_threads_x = 100;
  device->params.n_blocks_x = 1;
  vector_add(a, a, a);

  // Check equality
  if (!vector_equals(a, target)) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  // Clean-up
  destroy_vector(a);
  destroy_vector(target);

  return code;
}

int main() {
  // Initialize default CUDA device
  if (has_cuda()) {
    _DEVICE = get_device_by_id(0);
  }

  RETURN_CODE (*test_fn[N_TESTS])(void) = {&test_vector_add_cpu,
                                           &test_vector_sub_cpu,
                                           &test_vector_element_wise_prod_cpu,
                                           &test_vector_dot_product_cpu,
                                           &test_vector_l2_norm,
                                           &test_vector_lp_norm,
                                           &test_vector_max_norm,
                                           &test_vector_equals,
                                           &test_matrix_add_cpu,
                                           &test_matrix_sub_cpu,
                                           &test_matrix_mult_cpu,
                                           &test_matrix_mult_scalar_cpu,
                                           &test_matrix_trace,
                                           &test_matrix_lpq_norm,
                                           &test_matrix_frobenius_norm,
                                           &test_cuda_devices,
                                           &test_vector_move_cuda_cpu,
                                           &test_vector_add_cuda,
                                           &test_vector_sub_cuda,
                                           &test_vector_element_wise_prod_cuda,
                                           &test_matrix_move_cuda_cpu,
                                           &test_matrix_add_cuda,
                                           &test_matrix_sub_cuda,
                                           &test_matrix_mult_cuda,
                                           &test_vector_copy,
                                           &test_user_kernel_launch};
  char names[N_TESTS][50] = {"test_vector_add_cpu",
                             "test_vector_sub_cpu",
                             "test_vector_element_wise_prod_cpu",
                             "test_vector_dot_product_cpu",
                             "test_vector_l2_norm",
                             "test_vector_lp_norm",
                             "test_vector_max_norm",
                             "test_vector_equals",
                             "test_matrix_add_cpu",
                             "test_matrix_sub_cpu",
                             "test_matrix_mult_cpu",
                             "test_matrix_mult_scalar_cpu",
                             "test_matrix_trace",
                             "test_matrix_lpq_norm",
                             "test_frobenius_norm",
                             "test_cuda_devices",
                             "test_vector_move_cuda_cpu",
                             "test_vector_add_cuda",
                             "test_vector_sub_cuda",
                             "test_vector_element_wise_prod_cuda",
                             "test_matrix_move_cuda_cpu",
                             "test_matrix_add_cuda",
                             "test_matrix_sub_cuda",
                             "test_matrix_mult_cuda",
                             "test_vector_copy",
                             "test_user_kernel_launch"};

  printf(PREFIX "Tests started...\n");
  for (int i = 0; i < N_TESTS; i++) {
    printf(PREFIX "Ran test #%d \"%s\": %s\n", i + 1, names[i],
           test_fn[i]() == SUCCESS ? GRN "SUCCESS" RESET : RED "FAILED" RESET);
  }

  printf(PREFIX "Total tests: %d | Tests passed: %d | Tests failed: "
                "%d\n",
         _TESTS_RUN, _TESTS_PASSED, _TESTS_RUN - _TESTS_PASSED);

  // Clear devices
  clear_devices();

  return (_TESTS_PASSED == _TESTS_RUN) ? 0 : -1;
}
