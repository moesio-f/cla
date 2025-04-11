/* C API test suite.
 *
 * This file contains multiple tests for the
 *  C API.
 *
 * */
#include "../../cla/include/entities.h"
#include "../../cla/include/matrix_operations.h"
#include "../../cla/include/matrix_utils.h"
#include "../../cla/include/vector_operations.h"
#include "../../cla/include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#define N_TESTS 10
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"
#define PREFIX BLU "[CLA][Test Suite] " RESET

// Utility enumeration for custom error codes
typedef enum { SUCCESS, FAILED } RETURN_CODE;

// Global variables for test suite
int _TESTS_RUN = 0;
int _TESTS_PASSED = 0;

RETURN_CODE _test_vector_binop(Vector *(*operation)(Vector *, Vector *,
                                                    Vector *),
                               int dims, double *a_value, double *b_value,
                               double *target_value, double tol) {
  _TESTS_RUN++;
  RETURN_CODE code = SUCCESS;

  // Vector allocation
  Vector a = {a_value, dims, NULL};
  Vector b = {b_value, dims, NULL};

  // Operation
  Vector *result = operation(&a, &b, NULL);

  // Validation
  for (int i = 0; i < dims; i++) {
    if (fabs(result->arr[i] - target_value[i]) > tol) {
      code = FAILED;
      break;
    }
  }

  if (code == SUCCESS) {
    _TESTS_PASSED++;
  }

  // Clean-up
  destroy_vector(result);

  return code;
}

RETURN_CODE
_test_matrix_binop(Matrix *(*operation)(Matrix *, Matrix *, Matrix *), int rows,
                   int columns, double **a_value, double **b_value,
                   double **target_value, double tol) {
  _TESTS_RUN++;
  RETURN_CODE code = SUCCESS;

  // Matrix allocation
  Matrix a = {a_value, rows, columns, NULL};
  Matrix b = {b_value, rows, columns, NULL};

  // Operation
  Matrix *result = operation(&a, &b, NULL);

  // Validation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (fabs(result->arr[i][j] - target_value[i][j]) > tol) {
        code = FAILED;
        break;
      }
    }
  }

  if (code == SUCCESS) {
    _TESTS_PASSED++;
  }

  // Clean-up dynamically allocated matrices
  destroy_matrix(result);

  return code;
}

RETURN_CODE test_vector_add_cpu() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {2.0, 1.5};

  return _test_vector_binop(&vector_add, 2, a, b, target, 0.001);
}

RETURN_CODE test_vector_sub_cpu() {
  double a[2] = {1.0, 0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {0.0, -0.5};

  return _test_vector_binop(&vector_sub, 2, a, b, target, 0.001);
}

RETURN_CODE test_vector_element_wise_prod_cpu() {
  double a[2] = {3.0, -0.5};
  double b[2] = {1.0, 1.0};
  double target[2] = {3.0, -0.5};

  return _test_vector_binop(&vector_element_wise_prod, 2, a, b, target, 0.001);
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

RETURN_CODE test_matrix_add_cpu() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {4.0, 0.5}, t_2[2] = {-4.0, 9.0};

  return _test_matrix_binop(&matrix_add, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            0.001);
}

RETURN_CODE test_matrix_sub_cpu() {
  double a_1[2] = {3.0, -0.5}, a_2[2] = {-5.0, 8.0};
  double b_1[2] = {1.0, 1.0}, b_2[2] = {1.0, 1.0};
  double t_1[2] = {2.0, -1.5}, t_2[2] = {-6.0, 7.0};

  return _test_matrix_binop(&matrix_sub, 2, 2, (double *[2]){a_1, a_2},
                            (double *[2]){b_1, b_2}, (double *[2]){t_1, t_2},
                            0.001);
}

int main() {
  RETURN_CODE (*test_fn[N_TESTS])(void) = {&test_vector_add_cpu,
                                           &test_vector_sub_cpu,
                                           &test_vector_element_wise_prod_cpu,
                                           &test_vector_dot_product_cpu,
                                           &test_vector_l2_norm,
                                           &test_vector_lp_norm,
                                           &test_vector_max_norm,
                                           &test_vector_equals,
                                           &test_matrix_add_cpu,
                                           &test_matrix_sub_cpu};
  char names[N_TESTS][50] = {"test_vector_add_cpu",
                             "test_vector_sub_cpu",
                             "test_vector_element_wise_prod_cpu",
                             "test_vector_dot_product_cpu",
                             "test_vector_l2_norm",
                             "test_vector_lp_norm",
                             "test_vector_max_norm",
                             "test_vector_equals",
                             "test_matrix_add_cpu",
                             "test_matrix_sub_cpu"};

  printf(PREFIX "Tests started...\n");
  for (int i = 0; i < N_TESTS; i++) {
    printf(PREFIX "Ran test \"%s\": %s\n", names[i],
           test_fn[i]() == SUCCESS ? GRN "SUCCESS" RESET : RED "FAILED" RESET);
  }

  printf(PREFIX "Total tests: %d | Tests passed: %d | Tests failed: "
                "%d\n",
         _TESTS_RUN, _TESTS_PASSED, _TESTS_RUN - _TESTS_PASSED);
  return (_TESTS_PASSED == _TESTS_RUN) ? 0 : -1;
}
