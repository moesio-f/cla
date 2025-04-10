/* C API test suite.
 *
 * This file contains multiple tests for the
 *  C API.
 *
 * */
#include "../../cla/include/entities.h"
#include "../../cla/include/vector_operations.h"
#include "../../cla/include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// Utility enumeration for custom error codes
typedef enum { SUCCESS, FAILED } RETURN_CODE;

// Global variables for test suite
int _TESTS_RUN = 0;
int _TESTS_PASSED = 0;

RETURN_CODE _test_vector_binop(Vector *(*operation)(Vector *, Vector *,
                                                    Vector *),
                               int dims, double a_value, double b_value,
                               double target_value, double tol) {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  // Vector allocation
  Vector *a = create_vector(1, NULL, a_value);
  Vector *b = create_vector(1, NULL, b_value);

  // Operation
  Vector *result = operation(a, b, NULL);

  // Validation
  if (fabs(result->arr[0] - target_value) <= tol) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  // Clean-up
  destroy_vector(a);
  destroy_vector(b);
  destroy_vector(result);

  return code;
}

RETURN_CODE test_vector_add_cpu() {
  return _test_vector_binop(&vector_add, 10, 1.0, 1.0, 2.0, 0.001);
}

RETURN_CODE test_vector_sub_cpu() {
  return _test_vector_binop(&vector_sub, 10, 0.0, 1.0, -1.0, 0.001);
}

RETURN_CODE test_vector_element_wise_prod_cpu() {
  return _test_vector_binop(&vector_element_wise_prod, 10, 2.0, 3.0, 6.0,
                            0.001);
}

RETURN_CODE test_vector_dot_product_cpu() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector *a = const_vector(2, 2.0, NULL);

  if (fabs(vector_dot_product(a, a) - 8.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  destroy_vector(a);
  return code;
}

RETURN_CODE test_vector_l2_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector *a = const_vector(2, 2.0, NULL);
  if (fabs(vector_l2_norm(a) - sqrt(8.0)) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  destroy_vector(a);
  return code;
}

RETURN_CODE test_vector_lp_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector *a = const_vector(2, 3.0, NULL);

  if (fabs(vector_lp_norm(a, 1.0) - 6.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  destroy_vector(a);
  return code;
}

RETURN_CODE test_vector_max_norm() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;
  Vector *a = create_vector(4, NULL, 0.0, -1.0, 3.0, 10.0);

  if (fabs(vector_max_norm(a) - 10.0) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  destroy_vector(a);
  return code;
}

int main() {
  test_vector_add_cpu();
  test_vector_sub_cpu();
  test_vector_element_wise_prod_cpu();
  test_vector_dot_product_cpu();
  test_vector_l2_norm();
  test_vector_lp_norm();
  test_vector_max_norm();
  printf("Total tests: %d | Tests passed: %d | Tests failed: %d\n", _TESTS_RUN,
         _TESTS_PASSED, _TESTS_RUN - _TESTS_PASSED);
  return (_TESTS_PASSED == _TESTS_RUN) ? 0 : -1;
}
