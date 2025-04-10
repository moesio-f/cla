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

RETURN_CODE test_vector_add_cpu() {
  _TESTS_RUN++;
  RETURN_CODE code = FAILED;

  // Vector allocation
  Vector *a = create_vector(1, NULL, 1.0);
  Vector *b = create_vector(1, NULL, -1.0);

  // Operation
  Vector *result = vector_add(a, b, NULL);

  // Validation
  if (fabs(result->arr[0]) <= 0.001) {
    code = SUCCESS;
    _TESTS_PASSED++;
  }

  // Clean-up
  destroy_vector(a);
  destroy_vector(b);
  destroy_vector(result);

  return code;
}

int main() {
  test_vector_add_cpu();
  printf("Total tests: %d | Tests passed: %d | Tests failed: %d\n", _TESTS_RUN,
         _TESTS_PASSED, _TESTS_RUN - _TESTS_PASSED);
  return (_TESTS_PASSED == _TESTS_RUN) ? 0 : -1;
}
