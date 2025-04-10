#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

bool vector_equals(Vector *a, Vector *b) {
  assert(a->dims == b->dims);

  for (int i = 0; i < a->dims; i++) {
    if (fabs(a->arr[i] - b->arr[i]) > 0.001) {
      return false;
    }
  }

  return true;
}
