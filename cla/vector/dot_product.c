#include "../include/cuda_utils.h"
#include "../include/entities.h"
#include "../include/vector_operations.h"
#include "../include/vector_utils.h"
#include <stdlib.h>

double vector_dot_product(Vector *a, Vector *b) {
  double dot_product = 0.0;

  // Obtain temporary vector with element wise product
  Vector *result = vector_element_wise_prod(a, b, NULL);

  // Move result back to CPU
  // ...

  // Reduce sum
  for (int i = 0; i < result->dims; dot_product += result->arr[i++])
    ;

  // Clean up memory
  destroy_vector(result);
  return dot_product;
}
