#include "cla/include/device_management.h"
#include "cla/include/entities.h"
#include "cla/include/matrix_operations.h"
#include "cla/include/matrix_utils.h"
#include "cla/include/vector_operations.h"
#include "cla/include/vector_utils.h"
#include <stdio.h>
#include <stdlib.h>

void print_vector_result(char *title, Vector *a, Vector *b, Vector *dst,
                         char operation) {
  printf("%s\n", title);
  print_vector(a, " ");
  printf("%c ", operation);
  print_vector(b, " ");
  printf("= ");
  print_vector(dst, "\n\n");
}

void print_matrix_result(char *title, Matrix *a, Matrix *b, Matrix *dst,
                         char operation) {
  printf("%s", title);
  print_matrix(a, "\n");
  printf("%c", operation);
  print_matrix(b, "\n");
  printf("=");
  print_matrix(dst, "\n\n");
}

int main() {
  /** Some vector operations. */
  // We can instantiate vectors directly
  Vector vec_a = {(double[2]){1.0, 2.0}, 2, NULL, NULL};

  // ...or using constructors
  Vector *vec_b = create_vector(2, NULL, -1.0, 0.5);
  Vector *vec_dst = const_vector(2, 0.0, NULL);

  // Addition
  vector_add(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Addition", &vec_a, vec_b, vec_dst, '+');

  // Subtraction
  vector_sub(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Subtraction", &vec_a, vec_b, vec_dst, '-');

  // Element-wise product
  vector_element_wise_prod(&vec_a, vec_b, vec_dst);
  print_vector_result("Vector Element wise product", &vec_a, vec_b, vec_dst,
                      '*');

  // For vectors created by constructors, we can use the
  //    utility functions to clean-up their memory.
  destroy_vector(vec_b);
  destroy_vector(vec_dst);

  /** Some matrix operations. */
  // Matrix construction is a little more cumbersome
  //    without constructors. The test suite contains
  //    some examples.
  Matrix *mat_a = const_matrix(2, 2, 1.0, NULL);
  Matrix *mat_b = const_matrix(2, 2, 0.5, NULL);
  Matrix *mat_dst = const_matrix(2, 2, 0.0, NULL);

  // We can access the underlying data store
  //    and make changes
  mat_a->arr[0][0] = 2.0;
  mat_b->arr[1][1] = -3.0;

  // To run operations in GPU, we simply have to
  //    send the target matrices/vectors to GPU.
  if (has_cuda()) {
    // Devices can be selected by ID or name.
    CUDADevice *device = get_device_by_id(0);
    char dev_info[512];
    printf("The following device was selected:\n%s\n\n",
           device_to_str(device, dev_info));

    // This function expects to handle matrix/vectors
    //  created by the constructors. Otherwise,
    //  the behavior is undefined.
    matrix_to_cu(mat_a, device);
    matrix_to_cu(mat_b, device);
    matrix_to_cu(mat_dst, device);
  }

  // Addition
  matrix_add(mat_a, mat_b, mat_dst);
  print_matrix_result("Matrix Addition", mat_a, mat_b, mat_dst, '+');

  // Matrix multiplication
  matrix_mult(mat_a, mat_b, mat_dst);
  print_matrix_result("Matrix Multiplication", mat_a, mat_b, mat_dst, '*');

  // Clean-up
  destroy_matrix(mat_a);
  destroy_matrix(mat_b);
  destroy_matrix(mat_dst);

  return 0;
}
