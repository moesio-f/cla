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
#include <stdlib.h>
#include <string.h>

bool arg_is_gpu(char *arg) { return strcmp(arg, "GPU") == 0; }
bool arg_is_cpu(char *arg) { return strcmp(arg, "CPU") == 0; }

int main(int argc, char **argv) {
  // <program_name> <vector/matrix> <CPU/GPU>
  assert(argc == 3);
  assert((has_cuda() && arg_is_gpu(argv[2])) || arg_is_cpu(argv[2]));
  int n_alloc = 100;
  int n_alloc_dims = 10000;
  int n_alloc_rows = 100, n_alloc_columns = 100;
  CUDADevice *device = arg_is_gpu(argv[2]) ? get_device_by_id(0) : NULL;

  printf(PREFIX "%s %s memory leak tests started (run with valgrind).\n",
         argv[2], argv[1]);
  if (strcmp(argv[1], "vector") == 0) {
    // Allocation
    printf(PREFIX "Allocating %d vectors with %d dims...\n", n_alloc,
           n_alloc_dims);
    Vector **vectors = (Vector **)malloc(n_alloc * sizeof(Vector *));
    for (int i = 0; i < n_alloc; i++) {
      vectors[i] = const_vector(n_alloc_dims, 0.0, device);
    }

    // Clean-up
    printf(PREFIX "Cleaning vectors...\n");
    for (int i = 0; i < n_alloc; i++) {
      destroy_vector(vectors[i]);
    }
    free(vectors);
  } else if (strcmp(argv[1], "matrix") == 0) {
    // Allocation
    printf(PREFIX "Allocating %d matrix with (%d, %d) dims...\n", n_alloc,
           n_alloc_rows, n_alloc_columns);
    Matrix **matrices = (Matrix **)malloc(n_alloc * sizeof(Matrix *));
    for (int i = 0; i < n_alloc; i++) {
      matrices[i] = const_matrix(n_alloc_rows, n_alloc_columns, 0.0, device);
    }

    // Clean-up
    printf(PREFIX "Cleaning matrices...\n");
    for (int i = 0; i < n_alloc; i++) {
      destroy_matrix(matrices[i]);
    }
    free(matrices);
  } else {
    printf(PREFIX RED "Unknown argument.\n" RESET);
    return -1;
  }

  // Clear any leftover memory for DEVICES
  clear_devices();
  printf(PREFIX "Check valgrind results.\n");
  return 0;
}
