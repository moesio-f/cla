#include "../../cla/include/device_management.h"
#include "../../cla/include/entities.h"
#include "../../cla/include/matrix_operations.h"
#include "../../cla/include/matrix_utils.h"
#include "../../cla/include/vector_operations.h"
#include "../../cla/include/vector_utils.h"
#include "colors.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  int n = 100, dims = 10000, rows = 100, columns = 100;
  char answer;
  bool run_next = false;
  assert(has_cuda());
  CUDADevice *device = get_device_by_id(0);
  printf(PREFIX "Watch system memory usage"
                " (e.g., `nvidia-smi`, `pmap`, `htop`)\n");

  printf(PREFIX "Starting vector memory stability...\n");
  printf(PREFIX "Test parameters: n_allocs=%d, dims=%d\n", n, dims);
  do {
    printf(MAG "Running...\n" RESET);
    for (int i = 0; i < n; i++) {
      destroy_vector(const_vector(dims, -8492.4, NULL));
      destroy_vector(const_vector(dims, 42879.2, device));
    }

    printf(MAG "End vector testing? (y/n/Y/N): " RESET);
    scanf("%c", &answer);
    run_next = toupper(answer) == 'Y';
  } while (!run_next);

  printf("\n");
  printf(PREFIX "Starting matrix memory stability...\n");
  printf(PREFIX "Test parameters: n_allocs=%d, rows=%d, columns=%d\n", n, rows,
         columns);
  do {
    printf(MAG "Running...\n" RESET);
    for (int i = 0; i < n; i++) {
      destroy_matrix(const_matrix(rows, columns, -8492.4, NULL));
      destroy_matrix(const_matrix(rows, columns, 42879.2, device));
    }

    printf(MAG "End matrix testing? (y/n/Y/N): " RESET);
    scanf("%c", &answer);
    run_next = toupper(answer) == 'Y';
  } while (!run_next);
}
