extern "C" {
#include "../cla/include/device_management.h"
#include "../cla/include/entities.h"
#include "../cla/include/matrix_operations.h"
#include "../cla/include/matrix_utils.h"
#include "../cla/include/vector_operations.h"
#include "../cla/include/vector_utils.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

#define N_THREADS 512
#define N_VECTOR_OPERATORS 3
#define N_MATRIX_OPERATORS 3

char KIND_VEC[256] = "Vector";
Vector *(*VECTOR_OPERATORS[N_VECTOR_OPERATORS])(Vector *, Vector *,
                                                Vector *) = {
    &vector_add, &vector_sub, &vector_element_wise_prod};
char VECTOR_OPERATOR_NAMES[N_VECTOR_OPERATORS][256] = {
    "cla_gpu_add", "cla_gpu_sub", "cla_gpu_element_wise_prod"};
char KIND_MAT[256] = "Matrix";
Matrix *(*MATRIX_OPERATORS[N_MATRIX_OPERATORS])(Matrix *, Matrix *,
                                                Matrix *) = {
    &matrix_add, &matrix_sub, &matrix_mult};
char MATRIX_OPERATOR_NAMES[N_MATRIX_OPERATORS][256] = {
    "cla_gpu_add", "cla_gpu_sub", "cla_gpu_matmul"};

int imax(int a, int b) { return (a >= b) ? a : b; }

void print_help() {
  printf("usage: cla_cuda_benchmark [-h]"
         " n_runs dim_start dim_end dim_step kind\n\n");
  printf("Run benchmark tests for GPU-supported "
         "operations with vectors and matrices. Every "
         "operation is run `n_runs` times for each dimension "
         "in the interval [`dim_start`, `dim_end`].\n\n");
  printf("positional arguments:\n");
  printf("\tn_runs\t\tNumber of executions for each operation.\n");
  printf("\tdim_start\tInitial dimension for tests (for matrix, "
         "rows=columns).\n");
  printf("\tdim_end\t\tEnd dimension for tests (inclusive).\n");
  printf("\tdim_step\tSteps to generate each dimension.\n");
  printf("\tkind\t\tWhether to run benchmark for `vector`, "
         "`matrix` or `both`.\n\n");
  printf("options:\n");
  printf("\t-h\t\tShows this help message and exit.\n");
}

void write_headers(FILE *output) {
  fprintf(output, "name,kind,dims,run,time\n");
  fflush(output);
}

void write_line(char *name, char *kind, int dims, int run,
                double execution_time, FILE *output) {
  fprintf(output, "%s,%s,%d,%d,%.10f\n", name, kind, dims, run, execution_time);
  fflush(output);
}

void run_vector_benchmark(int n_runs, int dim_start, int dim_end, int dim_step,
                          CUDADevice *device, FILE *output) {
  // Allocate CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate execution_time
  float execution_time_ms = 0.0f;

  for (int dim = dim_start; dim <= dim_end; dim += dim_step) {
    // Allocate vectors
    Vector *src = const_vector(dim, 1.0, device);
    Vector *dst = const_vector(dim, 0.0, device);

    // Select appropriate kernel launch parameters
    int n_threads = N_THREADS;
    int n_blocks =
        imax(1, 1 + (int)ceil((dim - n_threads) / (double)n_threads));
    assert(n_threads * n_blocks >= dim);
    device->params.n_threads_x = n_threads;
    device->params.n_blocks_x = n_blocks;

    // Run operations
    for (int run = 0; run < n_runs; run++) {
      for (int op_idx = 0; op_idx < N_VECTOR_OPERATORS; op_idx++) {
        // Obtain target operation
        Vector *(*fn)(Vector *, Vector *, Vector *) = VECTOR_OPERATORS[op_idx];
        char *name = VECTOR_OPERATOR_NAMES[op_idx];

        // Obtain current GPU time
        cudaEventRecord(start);

        // Apply operation
        fn(src, src, dst);

        // Obtain end GPU time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&execution_time_ms, start, stop);

        // Write results
        write_line(name, KIND_VEC, dim, run, execution_time_ms / 1000.0,
                   output);
      }
    }

    // Release vectors
    destroy_vector(src);
    destroy_vector(dst);
  }
}

void run_matrix_benchmark(int n_runs, int dim_start, int dim_end, int dim_step,
                          CUDADevice *device, FILE *output) {

  // Allocate CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate execution_time
  float execution_time_ms = 0.0f;

  for (int dim = dim_start; dim <= dim_end; dim += dim_step) {
    // Allocate matrices
    Matrix *src = const_matrix(dim, dim, 1.0, device);
    Matrix *dst = const_matrix(dim, dim, 0.0, device);

    // Select appropriate kernel launch parameters
    int n_threads = (int)ceil(sqrt(N_THREADS));
    int n_blocks =
        imax(1, 1 + (int)ceil((dim - n_threads) / (double)n_threads));
    assert(n_threads * n_blocks >= dim);
    device->params.n_threads_x = n_threads;
    device->params.n_blocks_x = n_blocks;
    device->params.n_threads_y = n_threads;
    device->params.n_blocks_y = n_blocks;

    // Run operations
    for (int run = 0; run < n_runs; run++) {
      for (int op_idx = 0; op_idx < N_MATRIX_OPERATORS; op_idx++) {
        // Obtain target operation
        Matrix *(*fn)(Matrix *, Matrix *, Matrix *) = MATRIX_OPERATORS[op_idx];
        char *name = MATRIX_OPERATOR_NAMES[op_idx];

        // Obtain current GPU time
        cudaEventRecord(start);

        // Apply operation
        fn(src, src, dst);

        // Obtain end GPU time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&execution_time_ms, start, stop);

        // Write results
        write_line(name, KIND_MAT, dim, run, execution_time_ms / 1000.0,
                   output);
      }
    }

    // Release matrices
    destroy_matrix(src);
    destroy_matrix(dst);
  }
}

int main(int argc, char **argv) {
  // Check if should only print help
  assert(argc > 1);
  if (strcmp(argv[1], "-h") == 0) {
    print_help();
    return 0;
  }

  // Otherwise, should run benchmark with
  //    arguments
  if (argc != 6) {
    printf("!!!!! Wrong arguments !!!!!\n\n");
    print_help();
    return 0;
  }

  if (!has_cuda()) {
    printf("Host has no CUDA device available.\n");
    return -1;
  }

  // Get and default device
  CUDADevice *device = get_device_by_id(0);
  cudaSetDevice(0);

  // Parse arguments
  int n_runs = atoi(argv[1]), dim_start = atoi(argv[2]),
      dim_end = atoi(argv[3]), dim_step = atoi(argv[4]);
  bool run_vector = strcmp(argv[5], "vector") == 0;
  bool run_matrix = strcmp(argv[5], "matrix") == 0;
  bool run_both = strcmp(argv[5], "both") == 0;

  // Validate arguments
  assert(n_runs > 0);
  assert(dim_start <= dim_end);
  assert(dim_start > 0);
  assert(dim_step >= 1);
  assert(run_vector || run_matrix || run_both);

  // Create output file
  FILE *output = fopen("cla_cuda_benchmark.csv", "w+");
  write_headers(output);

  // Run Vector benchmark
  if (run_vector || run_both) {
    run_vector_benchmark(n_runs, dim_start, dim_end, dim_step, device, output);
  }

  // Run Matrix benchmark
  if (run_matrix || run_both) {
    run_matrix_benchmark(n_runs, dim_start, dim_end, dim_step, device, output);
  }

  // Close file
  fclose(output);
  return 0;
}
