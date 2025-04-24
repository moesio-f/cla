extern "C" {
#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/vector_utils.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
}

#include "cuda_runtime_api.h"

extern "C" void synchronize_devices() { cudaDeviceSynchronize(); }

extern "C" void populate_devices() {
  if (DEVICES != NULL) {
    return;
  }

  // Initialize global variable
  DEVICES = (AvailableCUDADevices *)malloc(sizeof(AvailableCUDADevices));
  DEVICES->devices = NULL;
  DEVICES->count = 0;

  // Try to get CUDA devices count
  cudaDeviceProp prop;
  cudaGetDeviceCount(&DEVICES->count);

  // If it has devices, get their metadata
  if (DEVICES->count > 0) {
    DEVICES->devices =
        (CUDADevice *)malloc(sizeof(CUDADevice) * DEVICES->count);
    for (int i = 0; i < DEVICES->count; i++) {
      cudaGetDeviceProperties(&prop, i);
      char *name = (char *)malloc(sizeof(char) * 256);
      strcpy(name, prop.name);
      DEVICES->devices[i] =
          (CUDADevice){i,
                       name,
                       prop.maxGridSize[0],
                       prop.maxGridSize[1],
                       prop.maxGridSize[2],
                       prop.maxThreadsDim[0],
                       prop.maxThreadsDim[1],
                       prop.maxThreadsDim[2],
                       prop.maxThreadsPerBlock,
                       (CUDAKernelLaunchParameters){0, 0, 0, 0, 0, 0}};
    }
  }
}

extern "C" void clear_devices() {
  // If no devices, return
  if (DEVICES == NULL) {
    return;
  }

  // If it has devices, deallocate memory
  //    for each one
  if (DEVICES->count > 0) {
    for (int i = 0; i < DEVICES->count; i++) {
      free(DEVICES->devices[i].name);
    }
    free(DEVICES->devices);
  }

  // Deallocate memory for global variable
  free(DEVICES);
  DEVICES = NULL;
}

extern "C" Vector *vector_to_cu(Vector *src, CUDADevice *device) {
  if (src->device != NULL) {
    // Maybe it's already on correct device?
    if (src->device == device) {
      return src;
    }

    // It's on another device!
    // Move data back to CPU before proceeding
    // Better handling requires CUDA Driver API
    vector_to_cpu(src);
  }

  // src is on CPU, activate target device
  cudaSetDevice(device->id);

  // Allocate memory for a vector in GPU
  size_t vec_size = sizeof(Vector);
  Vector *cu_vector = NULL;
  cudaMalloc(&cu_vector, vec_size);

  // Allocate memory for underlying array in GPU
  size_t arr_size = src->dims * sizeof(double);
  double *cu_arr = NULL;
  cudaMalloc(&cu_arr, arr_size);

  // Copy array from CPU to GPU
  cudaMemcpy(cu_arr, src->arr, arr_size, cudaMemcpyHostToDevice);

  // Clear CPU array
  free(src->arr);

  // Copy vector from CPU to GPU in-place
  src->arr = cu_arr;
  src->cu_vector = NULL;
  src->device = NULL;
  cudaMemcpy(cu_vector, src, vec_size, cudaMemcpyHostToDevice);

  // Update CPU vector metadata
  src->device = device;
  src->arr = NULL;
  src->cu_vector = cu_vector;

  // Return CPU vector metadata
  return src;
}

extern "C" Vector *vector_to_cpu(Vector *src) {
  // Maybe device is already on CPU
  if (src->device == NULL) {
    return src;
  }

  // Guarantee CUDA uses the correct device
  cudaSetDevice(src->device->id);

  // Preconditions
  assert(src->arr == NULL);

  // Allocate memory for underlying array in CPU
  size_t arr_size = src->dims * sizeof(double);
  double *vec_arr = (double *)malloc(arr_size);

  // Copy Vector from GPU to CPU in-place
  // We need to store the pointer to cu_vector
  //    since it will be override by the contents
  //    of cu_vector.
  Vector *cu_vector = src->cu_vector;
  cudaMemcpy(src, cu_vector, sizeof(Vector), cudaMemcpyDeviceToHost);

  // Copy the data from the cu_vector->arr to a CPU array
  double *cu_arr = src->arr;
  cudaMemcpy(vec_arr, src->arr, arr_size, cudaMemcpyDeviceToHost);

  // Clear GPU array
  cudaFree(cu_arr);

  // Clear GPU vector
  cudaFree(cu_vector);

  // Update CPU vector metadata
  src->arr = vec_arr;
  src->cu_vector = NULL;
  src->device = NULL;

  return src;
}

extern "C" Matrix *matrix_to_cu(Matrix *src, CUDADevice *device) {
  if (src->device != NULL) {
    // Maybe it's already on correct device?
    if (src->device == device) {
      return src;
    }

    // It's on another device!
    // Move data back to CPU before proceeding
    // Better handling requires CUDA Driver API
    matrix_to_cpu(src);
  }

  // src is on CPU, activate target device
  cudaSetDevice(device->id);

  // 1. Allocate memory in GPU for the matrix
  size_t mat_size = sizeof(Matrix);
  Matrix *cu_matrix = NULL;
  cudaMalloc(&cu_matrix, mat_size);

  // 2. Construct underlying array (i.e., a collection
  //    of n_rows pointers).
  size_t arr_size = src->rows * sizeof(double *);
  double **cu_arr = NULL;
  cudaMalloc(&cu_arr, arr_size);

  // 3. Construct row arrays and copy them to
  //    cu_array.
  // tmp is needed to set each index of the
  //    the cu_array on CPU prior to copying
  //    it to GPU.
  size_t row_size = src->columns * sizeof(double);
  double **tmp = (double **)malloc(arr_size);
  for (int i = 0; i < src->rows; i++) {
    // Allocate memory on GPU for row
    double *cu_row = NULL;
    cudaMalloc(&cu_row, row_size);

    // Copy the actual contents from src to
    //  the GPU row
    cudaMemcpy(cu_row, src->arr[i], row_size, cudaMemcpyHostToDevice);

    // Update the tmp with the address of this new
    //  row
    tmp[i] = cu_row;

    // Free the CPU memory
    free(src->arr[i]);
  }

  // Copy the tmp (array of GPU arrays addresses) to the
  //    GPU array
  cudaMemcpy(cu_arr, tmp, arr_size, cudaMemcpyHostToDevice);

  // Free CPU memory
  free(tmp);

  // 4. Now, we have the cu_array complete,
  //    all we need is to send a Matrix struct
  //    to GPU.
  free(src->arr);
  src->arr = cu_arr;
  src->cu_matrix = NULL;
  src->device = NULL;
  cudaMemcpy(cu_matrix, src, mat_size, cudaMemcpyHostToDevice);

  // Update CPU vector metadata
  src->device = device;
  src->arr = NULL;
  src->cu_matrix = cu_matrix;

  // Return CPU vector metadata
  return src;
}

extern "C" Matrix *matrix_to_cpu(Matrix *src) {
  // Maybe device is already on CPU
  if (src->device == NULL) {
    return src;
  }

  // Preconditions
  assert(src->arr == NULL);

  // Guarantee CUDA uses the correct device
  cudaSetDevice(src->device->id);

  // 1. Bring back the Matrix from GPU
  size_t mat_size = sizeof(Matrix);
  Matrix *cu_matrix = src->cu_matrix;
  cudaMemcpy(src, cu_matrix, mat_size, cudaMemcpyDeviceToHost);
  cudaFree(cu_matrix);

  // 2. Reconstruct array in CPU
  size_t arr_size = src->rows * sizeof(double *);
  size_t row_size = src->columns * sizeof(double);
  double **cu_arr = src->arr;
  src->arr = (double **)malloc(arr_size);
  cudaMemcpy(src->arr, cu_arr, arr_size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < src->rows; i++) {
    double *row = (double *)malloc(row_size);
    double *cu_row = src->arr[i];
    cudaMemcpy(row, cu_row, row_size, cudaMemcpyDeviceToHost);
    src->arr[i] = row;
    cudaFree(cu_row);
  }
  cudaFree(cu_arr);

  // Update CPU vector metadata
  src->cu_matrix = NULL;
  src->device = NULL;

  return src;
}
