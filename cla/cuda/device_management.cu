extern "C" {
#include "../include/device_management.h"
#include "../include/entities.h"
#include "../include/vector_utils.h"
#include <stdlib.h>
#include <string.h>
}

#include "cuda_runtime_api.h"

extern "C" void populate_devices() {
  if (DEVICES != NULL) {
    return;
  }

  // Initialize global variable
  DEVICES = (AvailableCUDADevices *)malloc(sizeof(AvailableCUDADevices));
  DEVICES->devices = NULL;
  DEVICES->count = 0;

  // Try to get CUDA devices count
  cudaGetDeviceCount(&DEVICES->count);

  // If it has devices, get their metadata
  if (DEVICES->count > 0) {
    DEVICES->devices =
        (CUDADevice *)malloc(sizeof(CUDADevice) * DEVICES->count);
    for (int i = 0; i < DEVICES->count; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      char *name = (char *)malloc(sizeof(char) * 256);
      strcpy(name, prop.name);
      DEVICES->devices[i] = (CUDADevice){i,
                                         name,
                                         prop.maxGridSize[0],
                                         prop.maxGridSize[1],
                                         prop.maxGridSize[2],
                                         prop.maxThreadsDim[0],
                                         prop.maxThreadsDim[1],
                                         prop.maxThreadsDim[2],
                                         prop.maxThreadsPerBlock};
    }
  }
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

  // Copy target vector to GPU by constructing a dummy one
  //    in CPU
  Vector *dummy = const_vector(src->dims, 0.0, NULL);
  double *dummy_arr = dummy->arr;
  dummy->arr = cu_arr;
  cudaMemcpy(cu_vector, dummy, vec_size, cudaMemcpyHostToDevice);
  dummy->arr = dummy_arr;
  destroy_vector(dummy);

  // Clear CPU array
  free(src->arr);

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

  // Allocate memory for underlying array in CPU
  size_t arr_size = src->dims * sizeof(double);
  double *vec_arr = (double *)malloc(arr_size);

  // Copy Vector from GPU to CPU, then copy array to CPU
  Vector *dummy = const_vector(src->dims, 0.0, NULL);
  cudaMemcpy(dummy, src->cu_vector, sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(vec_arr, dummy->arr, arr_size, cudaMemcpyDeviceToHost);

  // Clear GPU array and Vector
  cudaFree(dummy->arr);
  cudaFree(src->cu_vector);
  free(dummy);

  // Update CPU vector metadata
  src->arr = vec_arr;
  src->cu_vector = NULL;
  src->device = NULL;

  return src;
}
