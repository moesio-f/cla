#ifndef CLA_CUDA
#define CLA_CUDA
#include <stdbool.h>
#include "entities.h"

// Global variables to be initialized
extern CUDADevice* DEVICES;

// Global device management functions
void popullate_devices();
CUDADevice *get_device_by_name(char *name);

// Vector device management
Vector *vector_to_cu(Vector *src, CUDADevice *device, CopyStrategy strategy);
Vector *vector_to_cpu(Vector *src, CUDADevice *device, CopyStrategy strategy);

// Matrix device management
Matrix *matrix_to_cu(Matrix *src, CUDADevice *device, CopyStrategy strategy);
Matrix *matrix_to_cpu(Matrix *src, CUDADevice *device, CopyStrategy strategy);

#endif
