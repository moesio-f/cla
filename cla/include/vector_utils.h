#ifndef CLA_VECTOR_UTILS
#define CLA_VECTOR_UTILS
#include "entities.h"
#include <stdbool.h>

// Maybe allocate a vector if ptr is NULL.
Vector *maybe_alloc_vector(Vector *ptr, int dims, CUDADevice *device);

// Vector constructor
Vector *const_vector(int dims, double value, CUDADevice *device);

// Vector from literals
Vector *create_vector(int dims, CUDADevice *device, ...);

// Clean up vector from either CUDA or CPU
void destroy_vector(Vector *vector);

// Copy vector from src to dst (if NULL, automatically allocates memory)
Vector *copy_vector(Vector *src, Vector *dst);

// Print vector to stdout
void print_vector(Vector *a, char *suffix);

// Templates for vector operations
Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *), bool (*validate)(int, ...),
    Vector *a, Vector *b, Vector *dst, int alloc_dims,
    CUDADevice *alloc_device);

// Validation functions. Accept a variable number
//  of Vector* arguments.
bool vec_same_dims_same_devices(int n, ...);

#endif
