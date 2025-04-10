#ifndef CLA_CUDA_UTILS
#define CLA_CUDA_UTILS
#include "entities.h"
#include <stdbool.h>

// Templates for vector operations
Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *), bool (*validate)(int, ...),
    Vector *a, Vector *b, Vector *dst, int alloc_dims,
    CUDADevice *alloc_device);

#endif
