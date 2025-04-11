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

// Templates for matrix operations
Matrix *cpu_gpu_conditional_apply_matrix_operator(
    void (*cpu_op)(Matrix *, Matrix *, Matrix *),
    void (*gpu_op)(Matrix *, Matrix *, Matrix *), bool (*validate)(int, ...),
    Matrix *a, Matrix *b, Matrix *dst, int alloc_rows, int alloc_colums,
    CUDADevice *alloc_device);

#endif
