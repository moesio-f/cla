#ifndef CLA_CUDA_UTILS
#define CLA_CUDA_UTILS
#include "entities.h"
#include <stdbool.h>

// Templates for vector operations
Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *),
    bool (*validate)(Vector *, Vector *, Vector *), Vector *a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device);

Vector *cpu_gpu_conditional_apply_scalar_vector_operator(
    void (*cpu_op)(double *, Vector *, Vector *),
    void (*gpu_op)(double *, Vector *, Vector *), double a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device);

// Templates for matrix operations
Matrix *cpu_gpu_conditional_apply_matrix_operator(
    void (*cpu_op)(Matrix *, Matrix *, Matrix *),
    void (*gpu_op)(Matrix *, Matrix *, Matrix *),
    bool (*validate)(Matrix *, Matrix *, Matrix *), Matrix *a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device);

Matrix *cpu_gpu_conditional_apply_scalar_matrix_operator(
    void (*cpu_op)(double *, Matrix *, Matrix *),
    void (*gpu_op)(double *, Matrix *, Matrix *), double a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device);

#endif
