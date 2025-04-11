/**
 * @file: cuda_utils.h
 *
 * This header defines utilities related
 *  with CUDA (e.g., code execution, etc).
 * */
#ifndef CLA_CUDA_UTILS
#define CLA_CUDA_UTILS
#include "entities.h"
#include <stdbool.h>

/**
 * Run a vector-to-vector operation on either
 *  CPU or GPU (from the `device` property) after
 *  running the validation function.
 * If `dst` is NULL, automatically allocates a
 *  Vector with `alloc_dims` in the
 *  `alloc_device`.
 * Callers should ensure that `a` and `b` are
 *  on the same device prior to calling this
 *  function, otherwise it will throw an assertion
 *  error.
 * */
Vector *cpu_gpu_conditional_apply_vector_operator(
    void (*cpu_op)(Vector *, Vector *, Vector *),
    void (*gpu_op)(Vector *, Vector *, Vector *),
    bool (*validate)(Vector *, Vector *, Vector *), Vector *a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device);

/**
 * Run a scalar-to-vector operation on either
 *  CPU or GPU (from the `device` property).
 * If `dst` is NULL, automatically allocates a
 *  Vector with `alloc_dims` in the
 *  `alloc_device`.
 * The scalar `a` is automatically moved to GPU
 *  memory if the operation is to be run on GPU.
 * */
Vector *cpu_gpu_conditional_apply_scalar_vector_operator(
    void (*cpu_op)(double *, Vector *, Vector *),
    void (*gpu_op)(double *, Vector *, Vector *), double a, Vector *b,
    Vector *dst, int alloc_dims, CUDADevice *alloc_device);

/**
 * Run a matrix-to-matrix operation on either
 *  CPU or GPU (from the `device` property) after
 *  running the validation function.
 * If `dst` is NULL, automatically allocates a
 *  Matrix with `alloc_rowns` and `alloc_columns`
 *  in the `alloc_device`.
 * Callers should ensure that `a` and `b` are
 *  on the same device prior to calling this
 *  function, otherwise it will throw an assertion
 *  error.
 * */
Matrix *cpu_gpu_conditional_apply_matrix_operator(
    void (*cpu_op)(Matrix *, Matrix *, Matrix *),
    void (*gpu_op)(Matrix *, Matrix *, Matrix *),
    bool (*validate)(Matrix *, Matrix *, Matrix *), Matrix *a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device);

/**
 * Run a scalar-to-matrix operation on either
 *  CPU or GPU (from the `device` property).
 * If `dst` is NULL, automatically allocates a
 *  Matrix with `alloc_rowns` and `alloc_columns`
 *  in the `alloc_device`.
 * The scalar `a` is automatically moved to GPU
 *  memory if the operation is to be run on GPU.
 * */
Matrix *cpu_gpu_conditional_apply_scalar_matrix_operator(
    void (*cpu_op)(double *, Matrix *, Matrix *),
    void (*gpu_op)(double *, Matrix *, Matrix *), double a, Matrix *b,
    Matrix *dst, int alloc_rows, int alloc_columns, CUDADevice *alloc_device);

#endif
