/**
 * @file: matrix_utils.h
 *
 * This header files defines utility
 *  functions for matrix management.
 * */
#ifndef CLA_MATRIX_UTILS
#define CLA_MATRIX_UTILS
#include "entities.h"
#include <stdbool.h>

/**
 * If `ptr` is NULL, allocates memory for
 *  a matrix in the appropriate device with
 *  a default value of 0.
 * */
Matrix *maybe_alloc_matrix(Matrix *ptr, int rows, int columns,
                           CUDADevice *device);

/**
 * Create a constant matrix with the given
 *  value in the appropriate device.
 * */
Matrix *const_matrix(int rows, int columns, double value, CUDADevice *device);

/**
 * Create a matrix from a vector with the
 *  selected strategy.
 * */
Matrix *matrix_from_vector(Vector *a, Vector2MatrixStrategy strategy);

/**
 * Clean-ups a matrix constructed from the
 *  construction functions from its current
 *  device.
 * */
void destroy_matrix(Matrix *matrix);

/**
 * Copy matrix from src to dst (if NULL,
 *  automatically allocates memory)
 * */
Matrix *copy_matrix(Matrix *a, Matrix *dst);

/**
 * Prints a matrix to the stdout.
 * */
void print_matrix(Matrix *a, char *suffix);

/**
 * Check whether the matrices are on the same
 *  device and have the same dimensionality.
 * If one only needs to check two vectors,
 *  the third argument should be either `a` or
 *  `b`.
 * */
bool matrix_has_same_dims_same_devices(Matrix *a, Matrix *b, Matrix *dst);

/**
 * Check whether the matrices are compatible
 *  for matrix multiplication.
 * It checks if a*b is compatible and if dst
 *  can hold the result (i.e., c = a*b is valid).
 * */
bool matrix_is_mult_compat(Matrix *a, Matrix *b, Matrix *dst);

/**
 * Checks whether a matrix is square.
 * */
bool matrix_is_square(Matrix *a);

#endif
