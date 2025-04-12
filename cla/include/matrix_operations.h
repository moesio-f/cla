/**
 * @file: matrix_operations.h
 *
 * This header files defines all
 *  supported operations on matrix.
 * It is responsibility of the caller
 *  to ensure vectors are on the same
 *  device.
 * The destination matrix is automatically
 *  managed by the operation if it is NULL.
 * */
#ifndef CLA_MATRIX
#define CLA_MATRIX
#include "entities.h"
#include <stdbool.h>

/**
 * Matrix addition.
 * */
Matrix *matrix_add(Matrix *a, Matrix *b, Matrix *dst);

/**
 * Matrix subtraction.
 * */
Matrix *matrix_sub(Matrix *a, Matrix *b, Matrix *dst);

/**
 * Matrix multiplication. Matrices must be compatible.
 * */
Matrix *matrix_mult(Matrix *a, Matrix *b, Matrix *dst);

/**
 * Matrix multiplication by scalar.
 * */
Matrix *matrix_mult_scalar(double a, Matrix *b, Matrix *dst);

/**
 * Matrix trace. Matrix must be square.
 * */
double matrix_trace(Matrix *a);

/**
 * L_pq matrix norm.
 * */
double matrix_lpq_norm(Matrix *a, double p, double q);

/**
 * L_22 matrix norm (Frobenius).
 * */
double matrix_frobenius_norm(Matrix *a);

/**
 * Check whether two matrices are equals.
 * */
bool matrix_equals(Matrix *a, Matrix *b);

#endif
