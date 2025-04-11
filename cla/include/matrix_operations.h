#ifndef CLA_MATRIX
#define CLA_MATRIX
#include "entities.h"
#include <stdbool.h>

// Matrix operations, matrix must be on same device.
// If `dst` is NULL, allocate new matrix with same device as `a`
Matrix *matrix_add(Matrix *a, Matrix *b, Matrix *dst);
Matrix *matrix_sub(Matrix *a, Matrix *b, Matrix *dst);
Matrix *matrix_mult(Matrix *a, Matrix *b);
Matrix *matrix_mult_scalar(double a, Matrix *b, Matrix *dst);

// Matrix-vector operations, objects must be on same device.
Vector *mult_matrix_by_vector(Matrix *a, Vector *b);

// Operations that produce double (returns are always on CPU)
double matrix_trace(Matrix *a);
double matrix_lpq_norm(Matrix *a, double p, double q);
double matrix_frobenius_norm(Matrix *a);

// Comparisons (return are always on CPU)
bool matrix_equals(Matrix *a, Matrix *b);

#endif
