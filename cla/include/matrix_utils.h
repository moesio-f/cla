#ifndef CLA_MATRIX_UTILS
#define CLA_MATRIX_UTILS
#include "entities.h"
#include <stdbool.h>

// Maybe allocate a matrix if ptr is NULL.
Matrix *maybe_alloc_matrix(Matrix *ptr, int rows, int columns,
                           CUDADevice *device);

// Matrix construction
Matrix *const_matrix(int rows, int columns, double value, CUDADevice *device);

// Matrix from vector
Matrix *matrix_from_vector(Vector *a, Vector2MatrixStrategy strategy);

// Clean up matrix from either CUDA or CPU
void destroy_matrix(Matrix *matrix);

// Copy matrix from `src` to `dst` (if NULL, automatically allocates memory)
Matrix *copy_matrix(Matrix *a, Matrix *dst);

// Print matrix to stdout
void print_matrix(Matrix *a, char *suffix);

// Validation functions
bool matrix_has_same_dims_same_devices(Matrix *a, Matrix *b, Matrix *dst);
bool matrix_is_mult_compat(Matrix *a, Matrix *b, Matrix *dst);
bool matrix_is_square(Matrix *a);

#endif
