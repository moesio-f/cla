#ifndef MATRICES
#define MATRICES

#include "vector.h"
#include <stdbool.h>

typedef struct {
  double **arr;
  int rows, columns;
} Matrix;

// Initialization/destruction
Matrix *const_matrix(int rows, int columns, double value);
void destroy_matrix(Matrix *matrix);

// Utilities
Matrix *copy_matrix(Matrix *a, Matrix *dst);
Matrix *matrix_from_vector(Vector *a, bool copy);
void print_matrix(Matrix *a, char *suffix);

// Operations
Matrix *add_matrix(Matrix *a, Matrix *b, Matrix *dst);
Matrix *sub_matrix(Matrix *a, Matrix *b, Matrix *dst);
Matrix *mult_matrix(Matrix *a, Matrix *b);
Vector *mult_matrix_by_vector(Matrix *a, Vector *b);
Matrix *scalar_mult_matrix(float a, Matrix *b, Matrix *dst);
Matrix *inverse(Matrix *a, Matrix *dst);
double determinant(Matrix *a);

#endif
