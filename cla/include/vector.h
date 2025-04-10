#ifndef VECTORS
#define VECTORS
#include <stdbool.h>

typedef enum { POINT, DIRECTION } VectorType;

typedef struct {
  double *arr;
  double *x, *y, *z;
  int dims;
  VectorType type;
} Vector;

// Initialization/destruction
Vector *const_vector(int dims, VectorType type, double value);
Vector *create_vector(int dims, VectorType type, ...);
void destroy_vector(Vector *vector);

// Utilities
Vector *copy_vector(Vector *a, Vector *dst);
void print_vector(Vector *a, char *suffix);

// Operations
Vector *add_vector(Vector *a, Vector *b, Vector *dst);
Vector *sub_vector(Vector *a, Vector *b, Vector *dst);
Vector *scalar_mult_vector(double a, Vector *b, Vector *dst);
Vector *projection_vector(Vector *a, Vector *b, Vector *dst);
Vector *cross_product(Vector *a, Vector *b, Vector *dst);
Vector *element_wise_prod(Vector *a, Vector *b, Vector *dst);
double dot_product(Vector *a, Vector *b);
double l2_norm(Vector *a);
bool vector_equals(Vector *a, Vector *b);

#endif
