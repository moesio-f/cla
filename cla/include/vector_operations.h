#ifndef CLA_VECTOR
#define CLA_VECTOR
#include "entities.h"

// Vector operations, vectors must be on same device.
// If `dst` is NULL, allocate new vector with same device as `a`
Vector *vector_add(Vector *a, Vector *b, Vector *dst);
Vector *vector_sub(Vector *a, Vector *b, Vector *dst);
Vector *vector_mult_scalar(double a, Vector *b, Vector *dst);
Vector *vector_projection(Vector *a, Vector *b, Vector *dst);
Vector *vector_element_wise_prod(Vector *a, Vector *b, Vector *dst);

// Operations that produce double (returns are always on CPU)
double vector_dot_product(Vector *a, Vector *b);
double vector_lp_norm(Vector *a, double p);
double vector_max_norm(Vector *a);
double vector_l2_norm(Vector *a);
double vector_angle_between_rad(Vector *a, Vector *b);

// Comparisons (returns are always on CPU)
bool vector_equals(Vector *a, Vector *b);
bool vector_orthogonal(Vector *a, Vector *b);
bool vector_orthonormal(Vector *a, Vector *b);

#endif
