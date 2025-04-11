/**
 * @file: vector_operations.h
 *
 * This header files defines all
 *  supported operations on vectors.
 * It is responsibility of the caller
 *  to ensure vectors are on the same
 *  device.
 * The destination vector is automatically
 *  managed by the operation if it is NULL.
 * */
#ifndef CLA_VECTOR
#define CLA_VECTOR
#include "entities.h"

/**
 * Vector addition. Vectors must have same dims.
 * */
Vector *vector_add(Vector *a, Vector *b, Vector *dst);

/**
 * Vector subtraction. Vectors must have same dims.
 * */
Vector *vector_sub(Vector *a, Vector *b, Vector *dst);

/**
 * Vector multiplication by scalar.
 * */
Vector *vector_mult_scalar(double a, Vector *b, Vector *dst);

/**
 * Projection of vector a onto b.
 * */
Vector *vector_projection(Vector *a, Vector *b, Vector *dst);

/**
 * Element wise product between vectors.
 * */
Vector *vector_element_wise_prod(Vector *a, Vector *b, Vector *dst);

/**
 * Dot product.
 * */
double vector_dot_product(Vector *a, Vector *b);

/**
 * L_p norm.
 * */
double vector_lp_norm(Vector *a, double p);

/**
 * Max norm (return max value of vector).
 * */
double vector_max_norm(Vector *a);

/**
 * L_2 norm (Euclidean norm).
 * */
double vector_l2_norm(Vector *a);

/**
 * Return the angle (radians) between vectors.
 * */
double vector_angle_between_rad(Vector *a, Vector *b);

/**
 * Compares whether two vectors are equals (within
 *  a tolerance interval).
 * */
bool vector_equals(Vector *a, Vector *b);

/**
 * Check whether two vectors are orthogonal.
 * */
bool vector_orthogonal(Vector *a, Vector *b);

/**
 * Checks whether two vectors are orthonormal.
 * */
bool vector_orthonormal(Vector *a, Vector *b);

#endif
