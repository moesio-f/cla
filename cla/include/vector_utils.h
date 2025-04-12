/**
 * @file: vector_utils.h
 *
 * This header files defines utility
 *  functions for vector management.
 * */
#ifndef CLA_VECTOR_UTILS
#define CLA_VECTOR_UTILS
#include "entities.h"
#include <stdbool.h>

/**
 * If `ptr` is NULL, allocates memory for
 *  a vector in the appropriate device with
 *  a default value of 0.
 * */
Vector *maybe_alloc_vector(Vector *ptr, int dims, CUDADevice *device);

/**
 * Create a constant vector with the given
 *  value in the appropriate device.
 * */
Vector *const_vector(int dims, double value, CUDADevice *device);

/**
 * Create a vector from literals (double).
 * */
Vector *create_vector(int dims, CUDADevice *device, ...);

/**
 * Clean-ups a vector constructed from the
 *  construction functions from its current
 *  device.
 * */
void destroy_vector(Vector *vector);

/**
 * Copy vector from src to dst (if NULL,
 *  automatically allocates memory)
 * */
Vector *copy_vector(Vector *src, Vector *dst);

/**
 * Prints a vector to the stdout.
 * */
void print_vector(Vector *a, char *suffix);

/**
 * Check whether the vectors are on the same
 *  device and have the same dimensionality.
 * If one only needs to check two vectors,
 *  the third argument should be either `a` or
 *  `b`.
 * */
bool vector_has_same_dims_same_devices(Vector *a, Vector *b, Vector *dst);

#endif
