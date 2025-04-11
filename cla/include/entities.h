#ifndef CLA_ENTITIES
#define CLA_ENTITIES
#include <stdbool.h>
/* Entities.
 *
 * This header defines all entities
 *  used by the library. Operations and
 *  functionalities for such entities
 *  are defined in other header files.
 * */

// Enumeration for copy strategy between devices
typedef enum { KEEP_SRC, FREE_SRC } CopyStrategy;

// Enumeration for constructing matrix from vector
typedef enum { SINGLE_ROW, SINGLE_COLUMN } Vector2MatrixStrategy;

// CUDA Device struct
typedef struct {
  int id;
  char name[256];
} CUDADevice;

// Main vector struct
typedef struct Vector {
  // Data store as array
  double *arr;
  // Store number of dimensions
  int dims;
  // Store current CUDA device if any (NULL=CPU)
  CUDADevice *device;
  struct Vector *cu_vector;
} Vector;

// Main matrix struct
typedef struct Matrix {
  // Data store as array
  double **arr;
  // Store number of rows and columns
  int rows, columns;
  // Store CUDA device if any (NULL=CPU)
  CUDADevice *device;
  struct Matrix *cu_matrix;
} Matrix;

#endif
