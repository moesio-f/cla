/**
 * @file: entities.h
 *
 * This header defines all entities
 *  used by the library. Operations and
 *  functionalities for such entities
 *  are defined in other headers.
 * */
#ifndef CLA_ENTITIES
#define CLA_ENTITIES
#include <stdbool.h>

/**
 * Strategies for copying data between
 *  CPU and GPU (CUDA).
 * */
typedef enum {
  CopyStrategy_KEEP_SRC, /** Keep the data in source device. */
  CopyStrategy_FREE_SRC  /** Free the data in source device. */
} CopyStrategy;

/**
 * Strategy to construct a matrix from a vector.
 * */
typedef enum {
  Vector2MatrixStrategy_ROW,   /** Create a row-vector. */
  Vector2MatrixStrategy_COLUMN /** Create a column-vector. */
} Vector2MatrixStrategy;

/**
 * Represents a CUDA-capable device.
 * */
typedef struct {
  int id;         /** Unique integer identifier for this device. */
  char name[256]; /** Device name.  */
} CUDADevice;

/**
 * Represents the collection of CUDA-capable
 *  devices available for usage.
 * */
typedef struct {
  CUDADevice *devices; /** Array of available devices. */
  int deviceCount;     /** Number of CUDA devices available. */
} AvailableCUDADevices;

/**
 * Represents a vector (1d array).
 * */
typedef struct Vector {
  double *arr;        /** Data store. */
  int dims;           /** Number of dimensions (i.e., elements). */
  CUDADevice *device; /** Device where the data is located (NULL means CPU). */
  struct Vector *cu_vector; /** Vector in GPU memory (NULL if CPU). */
} Vector;

/**
 * Represents a matrix (2d array).
 * */
typedef struct Matrix {
  double **arr;       /** Data store. */
  int rows;           /** Number of rows. */
  int columns;        /** Number of columns. */
  CUDADevice *device; /** Device where the data is located (NULL means CPU). */
  struct Matrix *cu_matrix; /** Matrix in GPU memory (NULL if CPU). */
} Matrix;
#endif
