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
 * Strategy to construct a matrix from a vector.
 * */
typedef enum {
  Vector2MatrixStrategy_ROW,   /** Create a row-vector. */
  Vector2MatrixStrategy_COLUMN /** Create a column-vector. */
} Vector2MatrixStrategy;

/**
 * Represents the Kernel Launch parameters
 *  to be used by a given device. Positive
 *  values are valid, while zero or negative
 *  means that the default algorithm should
 *  be used instead.
 * */
typedef struct {
  int n_threads_x; /** Number of threads in x-dim. */
  int n_threads_y; /** Number of threads in y-dim. */
  int n_threads_z; /** Number of threads in z-dim. */
  int n_blocks_x;  /** Number of blocks in x-dim. */
  int n_blocks_y;  /** Number of blocks in y-dim. */
  int n_blocks_z;  /** Number of blocks in z-dim. */
} CUDAKernelLaunchParameters;

/**
 * Represents a CUDA-capable device.
 * */
typedef struct {
  int id;                    /** Unique integer identifier for this device. */
  char *name;                /** Device name.  */
  int max_grid_size_x;       /** Maximum size of x dimension of a grid. */
  int max_grid_size_y;       /** Maximum size of y dimension of a grid. */
  int max_grid_size_z;       /** Maximum size of z dimension of a grid. */
  int max_block_size_x;      /** Maximum size of x dimension of a block. */
  int max_block_size_y;      /** Maximum size of y dimension of a block. */
  int max_block_size_z;      /** Maximum size of z dimension of a block. */
  int max_threads_per_block; /** Maximum number of threads per block. */
  CUDAKernelLaunchParameters
      params; /** Launch parameters to be used by this device. */
} CUDADevice;

/**
 * Represents the collection of CUDA-capable
 *  devices available for usage.
 * */
typedef struct {
  CUDADevice *devices; /** Array of available devices. */
  int count;           /** Number of CUDA devices available. */
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
