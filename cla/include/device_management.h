/**
 * @file: device_management.h
 * Device management functions. Provide
 *  higher-level functions using the CUDA
 *  runtime API.
 * */
#ifndef CLA_CUDA
#define CLA_CUDA
#include "entities.h"
#include <stdbool.h>

extern AvailableCUDADevices
    *DEVICES; /** Global variable with available CUDA devices. */

/**
 * Synchronize all CUDA devices (i.e.,
 *  finish all kernels before proceeding).
 * */
void synchronize_devices();

/**
 * Populates the `DEVICES` global variable.
 * If is already populated, silently ignores.
 * This function is automatically called whenever
 *  a CUDA operation is required.
 * */
void populate_devices();

/**
 * Clears the `DEVICES` global variable.
 * If it is already clear, silently ignores.
 * This function should be used to clear allocated
 *  memory (i.e., cleanup, etc).
 * */
void clear_devices();

/**
 * Returns whether the system has any CUDA
 *  capable device.
 * */
bool has_cuda();

/**
 * Returns the number of CUDA devices.
 * */
int cuda_get_device_count();

/**
 * Return a device by the `id` property.
 * NULL if no device with this id is found.
 * */
CUDADevice *get_device_by_id(int id);

/**
 * Return a device by the `name` property.
 * NULL if no device with this name is found.
 * Device names are case sensitive.
 * */
CUDADevice *get_device_by_name(char *name);

/**
 * Converts device information to string.
 * If dst is NULL, allocates a string with 512
 *  characters (null-terminated).
 * */
char *device_to_str(CUDADevice *device, char *dst);

/**
 * Moves the `src` vector to the specified CUDA
 *  device using the selected strategy.
 * If `device` is NULL throws an assertion error.
 * This function presumes the `device` is in the
 *  `DEVICES` global variable.
 * This function set the `cu_vector` property
 *  of the Vector struct.
 * */
Vector *vector_to_cu(Vector *src, CUDADevice *device);

/**
 * Moves the `src` vector from the CUDA device
 *  back to the CPU.
 * If the vector isn't in GPU, silent return
 *  `src` without any additional checks.
 * This function unset the `cu_vector` property
 *  of the Vector struct.
 * */
Vector *vector_to_cpu(Vector *src);

/**
 * Moves the `src` matrix to the specified CUDA
 *  device.
 * If `device` is NULL throws an assertion error.
 * This function presumes the `device` is in the
 *  `DEVICES` global variable.
 * This function set the `cu_matrix` property
 *  of the Matrix struct.
 * */
Matrix *matrix_to_cu(Matrix *src, CUDADevice *device);

/**
 * Moves the `src` vector from the CUDA device
 *  back to the CPU.
 * If the vector isn't in GPU, silent return
 *  `src` without any additional checks.
 * This function unset the `cu_matrix` property
 *  of the Matrix struct.
 * */
Matrix *matrix_to_cpu(Matrix *src);
#endif
