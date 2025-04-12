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
 * Populates the `DEVICES` global variable.
 * */
void populate_devices();

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
 * Moves the `src` vector to the specified CUDA
 *  device using the selected strategy.
 * If `device` is NULL throws an assertion error.
 * This function presumes the `device` is in the
 *  `DEVICES` global variable.
 * This function set the `cu_vector` property
 *  of the Vector struct.
 * */
Vector *vector_to_cu(Vector *src, CUDADevice *device, CopyStrategy strategy);

/**
 * Moves the `src` vector from the CUDA device
 *  back to the CPU employing the selected
 *  strategy.
 * If the vector isn't in GPU, silent return
 *  `src` without any additional checks.
 * This function unset the `cu_vector` property
 *  of the Vector struct.
 * */
Vector *vector_to_cpu(Vector *src, CopyStrategy strategy);

/**
 * Moves the `src` matrix to the specified CUDA
 *  device using the selected strategy.
 * If `device` is NULL throws an assertion error.
 * This function presumes the `device` is in the
 *  `DEVICES` global variable.
 * This function set the `cu_matrix` property
 *  of the Matrix struct.
 * */
Matrix *matrix_to_cu(Matrix *src, CUDADevice *device, CopyStrategy strategy);

/**
 * Moves the `src` vector from the CUDA device
 *  back to the CPU employing the selected
 *  strategy.
 * If the vector isn't in GPU, silent return
 *  `src` without any additional checks.
 * This function unset the `cu_matrix` property
 *  of the Matrix struct.
 * */
Matrix *matrix_to_cpu(Matrix *src, CUDADevice *device, CopyStrategy strategy);
#endif
