#include "../include/device_management.h"
#include "../include/constants.h"
#include "../include/entities.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

AvailableCUDADevices *DEVICES = NULL;

CUDADevice *get_device_by_id(int id) {
  populate_devices();

  if (id >= DEVICES->count) {
    return NULL;
  }

  return &DEVICES->devices[id];
}

CUDADevice *get_device_by_name(char *name) {
  populate_devices();

  for (int i = 0; i < DEVICES->count; i++) {
    CUDADevice *device = &DEVICES->devices[i];
    if (strcmp(device->name, name) == 0) {
      return device;
    }
  }

  return NULL;
}

char *device_to_str(CUDADevice *device, char *dst) {
  if (dst == NULL) {
    dst = (char *)malloc(sizeof(char) * STR_SIZE);
  }

  // Format string
  sprintf(dst,
          "CUDADevice{id=%d, name=\"%s\", max_grid_size=(%d, %d, %d), "
          "max_block_size=(%d, %d, %d), "
          "max_threads_per_block=%d}",
          device->id, device->name, device->max_grid_size_x,
          device->max_grid_size_y, device->max_grid_size_z,
          device->max_block_size_x, device->max_block_size_y,
          device->max_block_size_z, device->max_threads_per_block);
  return dst;
}

int cuda_get_device_count() {
  populate_devices();
  int count = DEVICES->count;
  if (count <= 0) {
    clear_devices();
  }

  return count;
}

bool has_cuda() {
  populate_devices();
  bool has_cuda = DEVICES->count > 0;
  if (!has_cuda) {
    clear_devices();
  }

  return has_cuda;
}
