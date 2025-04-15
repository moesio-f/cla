from __future__ import annotations

import logging
from typing import Callable, Sequence

from pycla.bin.cla import CLA, _Vector

from .cuda_device import CUDADevice, Devices

LOGGER = logging.getLogger(__name__)


class Vector:
    def __init__(self, data: Sequence[int | float] | int | float, _pointer=None):
        if isinstance(data, int) or isinstance(data, float):
            data = [data]

        if not isinstance(data, Sequence) and _pointer is None:
            raise TypeError("Data should be a sequence or numeric.")

        # Initialize devices
        self._devices = Devices()

        if _pointer is None:
            # Create vector in CPU.
            # Pointer is the actual pointer to the
            #   vector, while contents contains the values
            #   from this pointer. Python always create a new
            #   object with the contents, in order to improve
            #   performance we keep both allocated.
            self._pointer = CLA.const_vector(len(data), 0.0, None)
            self._contents = self._pointer.contents

            # Initialize its contents
            for i in range(self._contents.dims):
                self._contents.arr[i] = data[i]
        else:
            # Directly initialize with a pointer
            self._pointer = _pointer
            self._contents = self._pointer.contents

    @property
    def device(self) -> CUDADevice | None:
        dev = self._contents.device
        return self._devices[dev.contents.id] if dev else None

    @property
    def dims(self) -> int:
        return self._contents.dims

    def to(self, device: CUDADevice | int | str | None) -> Vector:
        if not isinstance(device, CUDADevice):
            if isinstance(device, int) or isinstance(device, str):
                device = self._devices[device]
            elif device is not None:
                raise TypeError("Device should be CUDADevice, int, str or None.")

        if device is None:
            self.cpu()
            return self

        # Send to GPU
        self._pointer = CLA.vector_to_cu(
            self._pointer, self._devices._get_pointer(device)
        )
        self._contents = self._pointer.contents
        return self

    def cpu(self) -> Vector:
        self._pointer = CLA.vector_to_cpu(self._pointer)
        self._contents = self._pointer.contents
        return self

    def __len__(self) -> int:
        return self.dims

    def __getitem__(self, key: int | slice) -> list[float] | float:
        is_key_int = isinstance(key, int)
        is_key_slice = isinstance(key, slice)
        if not (is_key_int or is_key_slice):
            raise TypeError("Vectors should be indexed with int or slices.")

        if is_key_slice:
            # Some slices have form a:None, which
            #   have to be evaluated prior to accessing
            #   the pointer.
            key = self._sanitize_slice(key)

        dev = self.device
        if dev:
            self._log_warning_copying_to_cpu(dev)
            self.cpu()

        data = self._contents.arr[key]

        if dev:
            self.to(dev)

        return data

    def __setitem__(self, key: int | slice, value: float | list[float]):
        is_key_int = isinstance(key, int)
        is_key_slice = isinstance(key, slice)
        is_value_float = isinstance(value, float)
        is_value_list = isinstance(value, list)

        if not (is_key_int or is_key_slice):
            raise TypeError("Vectors should be indexed with int or slices.")

        if is_key_int and is_value_list:
            raise TypeError("Key (int) and values (list) are of incompatible type.")

        dev = self.device
        if dev:
            self._log_warning_copying_to_cpu(dev)
            self.cpu()

        if is_key_slice:
            # Some slices have form a:None, which
            #   have to be evaluated prior to accessing
            #   the pointer.
            key = self._sanitize_slice(key)

        self._contents.arr[key] = value

        if dev:
            self.to(dev)

    def __iter__(self):
        if self.device:
            raise SystemError(
                f"Cannot iterate over vector in GPU (device={self.device.short_str()})"
            )

        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: float | int) -> bool:
        for v in self:
            if float(value) == v:
                return True

        return False

    def __neg__(self) -> Vector:
        return self * -1

    def __pos__(self) -> Vector:
        return self

    def __add_generic(self, other: Vector | float | int, dst: Vector = None) -> Vector:
        # If float, broadcast
        if isinstance(other, float) or isinstance(other, int):
            other = Vector([float(other)] * self.dims)

        # If vector, run operations to create new vector
        #   with result.
        if isinstance(other, Vector):
            return self._maybe_handle_different_devices(other, CLA.vector_add, dst)

        raise TypeError("Other should be vector or float.")

    def __add__(self, other: Vector | float | int) -> Vector:
        return self.__add_generic(other)

    def __radd__(self, other: Vector | float | int) -> Vector:
        return self + other

    def __iadd__(self, other: Vector | float | int) -> Vector:
        return self.__add_generic(other, self)

    def __sub_generic(self, other: Vector | float | int, dst: Vector = None) -> Vector:
        # If float, broadcast
        if isinstance(other, float) or isinstance(other, int):
            other = Vector([float(other)] * self.dims)

        # If vector, run operations to create new vector
        #   with result.
        if isinstance(other, Vector):
            return self._maybe_handle_different_devices(other, CLA.vector_sub, dst)

        raise TypeError("Other should be vector or float.")

    def __sub__(self, other: Vector | float | int) -> Vector:
        return self.__sub_generic(other)

    def __rsub__(self, other: Vector | float | int) -> Vector:
        return -self + other

    def __isub__(self, other: Vector | float | int) -> Vector:
        return self.__sub_generic(other, self)

    def __mul_generic(self, other: Vector | float | int, dst: Vector = None) -> Vector:
        if isinstance(other, float) or isinstance(other, int):
            result = CLA.vector_mult_scalar(
                float(other), self._pointer, dst._pointer if dst else None
            )
            result = result if isinstance(result, Vector) else Vector(None, result)
            return result

        if isinstance(other, Vector):
            return self._maybe_handle_different_devices(
                other, CLA.vector_element_wise_prod, dst
            )

        raise TypeError("Other should be vector or float.")

    def __mul__(self, other: Vector | float | int) -> Vector:
        return self.__mul_generic(other)

    def __rmul__(self, other: Vector | float | int):
        return self * other

    def __imul__(self, other: Vector | float | int) -> Vector:
        return self.__mul_generic(other, self)

    def __matmul__(self, other: Vector) -> float:
        if isinstance(other, Vector):
            self_dev = self.device
            other_dev = other.device

            # Bring values to same device
            if self_dev != other_dev:
                common_dev = self_dev if self_dev is not None else other_dev
                self._log_warning_different_devices(self_dev, other_dev, None)
                self.to(common_dev)
                other.to(common_dev)

            # Apply function
            result = CLA.vector_dot_product(self._pointer, other._pointer)

            # Bring result to same device as self
            #   and other to other
            self.to(self_dev)
            other.to(other_dev)

            # Return result
            return result

        raise TypeError("Other should be vector.")

    def __truediv_generic(
        self, other: Vector | float | int, dst: Vector = None
    ) -> Vector:
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul_generic(1.0 / float(other), dst)

        if isinstance(other, Vector):
            # Update when C API supports vector division
            raise NotImplementedError("Currently not supported.")

        raise TypeError("Other should be vector or float.")

    def __truediv__(self, other: Vector | float | int, dst: Vector = None) -> Vector:
        return self.__truediv_generic(other)

    def __rtruedive__(self, other: Vector | float | int) -> Vector:
        raise NotImplementedError("Currently not supported.")

    def __itruediv__(self, other: Vector | float | int) -> Vector:
        return self.__truediv_generic(other, self)

    def __pow_generic(self, other: int, dst: Vector = None) -> Vector:
        raise NotImplementedError("Currently not supported.")

        if not isinstance(other, int):
            raise TypeError("Other should be integer.")

        # Copy seems bugged, fix on C API
        result = Vector(None, CLA.copy_vector(self._pointer, None))
        for i in range(other - 1):
            result = self.__mul_generic(result, result)

        if dst:
            # Direct pointer operation. Does a deep copy
            #   so that when result is GC'd, the self vector
            #   keeps existing in memory.
            # Contents don't change (this operation doesn't
            #   change dims/device).
            CLA.copy_vector(result._pointer, dst._pointer)
        else:
            dst = result

        return dst

    def __pow__(self, other: int) -> Vector:
        return self.__pow_generic(other)

    def __ipow__(self, other: int) -> Vector:
        return self.__pow_generic(other, self)

    def __str__(self) -> str:
        data = "<gpu>" if self.device else ", ".join(map(str, self))
        device = self.device.short_str() if self.device else "CPU"
        return f"Vector([{data}], dims={self.dims}, device={device})"

    def __dell__(self):
        CLA.destroy_vector(self._pointer)
        self._pointer = None
        self._contents = None

    def _maybe_handle_different_devices(
        self,
        other: Vector,
        cla_fn: Callable[[_Vector, _Vector, _Vector], _Vector],
        dst: Vector = None,
    ) -> Vector:
        self_dev = self.device
        other_dev = other.device

        # Bring values to same device
        if self_dev != other_dev:
            common_dev = self_dev if self_dev is not None else other_dev
            self._log_warning_different_devices(self_dev, other_dev, common_dev)
            self.to(common_dev)
            other.to(common_dev)

        # Apply function
        result = Vector(
            None, cla_fn(self._pointer, other._pointer, dst._pointer if dst else None)
        )

        # Bring result to same device as self
        #   and other to other
        result.to(self_dev)
        self.to(self_dev)
        other.to(other_dev)

        # Return result
        return result

    def _sanitize_slice(self, key: slice) -> slice:
        return slice(*key.indices(len(self)))

    @staticmethod
    def _log_warning_different_devices(a: CUDADevice, b: CUDADevice, dst: CUDADevice):
        def _str(dev: CUDADevice) -> str:
            return dev.short_str() if dev else "CPU"

        LOGGER.warning(
            "Vectors are in different devices (%s != %s).\n"
            "Result vector is going to be on %s.",
            _str(a),
            _str(b),
            _str(dst),
        )

    @staticmethod
    def _log_warning_copying_to_cpu(a: CUDADevice):
        LOGGER.warning("Vector is on %s, temporarily copying to CPU.", a.short_str())
