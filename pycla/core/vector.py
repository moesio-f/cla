from __future__ import annotations

import contextlib
import logging
import math
from typing import Callable, Sequence

from pycla.bin.cla import CLA, _Vector

from .cuda_device import CUDADevice, Devices

LOGGER = logging.getLogger(__name__)


class Vector:
    """Class for representing vectors. Each vector represents
    a collection of values in 1d grid with N dimensions.

    Vectors can reside in a single device at a time (i.e., CPU or
    a specific CUDA-capable GPU). This class allow multiple
    operations in such vectors.

    Vectors support slicing and indexing for both data access
    and writing. Beware, only vectors on CPU support direct
    memory access currently. Therefore, some care should be
    taken with GPU vectors (i.e., bring it to CPU first).
    """

    def __init__(self, data: Sequence[int | float] | int | float, _pointer=None):
        """Constructor.

        Args:
            data (Sequence[int | float], int, float): data values to be
                stored on the vector. All vectors are created on CPU.

        Raises:
            TypeError: if data is from an unsupported type.
        """
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

        # Set the default destination Vector
        #   for operations that return a new Vector.
        # This variable should only be used
        #   by the DestinationVector context.
        self._dst = None

    @property
    def device(self) -> CUDADevice | None:
        """Current device (None if CPU, or
        data about the GPU device).

        Returns:
            CUDADevice if GPU, None if CPU.
        """
        dev = self._contents.device
        return self._devices[dev.contents.id] if dev else None

    @property
    def dims(self) -> int:
        """Number of values in this
        device (i.e., dimensionality).

        Returns:
            Number of dimensions (int).
        """
        return self._contents.dims

    def to(self, device: CUDADevice | int | str | None) -> Vector:
        """Send a vector to another computing device.

        Args:
            device (CUDADevice | int | str | None): device to
                send this vector to. If None, sends back to CPU.
                An integer argument selects the device with this
                id, while string selects a device with the same
                name.

        Raises:
            TypeError: if argument is not of correct type.
            KeyError: if a device with the respective id/name
                couldn't be find.

        Returns:
            The same Vector with in the newer device.
        """
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
        return self

    def cpu(self) -> Vector:
        """Brings a vector to CPU.

        Returns:
            The same Vector in the CPU.
        """
        self._pointer = CLA.vector_to_cpu(self._pointer)
        return self

    def release(self):
        """Releases the underlying data for this
        vector. Should only be called for advanced
        usage (i.e., intensive computation). The
        Python object still exists, therefore it
        is recommend to use a `a.release()` followed
        by `del a` to avoid any errors.
        """
        CLA.destroy_vector(self._pointer)
        self._pointer = None
        self._contents = None

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
        return self.__add_generic(other, self._dst)

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
        return self.__sub_generic(other, self._dst)

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
        return self.__mul_generic(other, self._dst)

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
        return self.__truediv_generic(other, self._dst)

    def __rtruedive__(self, other: Vector | float | int) -> Vector:
        raise NotImplementedError("Currently not supported.")

    def __itruediv__(self, other: Vector | float | int) -> Vector:
        return self.__truediv_generic(other, self)

    def __pow_generic(self, other: int, dst: Vector = None) -> Vector:
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
        return self.__pow_generic(other, self._dst)

    def __ipow__(self, other: int) -> Vector:
        return self.__pow_generic(other, self)

    def __eq__(self, other: Vector) -> bool:
        if (
            not isinstance(other, Vector)
            or self.dims != other.dims
            or self.device != other.device
        ):
            return False

        if self.has_shared_data(self, other):
            return True

        return CLA.vector_equals(self._pointer, other._pointer)

    def __str__(self) -> str:
        if self.device:
            data = "<gpu>"
        else:
            data = ", ".join(map(str, self[:10])) + (", ..." if len(self) > 10 else "")
        device = self.device.short_str() if self.device else "CPU"
        return f"Vector([{data}], dims={self.dims}, device={device})"

    def __repr__(self) -> str:
        return str(self)

    def __dell__(self):
        release()

    def l2(self) -> float:
        """L2 norm.

        Returns:
            float: l2 norm.
        """
        return CLA.vector_l2_norm(self._pointer)

    def lp(self, p: float) -> float:
        """LP norm.

        Args:
            p (float): p-value for LP norm.

        Returns:
            float: lp norm.
        """
        return CLA.vector_lp_norm(self._pointer, p)

    def max(self) -> float:
        """Return the max value in
        this vector (max-norm).

        Returns:
            float: max norm.
        """
        return CLA.vector_max_norm(self._pointer)

    def projection(self, other: Vector) -> Vector:
        """Vector projection of self onto other.

        Args:
            other (Vector): vector for self to be
                projected to.

        Raises:
            TypeError: if other is not Vector.

        Returns:
            Vector: projection of self onto other.
        """
        if not isinstance(other, Vector):
            raise TypeError("Projection only supports other vector.")

        return self._maybe_handle_different_devices(
            other, CLA.vector_projection, self._dst
        )

    def angle_to(self, other: Vector, unit: "rad" | "deg" = "rad") -> float:
        """Returns the angle between the self and other vector.

        Args:
            other (Vector): other vector.
            unit ("rad" or "deg"): whether angle should be in
                radians or degrees.

        Returns:
            float: angle between vectors in radians/degrees.
        """
        # Obtain common device if any
        self_dev = self.device
        other_dev = other.device

        # Bring values to same device
        if self_dev != other_dev:
            common_dev = self_dev if self_dev is not None else other_dev
            self._log_warning_different_devices(self_dev, other_dev, common_dev)
            self.to(common_dev)
            other.to(common_dev)

        # Get radians angle
        rad = CLA.vector_angle_between_rad(self._pointer, other._pointer)

        # Send back to correct devices
        self.to(self_dev)
        other.to(other_dev)

        # Return angle
        return rad if unit == "rad" else rad * 180.0 / math.pi

    def is_orthogonal(self, other: Vector) -> bool:
        """Whether two vectors are orthogonal (i.e.,
        90deg angle) w.r.t to the other.

        Args:
            other (Vector): other vector.

        Returns:
            bool: True if orthogonal, False otherwise.
        """
        return CLA.vector_orthogonal(self._pointer, other._pointer)

    def is_orthonormal(self, other: Vector) -> bool:
        """Whether two vectors are orthonormal (i.e.,
        90deg angle and unit vectors) w.r.t to the other.

        Args:
            other (Vector): other vector.

        Returns:
            bool: True if orthonormal, False otherwise.
        """
        return CLA.vector_orthonormal(self._pointer, other._pointer)

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

    def _set_dst(self, dst: Vector):
        self._dst = dst

    def _clear_dst(self):
        self._dst = None

    @staticmethod
    def has_shared_data(a: Vector, b: Vector):
        """Returns whether two vectors share the
        same underlying data (i.e., changes on one
        vector's data is reflected into the other).

        Args:
            a (Vector): first vector.
            b (Vector): second vector.

        Returns:
            True if vectors share data, False
                otherwise.
        """
        return CLA.points_to_same_location(a._pointer, b._pointer)

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


class ShareDestionationVector(contextlib.AbstractContextManager):
    """Context manager to avoid allocating new vectors
    for each operation. Should be used for intensive
    computation that respects the device (i.e., are made
    on the same computational device) and dimensions of
    the target vector.
    """

    def __init__(self, *src: Vector):
        """Context constructor.

        Args:
            *src (Vector): vectors whose operations
                should store results in the same
                destination.
        """
        self._src = src
        first = self._src[0]

        # Assert compatible vectors
        assert all(
            map(
                lambda v: (v.device == first.device) and (v.dims == first.dims),
                self._src[1:],
            )
        )

        # Initialize destination Vector with same
        #   dims and device as src[0]
        self._dst = Vector([0.0] * first.dims)
        self._dst = self._dst.to(first.device)

        # The destination vector has itself as
        #   destination for any operation with
        #   it
        self._dst._set_dst(self._dst)

        # Obtain any set dst Vector
        self._initial_dst = list(map(lambda v: v._dst, self._src))

    def __enter__(self) -> Vector:
        # Set all src with dst
        for v in self._src:
            v._set_dst(self._dst)

        return self._dst

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unset the src and maybe reset the original
        #   source (i.e., nested context)
        for i, v in enumerate(self._src):
            v._clear_dst()
            dst = self._initial_dst[i]
            if dst:
                v._set_dst(dst)
