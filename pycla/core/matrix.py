from __future__ import annotations

import contextlib
import logging
from typing import Callable, Sequence

from pycla.bin.cla import CLA, _Matrix

from .cuda_device import CUDADevice, Devices

LOGGER = logging.getLogger(__name__)


class Matrix:
    """Class for representing matrices. Each matrix represents
    a collection of values in 2d grid with NxM dimensions.

    Matrices can reside in a single device at a time (i.e., CPU or
    a specific CUDA-capable GPU). This class allow multiple
    operations in such matrices.

    Matrices support slicing and indexing for both data access
    and writing. Beware, only matrices on CPU support direct
    memory access currently. Therefore, some care should be
    taken with GPU matrices (i.e., bring it to CPU first).
    """

    def __init__(
        self, data: Sequence[Sequence[int | float]] | int | float, _pointer=None
    ):
        """Constructor.

        Args:
            data (Sequence[Sequence[int | float]], int, float): data values to be
                stored on the matrix. All matrices are created on CPU.

        Raises:
            TypeError: if data is from an unsupported type.
            ValueError: if data 2d-grid has ragged dimensions.
        """
        if isinstance(data, int) or isinstance(data, float):
            data = [[float(data)]]

        if not isinstance(data, Sequence) and _pointer is None:
            raise TypeError("Data should be a sequence or numeric.")

        # Initialize devices
        self._devices = Devices()

        is_data_ragged = False
        if _pointer is None:
            # Create matrix in CPU.
            # Pointer is the actual pointer to the
            #   matrix, while contents contains the values
            #   from this pointer. Python always create a new
            #   object with the contents, in order to improve
            #   performance we keep both allocated.
            self._pointer = CLA.const_matrix(len(data), len(data[0]), 0.0, None)
            self._contents = self._pointer.contents

            # Initialize its contents
            for i in range(self._contents.rows):
                for j in range(self._contents.columns):
                    row = data[i]
                    if len(row) != self._contents.columns:
                        is_data_ragged = True
                        break

                    self._contents.arr[i][j] = row[j]
        else:
            # Directly initialize with a pointer
            self._pointer = _pointer
            self._contents = self._pointer.contents

        # Set the default destination Matrix
        #   for operations that return a new Matrix.
        # This variable should only be used
        #   by the ShareDestionationMatrix context.
        self._dst = None

        if is_data_ragged:
            self.release()
            raise ValueError(
                "Data can't be ragged (i.e., different number of columns per row)."
            )

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
    def rows(self) -> int:
        """Number of rows.

        Returns:
            Number of rows (int).
        """
        return self._contents.rows

    @property
    def columns(self) -> int:
        """Number of columns.

        Returns:
            Number of columns (int).
        """
        return self._contents.columns

    def to(self, device: CUDADevice | int | str | None) -> Matrix:
        """Send a matrix to another computing device.

        Args:
            device (CUDADevice | int | str | None): device to
                send this matrix to. If None, sends back to CPU.
                An integer argument selects the device with this
                id, while string selects a device with the same
                name.

        Raises:
            TypeError: if argument is not of correct type.
            KeyError: if a device with the respective id/name
                couldn't be find.

        Returns:
            The same Matrix with in the newer device.
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
        self._pointer = CLA.matrix_to_cu(
            self._pointer, self._devices._get_pointer(device)
        )
        return self

    def cpu(self) -> Matrix:
        """Brings a matrix to CPU.

        Returns:
            The same Matrix in the CPU.
        """
        self._pointer = CLA.matrix_to_cpu(self._pointer)
        return self

    def release(self):
        """Releases the underlying data for this
        matrix. Should only be called for advanced
        usage (i.e., intensive computation). The
        Python object still exists, therefore it
        is recommend to use a `a.release()` followed
        by `del a` to avoid any errors.
        """
        CLA.destroy_matrix(self._pointer)
        self._pointer = None
        self._contents = None

    def __len__(self) -> int:
        return self.rows * self.columns

    def __getitem__(
        self,
        key: Tuple[slice | int, slice | int],
    ) -> float | list[float] | list[list[float]]:
        row, column = key
        is_row_int = isinstance(row, int)
        is_row_slice = isinstance(row, slice)
        is_column_int = isinstance(column, int)
        is_column_slice = isinstance(column, slice)
        if not ((is_row_int or is_row_slice) and (is_column_int or is_column_slice)):
            raise TypeError("Matrices should be indexed with int or slices.")

        # Some slices have form a:None, which
        #   have to be evaluated prior to accessing
        #   the pointer.
        row = self._maybe_sanitize_if_slice(row, is_row=True)
        column = self._maybe_sanitize_if_slice(column, is_row=False)

        dev = self.device
        if dev:
            self._log_warning_copying_to_cpu(dev)
            self.cpu()

        # Fetch data
        if is_row_slice:
            # Initialize empty data and fetch rows
            data = []
            data_rows = self._contents.arr[row]

            # For each row, query the actual column
            for dr in data_rows:
                # Fetch column data
                dc = dr[column]

                # If it is a single element,
                #   wrap into a list
                if is_column_int:
                    dc = [dc]

                # Store data
                data.append(dc)
        else:
            data = self._contents.arr[row][column]

        if dev:
            self.to(dev)

        return data

    def __setitem__(
        self,
        key: Tuple[slice | int, slice | int],
        value: int | float | list[int | float] | list[list[int | float]],
    ):
        # TODO: improve validation and ensure
        #   correct typing for (key, value) pair.
        row, column = key
        is_row_int = isinstance(row, int)
        is_row_slice = isinstance(row, slice)
        is_column_int = isinstance(column, int)
        is_column_slice = isinstance(column, slice)
        is_value_float = isinstance(value, float) or isinstance(value, int)
        is_value_list = isinstance(value, list)
        is_value_list_list = False

        if not ((is_row_int or is_row_slice) and (is_column_int or is_column_slice)):
            raise TypeError("Matrices should be indexed with int or slices.")

        if is_value_list:
            is_value_list_list = isinstance(value[0], list)
            is_value_list = not is_value_list_list

        value_error = (
            (is_row_int and is_column_int and not is_value_float)
            or (is_row_int and is_column_slice and not is_value_list)
            or (is_row_slice and is_column_int and not is_value_list_list)
            or (is_row_slice and is_column_slice and not is_value_list_list)
        )
        if value_error:
            raise ValueError("Key and values are of incompatible type.")

        # Sanitize slices
        row = self._maybe_sanitize_if_slice(row, is_row=True)
        column = self._maybe_sanitize_if_slice(column, is_row=False)

        def _slice_length(s: slice):
            return 1 + (s.stop - 1 - s.start) // s.step

        # If key is (int, slice), we should have a single row and the appropriate
        #   number of columns.
        if is_row_int and is_column_slice:
            n_columns_value = _slice_length(column)
            if n_columns_value != len(value):
                raise ValueError(
                    "Invalid value for (int, slice) key. "
                    f"Should be list with {n_columns_value} values."
                )

        # If key is (slice, int), we should have multiple rows with one
        #   list of a single element
        if is_row_slice and is_column_int:
            n_rows_value = _slice_length(row)
            if n_rows_value != len(value) or any(
                not isinstance(v, list) or len(v) != 1 for v in value
            ):
                raise ValueError(
                    "Invalid value for (slice, int) key. "
                    f"Should be {n_rows_value} lists with 1 value each."
                )

        # If key is (slice, slice), we should have multiple rows
        #   with multiple columns
        if is_row_slice and is_column_slice:
            n_rows_value = _slice_length(row)
            n_columns_value = _slice_length(column)
            if n_rows_value != len(value) or any(
                not isinstance(v, list) or len(v) != n_columns_value for v in value
            ):
                raise ValueError(
                    "Invalid value for (slice, int) key. "
                    f"Should be {n_rows_value} lists with "
                    f"{n_columns_value} value each."
                )

        # Prepare for operation
        dev = self.device
        if dev:
            self._log_warning_copying_to_cpu(dev)
            self.cpu()

        # Set values
        for row_idx in range(row.start, row.stop, row.step) if is_row_slice else [row]:
            for col_idx in (
                range(column.start, column.stop, column.step)
                if is_column_slice
                else [column]
            ):
                # Obtaining the value to set (must be float/int)
                # We start by assuming it is a scalar
                value_at_idx = value
                if is_value_list:
                    # If it is a list, we take the appropriate
                    #   column from this list
                    value_at_idx = value_at_idx[col_idx]
                elif is_value_list_list:
                    # If it is a list of a list, we take
                    #   the appropriate index
                    value_at_idx = value_at_idx[row_idx][col_idx]

                # Update underlying array
                self._contents.arr[row_idx][col_idx] = float(value_at_idx)

        if dev:
            self.to(dev)

    def __iter__(self):
        if self.device:
            raise SystemError(
                f"Cannot iterate over matrix in GPU (device={self.device.short_str()})"
            )

        for i in range(self.rows):
            yield self[i, :]

    def __contains__(self, value: float | int) -> bool:
        for l in self:
            for v in l:
                if float(value) == v:
                    return True

        return False

    def __neg__(self) -> Matrix:
        return self * -1

    def __pos__(self) -> Matrix:
        return self

    def __add_generic(self, other: Matrix | float | int, dst: Matrix = None) -> Matrix:
        # If float, broadcast
        if isinstance(other, float) or isinstance(other, int):
            other = Matrix([[float(other)] * self.columns] * self.rows)

        # If matrix, run operations to create new matrix
        #   with result.
        if isinstance(other, Matrix):
            return self._maybe_handle_different_devices(other, CLA.matrix_add, dst)

        raise TypeError("Other should be matrix or float.")

    def __add__(self, other: Matrix | float | int) -> Matrix:
        return self.__add_generic(other, self._dst)

    def __radd__(self, other: Matrix | float | int) -> Matrix:
        return self + other

    def __iadd__(self, other: Matrix | float | int) -> Matrix:
        return self.__add_generic(other, self)

    def __sub_generic(self, other: Matrix | float | int, dst: Matrix = None) -> Matrix:
        # If float, broadcast
        if isinstance(other, float) or isinstance(other, int):
            other = Matrix([[float(other)] * self.columns] * self.rows)

        # If matrix, run operations to create new matrix
        #   with result.
        if isinstance(other, Matrix):
            return self._maybe_handle_different_devices(other, CLA.matrix_sub, dst)

        raise TypeError("Other should be matrix or float.")

    def __sub__(self, other: Matrix | float | int) -> Matrix:
        return self.__sub_generic(other, self._dst)

    def __rsub__(self, other: Matrix | float | int) -> Matrix:
        return -self + other

    def __isub__(self, other: Matrix | float | int) -> Matrix:
        return self.__sub_generic(other, self)

    def __mul_generic(self, other: float | int, dst: Matrix = None) -> Matrix:
        if isinstance(other, float) or isinstance(other, int):
            result = CLA.matrix_mult_scalar(
                float(other), self._pointer, dst._pointer if dst else None
            )
            result = result if isinstance(result, Matrix) else Matrix(None, result)
            return result

        raise TypeError("Other should be matrix or float.")

    def __mul__(self, other: float | int) -> Matrix:
        return self.__mul_generic(other, self._dst)

    def __rmul__(self, other: float | int):
        return self * other

    def __imul__(self, other: float | int) -> Matrix:
        return self.__mul_generic(other, self)

    def __matmul__(self, other: Matrix) -> Matrix:
        if isinstance(other, Matrix):
            return self._maybe_handle_different_devices(
                other, CLA.matrix_mult, self._dst
            )

        raise TypeError("Other should be matrix.")

    def __truediv_generic(
        self, other: Matrix | float | int, dst: Matrix = None
    ) -> Matrix:
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul_generic(1.0 / float(other), dst)

        if isinstance(other, Matrix):
            # Update when C API supports matrix division
            raise NotImplementedError("Currently not supported.")

        raise TypeError("Other should be matrix or float.")

    def __truediv__(self, other: Matrix | float | int, dst: Matrix = None) -> Matrix:
        return self.__truediv_generic(other, self._dst)

    def __rtruedive__(self, other: Matrix | float | int) -> Matrix:
        raise NotImplementedError("Currently not supported.")

    def __itruediv__(self, other: Matrix | float | int) -> Matrix:
        return self.__truediv_generic(other, self)

    def __eq__(self, other: Matrix) -> bool:
        if (
            not isinstance(other, Matrix)
            or self.rows != other.rows
            or self.columns != other.columns
            or self.device != other.device
        ):
            return False

        if self.has_shared_data(self, other):
            return True

        return CLA.matrix_equals(self._pointer, other._pointer)

    def __str__(self) -> str:
        if self.device:
            data = "<gpu>"
        else:
            data = ", ".join(
                map(
                    lambda v: f"[{', '.join(map(str, v[:5]))}{', ...' if len(v) > 5 else ''}]",
                    self[:5, :],
                )
            ) + (",\n..." if self.rows > 5 else "")
        device = self.device.short_str() if self.device else "CPU"
        return f"Matrix([{data}], dims=({self.rows}, {self.columns}), device={device})"

    def __repr__(self) -> str:
        return str(self)

    def __dell__(self):
        release()

    def frobenius(self) -> float:
        """Frobenius norm.

        Returns:
            float: frobenius norm.
        """
        return CLA.matrix_frobenius_norm(self._pointer)

    def lpq(self, p: float, q: float) -> float:
        """LPQ norm.

        Args:
            p (float): p-value for LPQ norm.
            q (float): q-value for LPQ norm.

        Returns:
            float: lpq norm.
        """
        return CLA.matrix_lpq_norm(self._pointer, p, q)

    def trace(self) -> float:
        """Return the trace of this
        matrix.

        Returns:
            float: trace.
        """
        return CLA.matrix_trace(self._pointer)

    def _maybe_handle_different_devices(
        self,
        other: Matrix,
        cla_fn: Callable[[_Matrix, _Matrix, _Matrix], _Matrix],
        dst: Matrix = None,
    ) -> Matrix:
        self_dev = self.device
        other_dev = other.device

        # Bring values to same device
        if self_dev != other_dev:
            common_dev = self_dev if self_dev is not None else other_dev
            self._log_warning_different_devices(self_dev, other_dev, common_dev)
            self.to(common_dev)
            other.to(common_dev)

        # Apply function
        result = Matrix(
            None, cla_fn(self._pointer, other._pointer, dst._pointer if dst else None)
        )

        # Bring result to same device as self
        #   and other to other
        result.to(self_dev)
        self.to(self_dev)
        other.to(other_dev)

        # Return result
        return result

    def _maybe_sanitize_if_slice(self, key: slice | int, is_row: bool) -> slice | int:
        target = self.rows if is_row else self.columns
        if not isinstance(key, slice):
            if key > target:
                raise ValueError("Key is out of bounds.")
            return key

        return slice(*key.indices(target))

    def _set_dst(self, dst: Matrix):
        self._dst = dst

    def _clear_dst(self):
        self._dst = None

    @staticmethod
    def has_shared_data(a: Matrix, b: Matrix):
        """Returns whether two matrices share the
        same underlying data (i.e., changes on one
        matrix's data is reflected into the other).

        Args:
            a (Matrix): first matrix.
            b (Matrix): second matrix.

        Returns:
            True if matrixces share data, False
                otherwise.
        """
        return CLA.points_to_same_location(a._pointer, b._pointer)

    @staticmethod
    def _log_warning_different_devices(a: CUDADevice, b: CUDADevice, dst: CUDADevice):
        def _str(dev: CUDADevice) -> str:
            return dev.short_str() if dev else "CPU"

        LOGGER.warning(
            "Matrixs are in different devices (%s != %s).\n"
            "Result matrix is going to be on %s.",
            _str(a),
            _str(b),
            _str(dst),
        )

    @staticmethod
    def _log_warning_copying_to_cpu(a: CUDADevice):
        LOGGER.warning("Matrix is on %s, temporarily copying to CPU.", a.short_str())


class ShareDestionationMatrix(contextlib.AbstractContextManager):
    """Context manager to avoid allocating new matrices
    for each operation. Should be used for intensive
    computation that respects the device (i.e., are made
    on the same computational device) and dimensions of
    the target matrix.
    """

    def __init__(self, *src: Matrix):
        """Context constructor.

        Args:
            *src (Matrix): matrices whose operations
                should store results in the same
                destination.
        """
        self._src = src
        first = self._src[0]

        # Assert compatible matrices
        assert all(
            map(
                lambda v: (v.device == first.device)
                and (v.rows == first.rows)
                and (v.columns == first.columns),
                self._src[1:],
            )
        )

        # Initialize destination Matrix with same
        #   dims and device as src[0]
        self._dst = Matrix([[0.0] * first.columns] * first.rows)
        self._dst = self._dst.to(first.device)

        # The destination matrix has itself as
        #   destination for any operation with
        #   it
        self._dst._set_dst(self._dst)

        # Obtain any set dst Matrix
        self._initial_dst = list(map(lambda v: v._dst, self._src))

    def __enter__(self) -> Matrix:
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
