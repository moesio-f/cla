from ctypes import *
from importlib import resources
from pathlib import Path


class _CUDADevice(Structure):
    _fields_ = [
        ("id", c_int),
        ("name", c_char_p),
        ("max_grid_size_x", c_int),
        ("max_grid_size_y", c_int),
        ("max_grid_size_z", c_int),
        ("max_block_size_x", c_int),
        ("max_block_size_y", c_int),
        ("max_block_size_z", c_int),
        ("max_threads_per_block", c_int),
    ]


class _Vector(Structure):
    pass


class _Matrix(Structure):
    pass


_Vector._fields_ = [
    ("arr", POINTER(c_double)),
    ("dims", c_int),
    ("device", POINTER(_CUDADevice)),
    ("cu_vector", POINTER(_Vector)),
]

_Matrix._fields_ = [
    ("arr", POINTER(POINTER(c_double))),
    ("rows", c_int),
    ("columns", c_int),
    ("device", POINTER(_CUDADevice)),
    ("cu_matrix", POINTER(_Matrix)),
]


def _get_latest_binary(
    root: resources.abc.Traversable, lib_name: str
) -> resources.abc.Traversable:
    files = [p for p in root.iterdir() if p.is_file() and p.name.startswith(lib_name)]
    return sorted(files, reverse=True)[0]


class _CLA:
    LIB_NAME: str = "libcla.so"
    LIBCLA_PATH: Path = _get_latest_binary(resources.files("pycla.bin"), LIB_NAME)

    def __init__(self):
        # Initialize library
        self._lib = CDLL(self.LIBCLA_PATH)

        # Set functions restypes and argtypes
        self._set_functions_types()

        # Populate CUDA devices
        self._lib.populate_devices()

    @property
    def version(self) -> str:
        version = self.LIBCLA_PATH.name.replace(self.LIB_NAME + ".", "")
        version = f"v{version if version else '0.0.0'}"
        return version

    @property
    def cuda_device_count(self) -> int:
        return self._lib.cuda_get_device_count()

    @property
    def has_cuda(self) -> bool:
        return self._lib.has_cuda()

    def get_device_by_id(self, id: int) -> POINTER(_CUDADevice):
        return self._lib.get_device_by_id(id)

    def get_device_by_name(self, name: bytes) -> POINTER(_CUDADevice):
        return self._lib.get_device_by_name(name)

    def vector_to_cu(
        self, src: POINTER(_Vector), device: POINTER(_CUDADevice)
    ) -> POINTER(_Vector):
        return self._lib.vector_to_cu(src, device)

    def vector_to_cpu(self, src: POINTER(_Vector)) -> POINTER(_Vector):
        return self._lib.vector_to_cpu(src)

    def matrix_to_cu(
        self, src: POINTER(_Matrix), device: POINTER(_CUDADevice)
    ) -> POINTER(_Matrix):
        return self._lib.matrix_to_cu(src, device)

    def matrix_to_cpu(self, src: POINTER(_Matrix)) -> POINTER(_Matrix):
        return self._lib.matrix_to_cpu(src)

    def const_vector(
        self, dims: int, value: float, device: POINTER(_CUDADevice)
    ) -> POINTER(_Vector):
        return self._lib.const_vector(dims, c_double(value), device)

    def copy_vector(
        self, src: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.copy_vector(src, dst)

    def destroy_vector(self, vector: POINTER(_Vector)):
        self._lib.destroy_vector(vector)

    def const_matrix(
        self, rows: int, columns: int, value: float, device: POINTER(_CUDADevice)
    ) -> POINTER(_Matrix):
        return self._lib.const_matrix(rows, columns, c_double(value), device)

    def copy_matrix(
        self, src: POINTER(_Matrix), dst: POINTER(_Matrix)
    ) -> POINTER(_Matrix):
        return self._lib.copy_matrix(src, dst)

    def destroy_matrix(self, matrix: POINTER(_Matrix)):
        self._lib.destroy_matrix(matrix)

    def vector_add(
        self, a: POINTER(_Vector), b: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.vector_add(a, b, dst)

    def vector_sub(
        self, a: POINTER(_Vector), b: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.vector_sub(a, b, dst)

    def vector_mult_scalar(
        self, a: float, b: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.vector_mult_scalar(c_double(a), b, dst)

    def vector_projection(
        self, a: POINTER(_Vector), b: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.vector_projection(a, b, dst)

    def vector_element_wise_prod(
        self, a: POINTER(_Vector), b: POINTER(_Vector), dst: POINTER(_Vector)
    ) -> POINTER(_Vector):
        return self._lib.vector_element_wise_prod(a, b, dst)

    def vector_dot_product(self, a: POINTER(_Vector), b: POINTER(_Vector)) -> float:
        return self._lib.vector_dot_product(a, b)

    def vector_lp_norm(self, a: POINTER(_Vector), p: float) -> float:
        return self._lib.vector_lp_norm(a, c_double(p))

    def vector_max_norm(self, a: POINTER(_Vector)) -> float:
        return self._lib.vector_max_norm(a)

    def vector_l2_norm(self, a: POINTER(_Vector)) -> float:
        return self._lib.vector_l2_norm(a)

    def vector_angle_between_rad(
        self, a: POINTER(_Vector), b: POINTER(_Vector)
    ) -> float:
        return self._lib.vector_angle_between_rad(a, b)

    def vector_equals(self, a: POINTER(_Vector), b: POINTER(_Vector)) -> bool:
        return self._lib.vector_equals(a, b)

    def vector_orthogonal(self, a: POINTER(_Vector), b: POINTER(_Vector)) -> bool:
        return self._lib.vector_orthogonal(a, b)

    def vector_orthonormal(self, a: POINTER(_Vector), b: POINTER(_Vector)) -> bool:
        return self._lib.vector_orthonormal(a, b)

    def matrix_add(
        self, a: POINTER(_Matrix), b: POINTER(_Matrix), dst: POINTER(_Matrix)
    ) -> POINTER(_Matrix):
        return self._lib.matrix_add(a, b, dst)

    def matrix_sub(
        self, a: POINTER(_Matrix), b: POINTER(_Matrix), dst: POINTER(_Matrix)
    ) -> POINTER(_Matrix):
        return self._lib.matrix_sub(a, b, dst)

    def matrix_mult(
        self, a: POINTER(_Matrix), b: POINTER(_Matrix), dst: POINTER(_Matrix)
    ) -> POINTER(_Matrix):
        return self._lib.matrix_mult(a, b, dst)

    def matrix_mult_scalar(
        self, a: float, b: POINTER(_Matrix), dst: POINTER(_Matrix)
    ) -> POINTER(_Matrix):
        return self._lib.matrix_mult_scalar(c_double(a), b, dst)

    def matrix_trace(self, a: POINTER(_Matrix)) -> float:
        return self._lib.matrix_trace(a)

    def matrix_lpq_norm(self, a: POINTER(_Matrix), p: float, q: float) -> float:
        return self._lib.matrix_lpq_norm(a, c_double(p), c_double(q))

    def matrix_frobenius_norm(self, a: POINTER(_Matrix)) -> float:
        return self._lib.matrix_frobenius_norm(a)

    def matrix_equals(self, a: POINTER(_Matrix), b: POINTER(_Matrix)) -> bool:
        return self._lib.matrix_equals(a, b)

    def _set_functions_types(self):
        self._lib.has_cuda.restype = c_bool
        self._lib.get_device_by_id.restype = POINTER(_CUDADevice)
        self._lib.get_device_by_name.restype = POINTER(_CUDADevice)
        self._lib.vector_to_cu.restype = POINTER(_Vector)
        self._lib.vector_to_cpu.restype = POINTER(_Vector)
        self._lib.matrix_to_cu.restype = POINTER(_Matrix)
        self._lib.matrix_to_cpu.restype = POINTER(_Matrix)
        self._lib.const_vector.restype = POINTER(_Vector)
        self._lib.copy_vector.restype = POINTER(_Vector)
        self._lib.destroy_vector.restype = None
        self._lib.const_matrix.restype = POINTER(_Matrix)
        self._lib.copy_matrix.restype = POINTER(_Matrix)
        self._lib.destroy_matrix.restype = None
        self._lib.vector_add.restype = POINTER(_Vector)
        self._lib.vector_sub.restype = POINTER(_Vector)
        self._lib.vector_mult_scalar.restype = POINTER(_Vector)
        self._lib.vector_projection.restype = POINTER(_Vector)
        self._lib.vector_element_wise_prod.restype = POINTER(_Vector)
        self._lib.vector_dot_product.restype = c_double
        self._lib.vector_lp_norm.restype = c_double
        self._lib.vector_max_norm.restype = c_double
        self._lib.vector_l2_norm.restype = c_double
        self._lib.vector_angle_between_rad.restype = c_double
        self._lib.vector_equals.restype = c_bool
        self._lib.vector_orthogonal.restype = c_bool
        self._lib.vector_orthonormal.restype = c_bool
        self._lib.matrix_add.restype = POINTER(_Matrix)
        self._lib.matrix_sub.restype = POINTER(_Matrix)
        self._lib.matrix_mult.restype = POINTER(_Matrix)
        self._lib.matrix_mult_scalar.restype = POINTER(_Matrix)
        self._lib.matrix_trace.restype = c_double
        self._lib.matrix_lpq_norm.restype = c_double
        self._lib.matrix_frobenius_norm.restype = c_double
        self._lib.matrix_equals.restype = c_bool

    def __del__(self):
        self._lib.clear_devices()

    @staticmethod
    def points_to_same_location(a: POINTER, b: POINTER) -> bool:
        a_ = cast(a, c_void_p).value
        b_ = cast(b, c_void_p).value
        return a_ == b_


CLA = _CLA()
