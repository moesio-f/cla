"""
Tests for the Vector
class.
"""

import math
from typing import Callable

import pytest

from pycla import DEVICES, Vector
from pycla.core import CUDADevice, ShareDestionationVector


def _make_operation_fns(operation: str) -> tuple[Callable, Callable, Callable]:
    match operation:
        case "+":

            def add(a, b):
                return a + b

            def iadd(a, b):
                a += b

            return add, add, iadd
        case "-":

            def sub(a, b):
                return a - b

            def isub(a, b):
                a -= b

            return sub, sub, isub
        case "*":

            def mul(a, b):
                return a * b

            def imul(a, b):
                a *= b

            return mul, mul, imul
        case _:
            raise ValueError("Unknown operation.")


@pytest.mark.parametrize(
    "data",
    [
        [0.0] * 100,
        [0] * 100,
        pytest.param(
            ["a", "b", "c"],
            marks=pytest.mark.xfail(
                reason="Vectors should only support float.",
                raises=TypeError,
                strict=True,
            ),
        ),
    ],
)
def test_create_vector(data: list):
    vector = Vector(data)

    # Assertions
    assert vector.device == None
    assert vector.dims == len(data)
    assert list(vector) == data

    # Assert memory is freed
    vector.release()


@pytest.mark.skipif(not DEVICES.has_cuda, reason="CUDA unavailable.")
@pytest.mark.parametrize("dims", [1, 100, 10_000, 100_000])
def test_cuda(dims: int):
    data = [1.0] * dims
    vector = Vector(data)

    # Assert is on CPU
    assert vector.device == None
    assert vector.dims == dims

    # Move to GPU
    device = DEVICES[0]
    vector.to(0)

    # Assert is on GPU
    assert vector.device == device
    assert vector.dims == dims

    # Move back to CPU and assert
    #   data.
    vector.cpu()
    assert vector.device == None
    assert vector.dims == dims
    assert list(vector) == data

    # Assert memory is freed
    vector.release()


@pytest.mark.parametrize(
    "value,dims,new_value",
    [(1.0, 123, 50.0), (-1.0, 514, 232.0), (0.0, 23, -1342.0), (300.0, 78, 921.0)],
)
def test_iterate(value: float, dims: int, new_value: float):
    # Vector initialization
    vector = Vector([value] * dims)

    # Reading
    for v in vector:
        assert v == value

    # Setting
    for i in range(len(vector)):
        vector[i] = new_value

    # Reading
    for v in vector:
        assert v == new_value

    # Assert memory is freed
    vector.release()


@pytest.mark.parametrize(
    "a,b,expected,dims,op,rop,iop",
    [
        (1.0, 2.0, 3.0, 100, *_make_operation_fns("+")),
        (1.0, 2.0, -1.0, 232, *_make_operation_fns("-")),
        (2.5, 1.5, 3.75, 4214, *_make_operation_fns("*")),
    ],
)
def test_vector_binary_operations(
    a: float,
    b: float,
    expected: float,
    dims: int,
    op: Callable[[Vector, Vector], Vector],
    rop: Callable[[Vector, Vector], Vector],
    iop: Callable[[Vector, Vector], None],
):
    # Initialization
    vec_a = Vector([a] * dims)
    vec_b = Vector([b] * dims)
    target = Vector([expected] * dims)

    # Left operation
    assert target == op(vec_a, vec_b)

    # Right operation
    assert target == rop(vec_a, vec_b)

    # Inplace operation
    iop(vec_a, vec_b)
    assert target == vec_a

    # Release
    vec_a.release()
    vec_b.release()
    target.release()


@pytest.mark.parametrize("data", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("dims", [1, 100, 100_000])
@pytest.mark.parametrize("power", [1, 2, 4])
def test_power_vector(data: float, dims: int, power: int):
    # Initialization
    vector = Vector([data] * dims)

    # Find expected value
    expected = data**power

    # Apply operation
    result = vector**power

    # Assert values match
    for v in result:
        assert v == expected

    # Release
    vector.release()
    result.release()


@pytest.mark.parametrize("a,b", [(2.0, 2.0), (2.0, 3.0), (2324.214, 1.21)])
@pytest.mark.parametrize("dims", [1, 10, 1000, 100_000])
def test_dot_product(a: float, b: float, dims: int):
    # Initialize vectors
    vec_a = Vector([a] * dims)
    vec_b = Vector([b] * dims)

    # Find the expected value
    expected = dims * a * b

    # Apply operation
    result = vec_a @ vec_b
    assert round(result, 1) == round(expected, 1)

    # Release
    vec_a.release()
    vec_b.release()


@pytest.mark.parametrize(
    "data,expected_l2,p,expected_lp",
    [([3.0, 4.0], 5.0, 1.0, 7.0), ([10.0] * 64, 80.0, 2.0, 80.0)],
)
def test_norms(data: list[float], p: float, expected_l2: float, expected_lp: float):
    # Initialization
    vector = Vector(data)

    # Assertions
    assert vector.max() == max(data)
    assert vector.l2() == expected_l2
    assert vector.lp(p) == expected_lp

    # Release
    vector.release()


@pytest.mark.parametrize(
    "a,b,expected", [([4.0, 3.0], [2.0, 8.0], [16.0 / 17.0, 64.0 / 17.0])]
)
def test_projection(a: list[float], b: list[float], expected: list[float]):
    # Initialize vectors
    vec_a = Vector(a)
    vec_b = Vector(b)
    vec_expected = Vector(expected)

    # Assertion
    result = vec_a.projection(vec_b)
    assert result == vec_expected

    # Release
    vec_a.release()
    vec_b.release()
    vec_expected.release()
    result.release()


@pytest.mark.parametrize(
    "a,b,expected_rad,expected_deg", [([4.0, 0.0], [0.0, 8.0], math.pi / 2.0, 90.0)]
)
def test_angle(
    a: list[float], b: list[float], expected_rad: float, expected_deg: float
):
    # Initialization
    vec_a = Vector(a)
    vec_b = Vector(b)

    # Assertions
    assert vec_a.angle_to(vec_b) == expected_rad
    assert vec_a.angle_to(vec_b, unit="deg") == expected_deg

    # Release
    vec_a.release()
    vec_b.release()


@pytest.mark.parametrize(
    "a,b,is_orthogonal,is_orthonormal",
    [
        ([4.0, 0.0], [0.0, 8.0], True, False),
        ([1.0, 0.0], [0.0, 1.0], True, True),
        ([1.4, 2.3], [8.23, 10.24], False, False),
    ],
)
def test_ortho(
    a: list[float], b: list[float], is_orthogonal: bool, is_orthonormal: bool
):
    # Initialization
    vec_a = Vector(a)
    vec_b = Vector(b)

    # Assertions
    assert vec_a.is_orthogonal(vec_b) == is_orthogonal
    assert vec_a.is_orthonormal(vec_b) == is_orthonormal

    # Release
    vec_a.release()
    vec_b.release()


@pytest.mark.parametrize("a,b", [(-12.0, 241.023), (123.241, 241791.042)])
@pytest.mark.parametrize("dims", [2, 10, 100, 100_000])
def test_same_destination(a: float, b: float, dims: int):
    # Initialization
    vec_a = Vector([a] * dims)
    vec_b = Vector([b] * dims)

    with ShareDestionationVector(vec_a, vec_b) as dst:
        # Assert initial condition
        vec_dst = dst
        assert Vector.has_shared_data(vec_dst, dst)

        # Do some operations and check results
        dst = vec_a + vec_b
        dst = vec_a * vec_b
        dst = vec_a**2

    # Check final condition
    assert Vector.has_shared_data(vec_dst, dst)

    # Release
    vec_a.release()
    vec_b.release()
    vec_dst.release()
