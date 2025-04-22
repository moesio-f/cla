"""
Tests for the Matrix
class.
"""

import math
from typing import Callable

import pytest

from pycla import DEVICES, Matrix
from pycla.core import CUDADevice, ShareDestionationMatrix


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
        case _:
            raise ValueError("Unknown operation.")


@pytest.mark.parametrize(
    "data",
    [
        [[0.0] * 100] * 100,
        [[0] * 100] * 100,
        pytest.param(
            [["a", "b", "c"]] * 10,
            marks=pytest.mark.xfail(
                reason="Matrix should only support float.",
                raises=TypeError,
                strict=True,
            ),
        ),
    ],
)
def test_create_matrix(data: list):
    matrix = Matrix(data)

    # Assertions
    assert matrix.device == None
    assert matrix.rows == len(data)
    assert matrix.columns == len(data[0])
    assert matrix[:, :] == data

    # Assert memory is freed
    matrix.release()


@pytest.mark.skipif(not DEVICES.has_cuda, reason="CUDA unavailable.")
@pytest.mark.parametrize("rows", [1, 1_000])
@pytest.mark.parametrize("columns", [1, 1_000])
def test_cuda(rows: int, columns: int):
    data = [[1.0] * columns] * rows
    matrix = Matrix(data)

    # Assert is on CPU
    assert matrix.device == None
    assert matrix.rows == rows
    assert matrix.columns == columns

    # Move to GPU
    device = DEVICES[0]
    matrix.to(0)

    # Assert is on GPU
    assert matrix.device == device
    assert matrix.rows == rows
    assert matrix.columns == columns

    # Move back to CPU and assert
    #   data.
    matrix.cpu()
    assert matrix.device == None
    assert matrix.rows == rows
    assert matrix.columns == columns
    assert matrix[:, :] == data

    # Assert memory is freed
    matrix.release()


@pytest.mark.parametrize(
    "value,rows,columns,new_value",
    [
        (1.0, 123, 421, 50.0),
        (-1.0, 21, 514, 232.0),
        (0.0, 65, 23, -1342.0),
        (300.0, 124, 78, 921.0),
    ],
)
def test_iterate(value: float, rows: int, columns: int, new_value: float):
    # Matrix initialization
    matrix = Matrix([[value] * columns] * columns)

    # Reading
    for row in matrix:
        for v in row:
            assert v == value

    # Setting
    for i in range(matrix.rows):
        for j in range(matrix.columns):
            matrix[i, j] = new_value

    # Reading
    for row in matrix:
        for v in row:
            assert v == new_value

    # Assert memory is freed
    matrix.release()


@pytest.mark.parametrize(
    "a,b,expected,dims,op,rop,iop",
    [
        (1.0, 2.0, 3.0, 100, *_make_operation_fns("+")),
        (1.0, 2.0, -1.0, 232, *_make_operation_fns("-")),
    ],
)
def test_matrix_binary_operations(
    a: float,
    b: float,
    expected: float,
    dims: int,
    op: Callable[[Matrix, Matrix], Matrix],
    rop: Callable[[Matrix, Matrix], Matrix],
    iop: Callable[[Matrix, Matrix], None],
):
    # Initialization
    mat_a = Matrix([[a] * dims] * dims)
    mat_b = Matrix([[b] * dims] * dims)
    target = Matrix([[expected] * dims] * dims)

    # Left operation
    assert target == op(mat_a, mat_b)

    # Right operation
    assert target == rop(mat_a, mat_b)

    # Inplace operation
    iop(mat_a, mat_b)
    assert target == mat_a

    # Release
    mat_a.release()
    mat_b.release()
    target.release()


@pytest.mark.parametrize("a,b", [(2.0, 2.0), (2.0, 3.0), (2324.214, 1.21)])
@pytest.mark.parametrize("dims", [1, 10, 100])
def test_mat_mul(a: float, b: float, dims: int):
    # Initialize vectors
    mat_a = Matrix([[a] * dims] * dims)
    mat_b = Matrix([[b] * dims] * dims)
    expected = Matrix([[a * b * dims] * dims] * dims)

    # Apply operation
    result = mat_a @ mat_b
    assert expected == result

    # Release
    mat_a.release()
    mat_b.release()
    expected.release()
    result.release()


@pytest.mark.parametrize(
    "data,expected_frob,p,q,expected_lpq",
    [
        ([[3.0, 0.0], [4.0, 0.0]], 5.0, 1.0, 1.0, 7.0),
        ([[10.0] * 8] * 8, 80.0, 2.0, 2.0, 80.0),
    ],
)
def test_norms(
    data: list[list[float]],
    expected_frob: float,
    p: float,
    q: float,
    expected_lpq: float,
):
    # Initialization
    matrix = Matrix(data)

    # Find expected trace
    trace = 0.0
    for i in range(matrix.rows):
        for j in range(matrix.columns):
            if i == j:
                trace += matrix[i, j]

    # Assertions
    assert matrix.frobenius() == expected_frob
    assert matrix.lpq(p, q) == expected_lpq
    assert matrix.trace() == trace

    # Release
    matrix.release()


@pytest.mark.parametrize("a,b", [(-12.0, 241.023), (123.241, 241791.042)])
@pytest.mark.parametrize("dims", [2, 10, 100, 1_000])
def test_same_destination(a: float, b: float, dims: int):
    # Initialization
    mat_a = Matrix([[a] * dims] * dims)
    mat_b = Matrix([[b] * dims] * dims)

    with ShareDestionationMatrix(mat_a, mat_b) as dst:
        # Assert initial condition
        mat_dst = dst
        assert Matrix.has_shared_data(mat_dst, dst)

        # Do some operations and check results
        dst = mat_a + mat_b
        dst = mat_a - mat_b
        dst = mat_a * 2

    # Check final condition
    assert Matrix.has_shared_data(mat_dst, dst)

    # Release
    mat_a.release()
    mat_b.release()
    mat_dst.release()
