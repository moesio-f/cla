import operator
import time
from argparse import ArgumentParser
from pathlib import Path

from pycla import Matrix, Vector
from pycla.core import ShareDestionationMatrix, ShareDestionationVector


def python_loop_add(is_matrix: bool):
    if is_matrix:

        def _op(src: list[list[float]], dst: list[list[float]]):
            for i in range(len(src)):
                for j in range(len(src[0])):
                    dst[i][j] = src[i][j] + src[i][j]

    else:

        def _op(src: list[float], dst: list[float]):
            for i in range(len(src)):
                dst[i] = src[i] + src[i]

    return _op


def python_loop_sub(is_matrix: bool):
    if is_matrix:

        def _op(src: list[list[float]], dst: list[list[float]]):
            for i in range(len(dst)):
                for j in range(len(dst[0])):
                    dst[i][j] = src[i][j] - src[i][j]

    else:

        def _op(src: list[float], dst: list[float]):
            for i in range(len(src)):
                dst[i] = src[i] - src[i]

    return _op


def python_loop_prod(is_matrix: bool):
    if is_matrix:

        def _op(src: list[list[float]], dst: list[list[float]]):
            for i in range(len(src)):
                for j in range(len(src[0])):
                    sum = 0.0
                    for k in range(len(src[0])):
                        sum += src[i][k] * src[k][j]
                    dst[i][j] = sum

    else:

        def _op(src: list[float], dst: list[float]):
            for i in range(len(src)):
                dst[i] = src[i] * src[i]

    return _op


def run_python_benchmark(
    n_runs: int,
    dim_start: int,
    dim_end: int,
    dim_step: int,
    lines: list,
    is_matrix: bool = False,
):
    for dim in range(dim_start, dim_end + 1, dim_step):
        src = [1.0] * dim
        dst = [0.0] * dim

        if is_matrix:
            src = [src] * dim
            dst = [dst] * dim

        for run in range(n_runs):
            for op in [python_loop_add, python_loop_sub, python_loop_prod]:
                start = time.perf_counter()
                op(is_matrix)(src, dst)
                end = time.perf_counter()

                name = (
                    name
                    if "prod" not in (name := op.__name__)
                    else f"python_loop_{'matmul' if is_matrix else 'element_wise_prod'}"
                )
                kind = "Matrix" if is_matrix else "Vector"
                lines.append(f"{name},{kind},{dim},{run},{end - start:.10f}")


def run_pycla_benchmark(
    n_runs: int,
    dim_start: int,
    dim_end: int,
    dim_step: int,
    lines: list,
    is_matrix: bool = False,
):
    ent_cls = Matrix if is_matrix else Vector
    ctx_cls = ShareDestionationMatrix if is_matrix else ShareDestionationVector
    for dim in range(dim_start, dim_end + 1, dim_step):
        src = [1.0] * dim
        if is_matrix:
            src = [src] * dim

        src = ent_cls(src)
        with ctx_cls(src) as dst:
            for run in range(n_runs):
                for op in [
                    operator.add,
                    operator.sub,
                    operator.matmul if is_matrix else operator.mul,
                ]:
                    start = time.perf_counter()
                    op(src, src)
                    end = time.perf_counter()

                    name = "pycla_cpu_" + (
                        name if (name := op.__name__) != "mul" else "element_wise_prod"
                    )
                    kind = "Matrix" if is_matrix else "Vector"
                    lines.append(f"{name},{kind},{dim},{run},{end - start:.10f}")

        # Release memory
        src.release()
        dst.release()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="pycla_benchmark", description="Run benchmark tests for pycla."
    )
    parser.add_argument(
        "n_runs", type=int, help="Number of executions for each operation."
    )
    parser.add_argument(
        "dim_start",
        type=int,
        help="Initial dimension for tests (for matrix, rows=columns).",
    )
    parser.add_argument(
        "dim_end", type=int, help="End dimension for tests (inclusive)."
    )
    parser.add_argument("dim_step", type=int, help="Steps to generate each dimension.")

    # Parse arguments
    args = parser.parse_args()
    n_runs = args.n_runs
    dim_start = args.dim_start
    dim_end = args.dim_end
    dim_step = args.dim_step

    # Create output file
    output = Path("pycla_benchmark.csv")
    lines = ["name,kind,dims,run,time"]

    # Run Pure Python benchmarks
    run_python_benchmark(n_runs, dim_start, dim_end, dim_step, lines, is_matrix=False)
    run_python_benchmark(n_runs, dim_start, dim_end, dim_step, lines, is_matrix=True)

    # Run PyCLA benchmarks
    run_pycla_benchmark(n_runs, dim_start, dim_end, dim_step, lines, is_matrix=False)
    run_pycla_benchmark(n_runs, dim_start, dim_end, dim_step, lines, is_matrix=True)

    # Write all lines to output
    output.write_text("\n".join(lines), "utf-8")
