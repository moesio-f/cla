import re
from argparse import ArgumentParser
from pathlib import Path

_TYPE_MAPPER = {
    "Vector": "_Vector",
    "Matrix": "_Matrix",
    "double": "c_double",
    "bool": "c_bool",
    "CUDADevice": "_CUDADevice",
}


def main(header_file: Path, output_file: Path):
    # Guarantee files exists
    assert header_file.exists()
    if output_file.exists():
        output_file.unlink()

    # Read header_file
    header_contents = header_file.read_text("utf-8")

    # Create data store
    fn_defs = []
    restype_defs = []

    # Iterate matches of function definitions
    # Beware, this pattern only recognizes some functions
    #   more complex ones should be done manually.
    pattern = re.compile(
        r"(?P<return_type>.+) (?P<fun_name>.+)\((?P<fun_args>(?:\w|\*| |,|\n|\r\n|\r)+)\);"
    )
    for m in pattern.finditer(header_contents):
        return_type = m.group("return_type").strip()
        is_return_pointer = False
        fun_name = m.group("fun_name").strip()
        fun_args = m.group("fun_args").strip()

        # Maybe return type is a pointer?
        if fun_name.startswith("*"):
            is_return_pointer = True
            fun_name = fun_name.strip("*")

        # Parse args
        # fun_args goes from string -> [string] -> [(type, name, is_pointer)]
        fun_args = fun_args.split(",")
        for i in range(len(fun_args)):
            arg_type, arg_name = fun_args[i].strip().split(" ")
            is_arg_pointer = False
            if arg_name.startswith("*"):
                is_arg_pointer = True
                arg_name = arg_name.strip("*")
            fun_args[i] = (arg_type, arg_name, is_arg_pointer)

        # Prepare return_type
        has_return = return_type != "void"
        return_type = _TYPE_MAPPER.get(return_type, return_type)
        return_type = f"POINTER({return_type})" if is_return_pointer else return_type

        # Prepare function args
        def _args_to_fn_def(d: tuple) -> str:
            arg_type, arg_name, is_arg_pointer = d
            arg_type = _TYPE_MAPPER.get(arg_type, arg_type)
            arg_type = f"POINTER({arg_type})" if is_arg_pointer else arg_type
            return f"{arg_name}: {arg_type}"

        fun_args_def = ", ".join(map(_args_to_fn_def, fun_args))

        def _args_to_fn_call(d: tuple) -> str:
            arg_type, arg_name, is_arg_pointer = d
            return f"c_double({arg_name})" if arg_type == "double" else arg_name

        fun_args_call = ", ".join(map(_args_to_fn_call, fun_args))

        # Construct function definition in Python with
        #   ctypes
        fn_defs.append(
            f"def {fun_name} (self, {fun_args_def}){'-> ' + return_type if has_return else ''}:\n"
            f"{'    return ' if has_return else ''}"
            f"self._lib.{fun_name}({fun_args_call})"
        )

        # Construct restype definition
        restype_defs.append(
            f"self._lib.{fun_name}.restype = "
            f"{return_type if has_return else 'None'}"
        )

    fn_defs = "\n\n".join(fn_defs)
    restype_defs = "\n".join(restype_defs)
    output_file.write_text(f"{fn_defs}\n{20*'='}\n{restype_defs}\n", "utf=8")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="cla_header_to_ctypes",
        description="Convert CLA C headers to functions for Python ctypes.",
    )
    parser.add_argument("header", type=Path, help="Source header file.")
    parser.add_argument(
        "-o", "--outfile", type=Path, help="Output file.", default="cla_ctypes.def"
    )
    args = parser.parse_args()

    main(args.header, args.outfile)
