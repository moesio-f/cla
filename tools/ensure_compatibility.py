"""Script que realiza instalação
e execução dos testes da biblioteca
com múltiplas versões Python e com
as dependências mínimas e máximas.
"""

import shutil
import subprocess
from pathlib import Path

VERSIONS = ["3.10", "3.11", "3.12"]
ENV_PREFIX = ".compat.venv.py{}"
SCRIPT_DIR = Path(__file__).parent
PROJ_DIR = SCRIPT_DIR.joinpath("..").resolve()


def prepare_envinroment(version: str) -> str:
    venv_path = str(PROJ_DIR.joinpath(ENV_PREFIX.format(version.replace(".", ""))))
    for command in [
        ["uv", "venv", venv_path, "--python", version, "--seed"],
        [f"{venv_path}/bin/python", "-m", "pip", "install", "--upgrade", "pip", "uv", "pytest"],
    ]:
        subprocess.run(command, check=True)

    return venv_path


def install_lib(venv: str):
    command = [f"{venv}/bin/python", "-m", "uv", "pip", "install", "."]
    subprocess.run(command, check=True)


def run_pytest(venv: str):
    command = [f"{venv}/bin/python", "-m", "pytest", "-x"]
    subprocess.run(command, check=True)


def run_examples(venv: str):
    bin = f"{venv}/bin/"
    for command in [
        [f"{bin}/python", "-m", "uv", "pip", "install", "papermill", "jupyterlab"],
        *[
            [f"{bin}/papermill", f, "out.ipynb"]
            for f in list(PROJ_DIR.joinpath("examples").glob("*.ipynb"))
        ],
    ]:
        subprocess.run(command, check=True)

    # Cleanup
    PROJ_DIR.joinpath("out.ipynb").unlink(missing_ok=True)


def cleanup_environments():
    glob = f"{ENV_PREFIX.format('', '')}*/"
    for p in PROJ_DIR.glob(glob):
        if not p.is_dir():
            continue

        assert ".compat.venv.py" in p.name
        shutil.rmtree(p)
        p.unlink(missing_ok=True)


def main():
    # Maybe some environments where not cleanup
    cleanup_environments()

    for v in VERSIONS:
        # Prepare environment
        venv = prepare_envinroment(v)

        # Install packages
        install_lib(venv)

        # Run pytest
        run_pytest(venv)

        # Run sample codes
        run_examples(venv)

        # Cleanup if no error
        cleanup_environments()


if __name__ == "__main__":
    main()
