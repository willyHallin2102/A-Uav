#!/usr/bin/env bash


# Setting up error management during the installation
#
#   - A command exists with a non-zero status `-e`
#   - An undefined variable is being used `-u`
#   - A pipeline fails anywhere `-o pipefail`
#
set -euo pipefail


# Python version and name of the virtual environment
#   Check if version also exist
VENV=".venv"
PYTHON="${1:-python3.11}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON was not found"
    exit 1
fi

echo "Using: $("$PYTHON" --version)"


# Building the Virtual Environment.
#
#   Making the environment with python3.11
#   Activates it, pip assign all to the 3rd party library to .venv
#   Installs basic tooling systems

"$PYTHON" -m venv "$VENV"

#
#   -- shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# The core libraries
python -m pip install                   \
    "tensorflow[and-cuda]==2.20.0"      \
    "numpy<2.0"                         \
    scipy                               \
    pyarrow                             \
    polars                              \
    numba                               \
    matplotlib                          \
    seaborn                             \
    orjson                              \
    einops                              \
    h5py                                \
    tensorboard


# Sionna
#   From the website downloading into the environment using 
#   git command
python -m pip install git+https://github.com/NVlabs/sionna.git@main
python -m pip freeze > requirements.txt

