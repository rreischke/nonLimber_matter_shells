from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("levinpower",
        ["src/Levin_power.cpp", "python/pybind11_interface.cpp"],
        cxx_std=11,
        include_dirs=["src"],
        libraries=["m", "gsl", "gslcblas"],
        extra_compile_args=["-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        ),
]

setup(
    name="levinpower",
    version=__version__,
    #author="Robert Reischke",
    #author_email="s",
    #url="",
    #description="",
    #long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
)