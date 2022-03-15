from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("run_PPO.pyx", annotate=True)
)