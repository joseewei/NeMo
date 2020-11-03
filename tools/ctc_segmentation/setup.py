from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("nemo_ctc_segmentation.pyx"), include_dirs=[numpy.get_include()])
